"""
Keyword-extraction wrapper for the LLM Enhanced Structural Causal Graph (SCG)
pipeline.

This module never modifies or duplicates keywords_extraction.py,
variable_filter.py, or causal_graph.py. It reuses
keywords_extraction.extract_keywords() and keywords_extraction.process()
as-is, and only intervenes at the keyword-selection step.

Two problems it addresses, without changing the underlying KG pipeline:

  1. Keywords-section blind spot. When a paper declares an explicit
     "Keywords:" list, extract_keywords() returns that list directly and
     never scans the paper body — so a real variable mentioned only in the
     body text is never considered as a candidate.
  2. Overlong malformed candidates. The Keywords-section parser joins
     multiple collected lines and splits only on comma-type delimiters, with
     no length check, which can occasionally produce a single ~30-word
     "keyword".

Design: extract_keywords() is called once, unmodified, to check whether a
paper actually has a declared Keywords: section.
  - If it does not (the algorithmic TF-IDF/YAKE/KeyBERT path), the result is
    returned as-is. Widening the candidate pool on this path was tested and
    found to change results for the worse (see enhanced_extract_keywords()'s
    docstring), so this path is intentionally left untouched.
  - If it does, a supplementary algorithmic pass is run over the full body
    text (with the trigger word masked so it can't re-detect the same
    section) to recover body-only variables, merged with the declared list,
    deduplicated, and trimmed of overlong candidates.

The final keyword count returned is always capped at the original k — only
which candidates fill those k slots changes.

Also provides extract_dataset_variables(), which supplies dataset-extraction
variables to causal_graph.extract_causal_relations()'s extra_variables
parameter instead of rendering them as raw KG nodes.
"""
import re
from unittest.mock import patch

import keywords_extraction
from keywords_extraction import extract_keywords as _original_extract_keywords

MAX_KEYWORD_WORDS = 8
WIDE_K_MULTIPLIER = 3


def _mask_keywords_trigger(input_text):
    """Replace the literal word 'keyword(s)' so a second extract_keywords()
    call cannot re-detect a Keywords: section, forcing it down the
    algorithmic (TF-IDF + YAKE + KeyBERT) path over the full body text."""
    return re.sub(r"(?i)keywords?", "topics", input_text)


def _dedupe_preserve_order(keywords):
    seen = set()
    result = []
    for kw in keywords:
        key = kw.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(kw)
    return result


def _drop_long_candidates(keywords, max_words=MAX_KEYWORD_WORDS):
    return [kw for kw in keywords if len(kw.split()) <= max_words]


def _declared_vocab(declared_keywords):
    vocab = set()
    for kw in declared_keywords:
        vocab.update(kw.lower().split())
    return vocab


def _drop_declared_fragments(candidates, declared_vocab):
    """Drop supplementary candidates made entirely of words already present
    in the declared Keywords: list (e.g. "ice"/"sea" out of "sea ice").
    These high-term-frequency fragments otherwise dominate the algorithmic
    pass's ranking, crowding genuinely new body-text variables (e.g. "ocean
    heat", "snow cover") out of the final k slots."""
    kept = []
    for kw in candidates:
        words = kw.lower().split()
        if words and all(w in declared_vocab for w in words):
            continue
        kept.append(kw)
    return kept


def enhanced_extract_keywords(input_text, k):
    """
    Drop-in replacement for keywords_extraction.extract_keywords() — same
    (input_text, k) signature, same return dict shape (keywords /
    from_keywords_section / total_found).

    Only widens the candidate pool when a Keywords: section is actually
    detected — the one case with a real gap to fix (the shortcut skips the
    paper body entirely). On the algorithmic path (no declared Keywords:
    section), extract_keywords() scores candidates via scale(), a softmax
    over however many candidates it is given — asking for 3x more candidates
    therefore reranks the same top terms rather than just enlarging the
    pool, which can surface low-value single-word terms (e.g. "causal",
    "model", "pcmci") above genuine variables. The algorithmic path is
    therefore called with the plain k and its result returned untouched,
    matching plain keywords_extraction.process()'s output exactly.
    """
    primary = _original_extract_keywords(input_text, k)

    if not primary.get('from_keywords_section'):
        return primary

    # A Keywords: section was found, which means the body text was never
    # scanned. Widen now and run a supplementary algorithmic pass over the
    # full text (trigger word masked) to recover body-text variables too.
    # Widening here carries none of the reranking risk described above: the
    # declared-keywords branch returns a literal parsed list before any
    # TF-IDF/YAKE/KeyBERT scoring runs, so re-fetching it with more slots is
    # a cheap re-parse, not a rescoring of the primary result.
    k_wide = max(k, k * WIDE_K_MULTIPLIER)
    primary = _original_extract_keywords(input_text, k_wide)
    candidates = list(primary['keywords'])
    masked_text = _mask_keywords_trigger(input_text)
    supplementary = _original_extract_keywords(masked_text, k_wide)
    declared_vocab = _declared_vocab(candidates)
    new_terms = _drop_declared_fragments(supplementary['keywords'], declared_vocab)
    candidates += new_terms

    candidates = _dedupe_preserve_order(candidates)
    candidates = _drop_long_candidates(candidates)

    total_found = len(candidates)
    final_keywords = candidates[:k]

    return {
        'keywords': final_keywords,
        'from_keywords_section': True,
        'total_found': total_found,
        'method': primary.get('method', 'Enhanced (widened + merged)'),
        'enhanced': True,
    }


def process_enhanced(file_path, k, filter_variables=True, llm_model="mistral", use_gpt4_datasets=False):
    """
    Same signature and return shape as keywords_extraction.process(). Runs
    the real, unmodified process(), with its internal extract_keywords()
    call temporarily swapped for enhanced_extract_keywords(). process()'s
    own orchestration — variable filtering, chunking, relation/dataset
    extraction, scoring, KG node/edge assembly — is untouched and never
    duplicated here.
    """
    with patch.object(keywords_extraction, 'extract_keywords', enhanced_extract_keywords):
        return keywords_extraction.process(
            file_path, k,
            filter_variables=filter_variables,
            llm_model=llm_model,
            use_gpt4_datasets=use_gpt4_datasets,
        )


def extract_dataset_variables(nodes, datasets):
    """
    Collect dataset-extraction variables (from process_enhanced()'s own
    `datasets` return value) that are not already present in the raw KG node
    list, filtered to real climate variables via VariableFilter (unchanged,
    reused from variable_filter.py). Intended to be passed as
    causal_graph.extract_causal_relations()'s `extra_variables` parameter,
    so the LLM causal-reasoning step can consider a dataset-only variable
    even though it never earned a KG relation of its own.

    This is the current approach for surfacing dataset variables to the
    pipeline; an earlier approach rendered them directly as raw KG nodes
    fanned out from a single generic "Datasets" hub, which lost the
    information of which specific dataset each variable actually came from.
    """
    from variable_filter import VariableFilter
    vf = VariableFilter()

    existing_nodes_lower = {n.lower().strip() for n in nodes}
    seen = set()
    result = []
    for dataset_info in datasets or []:
        for var in dataset_info.get('variables', []):
            var = var.strip()
            var_lower = var.lower()
            if not var or var_lower in existing_nodes_lower or var_lower in seen:
                continue
            if vf.is_variable(var):
                seen.add(var_lower)
                result.append(var)
    return result
