"""
Causal-graph layer built on top of the Knowledge Graph (keywords_extraction.py,
neo4j_storage.py — both unmodified by this module).

Three responsibilities, each independent of the others:
  1. Node-quality filtering (is_valid_node, _clean_node) — a reusable text-
     quality check for whether a KG node string looks like a real research
     variable rather than a method name or a sentence fragment.
  2. LLM causal-relation extraction (extract_causal_relations) — a second
     LLM pass over KG edges that filters for genuine causal claims and
     discards correlation/taxonomy relations.
  3. Statistical causal discovery orchestration (run_causal_discovery) and
     PyVis rendering (generate_causal_graph) — runs the causal-learn/
     tigramite/lingam/torch method implementations in causal_discovery.py
     against a real dataset, independent of any LLM-derived graph.
"""
import re
import json
import csv
import time
import ollama
from pyvis.network import Network

from causal_discovery import prettify_node_name

try:
    import wordninja
except ImportError:
    wordninja = None

# Words that indicate a node is a sentence fragment from the paper, not a variable
_SENTENCE_STARTERS = {
    'abstract', 'this', 'the', 'we', 'our', 'in', 'here', 'study',
    'paper', 'article', 'introduction', 'results', 'conclusion',
    'figure', 'table', 'section', 'however', 'therefore', 'although',
    'based', 'using', 'used', 'proposed', 'present', 'show', 'shown',
    'demonstrate', 'investigate', 'analyse', 'analyze', 'evaluate',
}

# Computational / ML / method terms that are never research variables
_TECH_BLOCKLIST = {
    # ML model families
    'machine learning', 'deep learning', 'neural network', 'neural networks',
    'random forest', 'decision tree', 'decision trees', 'gradient boosting',
    'xgboost', 'lightgbm', 'catboost', 'support vector machine', 'svm',
    'naive bayes', 'k-nearest neighbor', 'knn',
    # DL architectures
    'lstm', 'gru', 'rnn', 'cnn', 'transformer', 'bert', 'gpt',
    'convolutional neural network', 'recurrent neural network',
    'multilayer perceptron', 'mlp', 'autoencoder', 'gan', 'vae',
    'attention mechanism', 'self-attention', 'encoder', 'decoder',
    # Training / optimisation
    'backpropagation', 'gradient descent', 'stochastic gradient',
    'learning rate', 'batch size', 'epoch', 'epochs', 'hyperparameter',
    'dropout', 'regularization', 'batch normalization', 'layer normalization',
    'overfitting', 'underfitting', 'early stopping', 'weight decay',
    # Data / feature engineering
    'feature extraction', 'feature selection', 'feature engineering',
    'dimensionality reduction', 'principal component analysis', 'pca',
    'data augmentation', 'data preprocessing', 'normalization',
    'train test split', 'cross validation', 'k-fold',
    'transfer learning', 'fine-tuning', 'pre-training',
    # Evaluation metrics
    'accuracy', 'precision', 'recall', 'f1 score', 'f1-score',
    'rmse', 'mae', 'mse', 'r-squared', 'auc', 'roc curve',
    'mean absolute error', 'root mean square error', 'mean squared error',
    'confusion matrix', 'classification report',
    # General computational terms
    'algorithm', 'model architecture', 'hyperparameter tuning',
    'embedding', 'tokenization', 'vectorization',
    'inference', 'prediction model', 'forecasting model', 'models', 'model',
}

# Causal-discovery method-name acronyms (this same codebase's own
# causal_discovery.py implements these) -- never a physical research
# variable, but matched as a WHOLE WORD only (not substring, unlike
# _TECH_BLOCKLIST above), since several are short enough that naive
# substring containment would false-positive on ordinary words
# (e.g. "ges" is a substring of "changes"/"images"/"stages").
_METHOD_NAME_ACRONYMS = {
    'pc', 'pcmci', 'pcmci+', 'fci', 'cdnod', 'lingam', 'tcdf',
    'daggnn', 'dag-gnn', 'lpcmci', 'ges',
}

# Generic discourse/structural terms that are never causal variables
_GENERIC_BLOCKLIST = {
    'results', 'result', 'method', 'methods', 'methodology',
    'dataset', 'data set', 'data', 'analysis', 'analyses',
    'study area', 'study site', 'approach', 'objective', 'objectives',
    'discussion', 'background', 'overview', 'summary', 'conclusion',
    'experiment', 'experiments', 'finding', 'findings',
    'observation', 'observations', 'literature', 'review',
}

# KG relation types that describe taxonomy/structure, not causation
_NON_CAUSAL_RELATIONS = {
    'IS_A', 'PART_OF', 'IS_PART_OF', 'RELATED_TO', 'MEASURED_BY',
    'USED_FOR', 'IS_USED_FOR', 'CORRELATES_WITH', 'ASSOCIATED_WITH',
    'SIMILAR_TO', 'DEFINED_AS',
}

_LEADING_ARTICLE_RE = re.compile(r'^(the|a|an)\s+', re.IGNORECASE)

# Below this length, a spaceless token is treated as a legitimate short word
# or acronym (e.g. 'sst', 'co2') and never sent through _resplit_if_squished —
# wordninja has no dictionary entry for most domain acronyms and will force a
# bad split on them (e.g. 'prectotcorr' -> ['pre', 'c', 'to', 't', 'corr']).
_MIN_RESPLIT_LEN = 10


def _resplit_if_squished(t: str) -> str:
    """
    pdfplumber (keywords_extraction.py's PDF text extraction) occasionally
    loses the space between adjacent words, so a KG node can arrive here as
    'neuralnetwork' or 'seaiceextent' instead of 'neural network' / 'sea ice
    extent'. That silently defeats _TECH_BLOCKLIST's substring matching
    (which looks for 'neural network' WITH a space) and makes the node
    unreadable in the UI. Re-split spaceless text with wordninja's
    frequency-weighted segmenter, but only keep the result if every fragment
    is at least 2 characters — a single-letter fragment means wordninja was
    forced into a low-confidence split on text it doesn't recognize, so the
    original is kept as-is rather than risking a worse mangling.
    """
    if wordninja is None or ' ' in t or len(t) < _MIN_RESPLIT_LEN:
        return t
    parts = wordninja.split(t)
    if len(parts) < 2 or any(len(p) < 2 for p in parts):
        return t
    return ' '.join(parts)


def _clean_node(text: str) -> str:
    """Canonical form for a node, used for validation, whitelist matching, and dedup
    so 'the Sea Ice Extent' and 'sea ice  extent' collapse to the same string.
    Also re-splits spaceless text lost to PDF-extraction artifacts (see
    _resplit_if_squished)."""
    t = text.strip().lower()
    t = _LEADING_ARTICLE_RE.sub('', t)
    t = re.sub(r'\s+', ' ', t)
    t = _resplit_if_squished(t.strip())
    return t.strip()


def _normalize_relation(rel: str) -> str:
    """Canonical form for a relation label, e.g. 'increases'/'INCREASES'/'Increases' -> 'INCREASES'."""
    return re.sub(r'[^A-Z0-9_]', '_', rel.strip().upper()).strip('_')


def _collapse_contained_extras(extra_nodes: set, edge_nodes: set) -> set:
    """
    Drop edge-less candidate variables (extra_nodes, from dataset
    extraction) that are fully contained, word-for-word, inside another
    surviving whitelist entry -- e.g. "sea ice" and "arctic sea ice" both
    collapse away if "arctic sea ice extent" is also present, since all
    three describe the same physical quantity at different levels of
    specificity. Hyphens are treated the same as spaces for this comparison
    only (e.g. "ocean atmospheric" vs "ocean-atmospheric variables") -- the
    original string is still what is kept and displayed.

    Never drops an edge_node: those have a real KG edge and must remain
    referenceable in the KNOWLEDGE GRAPH EDGES section of the prompt, even
    if a longer extra_node also happens to contain it.
    """
    all_nodes = edge_nodes | extra_nodes
    normalized = {n: n.replace('-', ' ') for n in all_nodes}

    kept = set()
    for n in extra_nodes:
        n_norm = normalized[n]
        subsumed = any(
            other != n and normalized[other] != n_norm
            and re.search(r'\b' + re.escape(n_norm) + r'\b', normalized[other])
            for other in all_nodes
        )
        if not subsumed:
            kept.add(n)
    return kept


def is_valid_node(text: str) -> bool:
    """Return True only if a node looks like a research variable, not a method or sentence.

    Public (not underscore-prefixed) because it is reused outside this
    module's own LLM causal-extraction path: any UI that builds a variable-
    mapping list directly from raw KG nodes (rather than from an already-
    LLM-vetted causal-relation list) can import this to filter out
    unfiltered method/generic terms the same way.
    """
    t = _clean_node(text)
    if not t:
        return False

    # Reject anything longer than 60 characters
    if len(t) > 60:
        return False

    # Reject anything with more than 5 words
    words = t.split()
    if len(words) > 5:
        return False

    # Reject obvious parsing artifacts: a bare comma inside a single
    # candidate string (e.g. "gc,c", "x t,c") is never a legitimate variable
    # name -- these come from garbled LaTeX/notation extraction during the
    # dataset-description LLM pass, not real measured quantities.
    if ',' in t:
        return False

    # Reject if the first word is a known sentence-starter
    if words[0] in _SENTENCE_STARTERS:
        return False

    # Reject if node text matches or contains any technical/method/generic term
    if any(term in t for term in _TECH_BLOCKLIST) or any(term in t for term in _GENERIC_BLOCKLIST):
        return False

    # Reject causal-discovery method-name acronyms, matched as a whole word
    if any(w in _METHOD_NAME_ACRONYMS for w in words):
        return False

    return True


def extract_causal_relations(kg_edges: list, model: str = "mistral", extra_variables: list = None) -> list:
    """
    Pass 2: receives KG edges (list of dicts with source/relation/target/score)
    and asks the LLM to identify which ones are causal.

    Three layers of protection against noise:
      1. Pre-filter: drop any edge whose source or target fails is_valid_node()
      2. Prompt constraint: send the LLM a whitelist of valid node names
      3. Post-validate: discard any parsed CAUSE/EFFECT not in the whitelist

    extra_variables: optional additional candidate variable names with no
    known KG edge of their own (e.g. dataset-extraction variables) -- added
    to the whitelist so the LLM can also consider causal relationships
    involving them, alongside the variables that do have a KG edge.
    _parse_causal_output() validates purely against the whitelist (not
    against which edges were given), so no other change is needed for these
    to be accepted if the LLM proposes a pair involving one.

    These dataset-only variables are given deliberate emphasis, not just
    listed alongside the rest: the prompt presents them as their own
    labeled, prioritized group (ahead of the KG-edge variables), and if the
    first response doesn't use any of them, one additional targeted retry
    asks the model to specifically reconsider them before the function
    returns. See the prompt construction and the post-parse retry block
    below for the exact mechanism.

    Cross-source matching: any extra_variable whose cleaned text exactly
    matches an existing KG-edge node (matched_nodes) is surfaced as its own
    highest-priority group in the prompt, separate from both the dataset-
    only group and the plain KG-edge group -- two independent extractions
    (the paper's running text and its dataset description) agreeing on a
    variable is stronger evidence than either alone. This is purely a
    prompt-presentation change; matched_nodes were already part of the
    whitelist via edge_nodes before this grouping existed.

    Returns list of dicts: {cause, effect, label, confidence}
    """
    print(f"\nUsing LLM model for causal extraction: {model}")
    print(f"Raw KG edges received: {len(kg_edges)}")

    if not kg_edges:
        return []

    # ── Layer 1: pre-filter noisy edges ──────────────────────────────────
    clean_edges = [
        e for e in kg_edges
        if is_valid_node(e.get('source', '')) and is_valid_node(e.get('target', ''))
    ]
    print(f"After noise filter: {len(clean_edges)}/{len(kg_edges)} edges kept "
          f"({len(kg_edges) - len(clean_edges)} dropped as non-variable text)")

    if not clean_edges:
        print("⚠️  All KG edges were filtered out as noisy. Check keyword extraction quality.")
        return []

    # ── Layer 1.5: drop non-causal relation types + normalize + dedupe ────
    seen = set()
    deduped = []
    for e in clean_edges:
        rel = _normalize_relation(e.get('relation', ''))
        if rel in _NON_CAUSAL_RELATIONS:
            continue

        src = _clean_node(e['source'])
        tgt = _clean_node(e['target'])
        key = (src, rel, tgt)
        if key not in seen:
            seen.add(key)
            deduped.append({'source': src, 'relation': rel, 'target': tgt, 'score': e.get('score', 0)})

    print(f"After dropping non-causal relation types + dedup: {len(deduped)} edge(s) remain")

    if not deduped:
        print("⚠️  All edges were non-causal relation types (IS_A, PART_OF, etc.).")
        return []

    # ── Build valid node whitelist (already canonical from dedup step) ────
    edge_nodes = {e['source'] for e in deduped} | {e['target'] for e in deduped}

    # Extra candidate variables (e.g. from dataset extraction) with no known
    # KG edge of their own -- cleaned/validated the same way edge endpoints
    # are, then unioned into the whitelist so the LLM can consider them too.
    # Any dataset variable whose cleaned text exactly matches an existing
    # edge_node is cross-source confirmation -- the paper's running text
    # (KG relation) and its dataset description independently agree this is
    # a real physical variable. Tracked separately as matched_nodes so the
    # prompt can call these out as the strongest candidates, instead of
    # silently folding them into edge_nodes with no distinction.
    extra_nodes = set()
    matched_nodes = set()
    if extra_variables:
        cleaned_extra = {_clean_node(v) for v in extra_variables if is_valid_node(v)}
        matched_nodes = cleaned_extra & edge_nodes
        extra_nodes = cleaned_extra - edge_nodes
        extra_nodes = _collapse_contained_extras(extra_nodes, edge_nodes)
        if extra_nodes:
            print(f"Extra candidate variables (no known KG edge): {len(extra_nodes)}")
        if matched_nodes:
            print(f"Variables confirmed by BOTH KG relation and dataset description: "
                  f"{len(matched_nodes)} -> {sorted(matched_nodes)}")

    valid_nodes = sorted(edge_nodes | extra_nodes)
    valid_nodes_lower = set(valid_nodes)
    print(f"Valid variable whitelist ({len(valid_nodes)}): {', '.join(valid_nodes[:15])}"
          f"{' ...' if len(valid_nodes) > 15 else ''}")
    print("\nKG edges sent to LLM for causal analysis:")
    for e in deduped:
        print(f"  ({e['source']}, {e['relation']}, {e['target']})")

    # Three labeled groups, most-confirmed first:
    #   1. matched_nodes -- present in BOTH the KG relations and the dataset
    #      description (two independent sources agree). Highest priority.
    #   2. Dataset-only variables -- own labeled, prioritized group, listed
    #      before the plain KG-edge variables (models tend to weight early
    #      prompt content more heavily), with directive rather than
    #      permissive language -- "actively check" instead of "you may
    #      propose". Unchanged from the earlier dataset-priority pass.
    #   3. Remaining KG-edge variables (excluding anything already shown in
    #      group 1, to avoid listing the same name twice).
    dataset_only_nodes = sorted(extra_nodes)
    kg_edge_nodes = sorted(edge_nodes - matched_nodes)
    dataset_only_lower = {n.lower() for n in dataset_only_nodes}

    matched_block = ""
    if matched_nodes:
        matched_block = (
            "Variables confirmed by BOTH the paper's text (knowledge-graph relation)\n"
            "AND its dataset description — HIGHEST PRIORITY. Two independent sources\n"
            "agree these are real physical variables, so give them your strongest\n"
            "consideration for a causal pair:\n"
            + "\n".join(f"  - {n}" for n in sorted(matched_nodes))
            + "\n\n"
        )

    dataset_block = (
        "Measured dataset variables — PRIORITIZE these. Actively check EACH ONE\n"
        "below for a plausible causal link to any other variable in this prompt,\n"
        "even if no knowledge-graph edge below mentions it. These come from the\n"
        "paper's own dataset description, not just its running text, so they are\n"
        "real measured quantities:\n"
        + "\n".join(f"  - {n}" for n in dataset_only_nodes)
        if dataset_only_nodes
        else "Measured dataset variables: (none identified for this paper)"
    )
    kg_nodes_str = "\n".join(f"  - {n}" for n in kg_edge_nodes)
    edges_str = "\n".join(
        f"({e['source']}, {e['relation']}, {e['target']})"
        for e in deduped
    )

    # ── Layer 2: constrained prompt ───────────────────────────────────────
    prompt = f"""You are an expert in causal reasoning for climate, Arctic, and Earth sciences.

DOMAIN CONTEXT: This paper is from climate / environmental / Earth science research.
You are identifying causal relationships between PHYSICAL and ENVIRONMENTAL variables only.

VALID VARIABLE NAMES (copy VERBATIM — do not paraphrase, abbreviate, or invent):

{matched_block}{dataset_block}

Variables with an existing knowledge-graph relation:
{kg_nodes_str}

KNOWLEDGE GRAPH EDGES to analyse:
{edges_str}

TASK: Identify edges where one physical/environmental variable directly causes, drives, triggers, or produces change in another. Give the measured dataset variables listed above special attention — they are real quantities from the paper's own data and are strong causal candidates even without a knowledge-graph edge already connecting them.

IMPORTANT — CORRELATION IS NOT CAUSATION: Two variables can rise and fall together,
or simply co-occur in the same sentence, without one causing the other. Only output a
pair if you can identify a plausible physical mechanism by which the cause produces the
effect. If you are only confident the two are associated/correlated, do NOT output them.

ACCEPT pairs where CAUSE and EFFECT are:
- Physical quantities (temperature, pressure, salinity, humidity)
- Environmental phenomena (sea ice, permafrost, albedo, precipitation)
- Biogeochemical processes (carbon flux, evaporation, photosynthesis)
- Climate indices or forcings (radiative forcing, ENSO, heat flux)
- Measured dataset variables (SST, SIC, SLP, OHC, AOD)

REJECT any pair where CAUSE or EFFECT is:
- A computational method (machine learning, neural network, LSTM, regression)
- A model architecture or algorithm name
- A statistical technique or evaluation metric (RMSE, accuracy, R-squared)
- A software tool, framework, or dataset name (not a variable measured by it)
- A general research activity (training, prediction, classification)
- Merely correlated or co-occurring without a clear causal mechanism

For each accepted causal pair output EXACTLY (one blank line between blocks):

CAUSE: <exact string from the list>
EFFECT: <exact string from the list>
LABEL: <mechanism in UPPERCASE_WITH_UNDERSCORES>
CONFIDENCE: <0.0 to 1.0>

Rules:
- CAUSE and EFFECT must be VERBATIM from the list above
- You may reverse direction if the relation implies it (e.g. "B CAUSED_BY A" → CAUSE=A, EFFECT=B)
- LABEL examples: DRIVES, CAUSES, LEADS_TO, TRIGGERS, AMPLIFIES, INHIBITS, ACCELERATES, REDUCES, WARMS, MELTS, INCREASES, DECREASES
- CONFIDENCE: 1.0 = direct physical mechanism | 0.6 = established link | 0.3 = indirect / uncertain
- Skip: CORRELATES_WITH, IS_A, MEASURED_BY, RELATED_TO, IS_USED_FOR, IS_PART_OF
- Output ONLY the blocks — no explanations, headers, or extra text

Now identify all causal relationships:
"""

    def _uses_dataset_variable(rels):
        return any(
            r['cause'].lower() in dataset_only_lower or r['effect'].lower() in dataset_only_lower
            for r in rels
        )

    try:
        print(f"\n🔄 Sending {len(deduped)} edge(s) to {model} for causal reasoning...")
        content = _call_ollama_with_retry(model, prompt)
        print(f"\n✅ LLM response received ({len(content)} chars):")
        print(content)

        causal_rels = _parse_causal_output(content, valid_nodes_lower, min_confidence=0.4)

        if not causal_rels and content.strip():
            print("⚠️  No valid causal pairs parsed on first pass — retrying with a stricter format reminder.")
            repair_prompt = prompt + (
                "\n\nREMINDER: Output ONLY CAUSE/EFFECT/LABEL/CONFIDENCE blocks, exactly as specified above. "
                "No prose, no markdown, no numbering, no extra commentary."
            )
            content = _call_ollama_with_retry(model, repair_prompt)
            print(f"\n✅ LLM retry response received ({len(content)} chars):")
            print(content)
            causal_rels = _parse_causal_output(content, valid_nodes_lower, min_confidence=0.4)

        # Dataset-focus retry: if measured dataset variables were available
        # but the accepted pairs don't use any of them, ask once more,
        # specifically about that group. Merges any new pairs in rather
        # than replacing the first pass's results, and never discards a
        # pair already accepted.
        if dataset_only_nodes and not _uses_dataset_variable(causal_rels):
            print("⚠️  No accepted pair used a measured dataset variable — retrying with a dataset-focused reminder.")
            focus_prompt = prompt + (
                "\n\nREMINDER: Your previous answer did not use any of the MEASURED DATASET "
                "VARIABLES listed above. Review that list again and identify at least one "
                "additional plausible causal pair involving one of them, if one genuinely "
                "exists — do not force one if none is physically justified. Output in the "
                "same CAUSE/EFFECT/LABEL/CONFIDENCE format, only the new pair(s)."
            )
            focus_content = _call_ollama_with_retry(model, focus_prompt)
            print(f"\n✅ Dataset-focus retry response received ({len(focus_content)} chars):")
            print(focus_content)
            focus_rels = _parse_causal_output(focus_content, valid_nodes_lower, min_confidence=0.4)

            existing_keys = {(r['cause'].lower(), r['effect'].lower()) for r in causal_rels}
            added = 0
            for r in focus_rels:
                key = (r['cause'].lower(), r['effect'].lower())
                if key not in existing_keys:
                    causal_rels.append(r)
                    existing_keys.add(key)
                    added += 1
            print(f"   Dataset-focus retry added {added} new pair(s).")

        # Independent, computed confidence signal -- the LLM's own CONFIDENCE
        # field is self-reported (see _parse_causal_output), not verified
        # against anything, and clusters near 1.0 regardless of whether the
        # pair is actually sound. embedding_confidence is a second, separate
        # number computed the same way Stage 1's KG relations already are
        # (cosine similarity between the embedding of "cause label" and
        # "effect"), added alongside -- not in place of -- the LLM's own
        # confidence, so both are visible and neither silently overrides
        # the other.
        for r in causal_rels:
            r['embedding_confidence'] = _embedding_confidence(r['cause'], r['label'], r['effect'])

        print(f"\n📊 Causal Extraction Summary:")
        print(f"   Accepted causal pairs: {len(causal_rels)}")
        for r in causal_rels:
            print(f"   ({r['cause']}) --{r['label']}--> ({r['effect']})  "
                  f"[LLM confidence: {r['confidence']}, computed (embedding) confidence: {r['embedding_confidence']}]")

        return causal_rels
    except Exception as e:
        print(f"Causal extraction error: {e}")
        return []


_embedding_model = None


def _get_embedding_model():
    """Lazily loads and caches the sentence-embedding model, mirroring
    keywords_extraction.py's use of the same model for Stage 1's KG-relation
    confidence scores -- loaded lazily here (not at module import) so this
    module's cost stays low for callers that never touch causal-relation
    scoring, consistent with how causal_discovery.py lazily imports its own
    heavy/optional dependencies."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _embedding_confidence(cause: str, label: str, effect: str) -> float:
    """
    Independent, computed confidence signal for an LLM-proposed causal pair:
    cosine similarity between the embedding of "cause label" and the
    embedding of "effect" -- the exact same formula
    keywords_extraction.confidence_scores() already uses for Stage 1's KG
    relations, applied here to Stage 2's causal-LLM output.

    This is NOT a replacement for the LLM's own self-reported CONFIDENCE
    field (_parse_causal_output already parses that): it is a second,
    non-self-referential number to sanity-check it against, since the LLM's
    own confidence is generated by the same call that produced the pair and
    has no independent basis -- it clusters near 1.0 regardless of whether
    the pair is actually sound (e.g. a hallucinated pair can self-report
    CONFIDENCE: 1.0 just as easily as a correct one).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    model = _get_embedding_model()
    emb = model.encode([f"{cause} {label}", effect])
    return round(float(cosine_similarity([emb[0]], [emb[1]])[0][0]), 3)


def _call_ollama_with_retry(model: str, prompt: str, max_retries: int = 2, seed: int = 42) -> str:
    """Calls Ollama, retrying on transient failures.

    temperature=0 + a fixed seed make the causal-reasoning call reproducible
    across runs: the same KG edges/whitelist/prompt now produce the same
    accepted causal pairs every time, instead of a different subset each
    run under Ollama's default (non-zero-temperature) sampling. This is the
    one call site shared by both the LLM Causal Graph section and the
    Enhanced Structural Causal Graph section, so both become reproducible
    together.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "seed": seed},
            )
            return response['message']['content']
        except Exception as e:
            last_err = e
            print(f"⚠️  Ollama call failed (attempt {attempt}/{max_retries}): {e}")
    raise RuntimeError(f"Ollama call failed after {max_retries} attempts: {last_err}")


def _parse_causal_output(
    text: str,
    valid_nodes_lower: set | None = None,
    min_confidence: float = 0.4,
) -> list:
    """
    Parse the LLM causal output into structured dicts.
    Discards pairs whose nodes are not in valid_nodes_lower (hallucinations),
    and pairs below min_confidence (weak/uncertain links).
    """
    results = []
    blocks = re.split(r'\n\s*\n', text.strip())

    for block in blocks:
        cause = effect = label = None
        confidence = 0.5

        for line in block.strip().split('\n'):
            line = line.strip()
            key = line.upper()
            if key.startswith('CAUSE:'):
                cause = line.split(':', 1)[1].strip()
            elif key.startswith('EFFECT:'):
                effect = line.split(':', 1)[1].strip()
            elif key.startswith('LABEL:'):
                raw = line.split(':', 1)[1].strip()
                label = re.sub(r'[^A-Z0-9_]', '_', raw.upper()).strip('_')
            elif key.startswith('CONFIDENCE:'):
                match = re.search(r'[\d.]+', line.split(':', 1)[1])
                if match:
                    try:
                        confidence = max(0.0, min(1.0, float(match.group())))
                    except ValueError:
                        confidence = 0.5

        if not (cause and effect and label):
            continue

        # Post-validate against whitelist
        if valid_nodes_lower is not None:
            if cause.lower() not in valid_nodes_lower or effect.lower() not in valid_nodes_lower:
                print(f"⚠️  Rejected hallucinated pair: '{cause}' → '{effect}'")
                continue

        # Skip self-loops
        if cause.lower() == effect.lower():
            continue

        # Drop low-confidence pairs (likely method/correlation noise)
        if confidence < min_confidence:
            print(f"⚠️  Rejected low-confidence pair ({confidence}): '{cause}' → '{effect}'")
            continue

        results.append({
            'cause': cause,
            'effect': effect,
            'label': label,
            'confidence': round(confidence, 2)
        })

    return results


def _confidence_to_color(confidence: float) -> str:
    if confidence >= 0.8:
        return '#C0392B'
    elif confidence >= 0.6:
        return '#E74C3C'
    elif confidence >= 0.4:
        return '#E67E22'
    else:
        return '#F39C12'


def generate_causal_graph(causal_relations: list, output_path: str = "causal_graph.html") -> tuple:
    """
    Build a directed PyVis causal graph from causal relation dicts.
    Returns (Network, html_string).
    """
    net = Network(height="500px", width="100%", directed=True,
                  bgcolor="#0D1117", font_color="#FFFFFF")

    net.set_options("""{
        "physics": {
            "enabled": true,
            "repulsion": {
                "nodeDistance": 250,
                "springLength": 300,
                "springConstant": 0.04
            },
            "solver": "repulsion"
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 1.2}},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
            "font": {"size": 11, "color": "#CCCCCC", "strokeWidth": 0}
        },
        "nodes": {
            "font": {"size": 13, "color": "#FFFFFF"},
            "borderWidth": 2
        }
    }""")

    if not causal_relations:
        return net, ""

    cause_count = {}
    effect_count = {}
    for rel in causal_relations:
        cause_count[rel['cause']] = cause_count.get(rel['cause'], 0) + 1
        effect_count[rel['effect']] = effect_count.get(rel['effect'], 0) + 1

    all_nodes = set(r['cause'] for r in causal_relations) | set(r['effect'] for r in causal_relations)

    for node in all_nodes:
        is_cause = node in cause_count
        is_effect = node in effect_count

        if is_cause and is_effect:
            color = {"background": "#8E44AD", "border": "#6C3483"}
        elif is_cause:
            color = {"background": "#C0392B", "border": "#922B21"}
        else:
            color = {"background": "#E67E22", "border": "#CA6F1E"}

        size = 20 + (cause_count.get(node, 0) * 5) + (effect_count.get(node, 0) * 3)
        size = min(size, 50)

        # Display only — net.add_node's first arg (the node ID) and add_edge()
        # below both stay on the raw canonical `node`/`rel['cause']`/`rel['effect']`
        # strings so edges keep matching up; only label=/title= text is prettified.
        causes_list = ", ".join(prettify_node_name(r['effect']) for r in causal_relations if r['cause'] == node)
        caused_by_list = ", ".join(prettify_node_name(r['cause']) for r in causal_relations if r['effect'] == node)
        tooltip = f"<b>{prettify_node_name(node)}</b>"
        if causes_list:
            tooltip += f"<br>Causes: {causes_list}"
        if caused_by_list:
            tooltip += f"<br>Caused by: {caused_by_list}"

        net.add_node(node, label=prettify_node_name(node), color=color, size=size, title=tooltip)

    for rel in causal_relations:
        edge_color = _confidence_to_color(rel['confidence'])
        width = round(1 + rel['confidence'] * 3, 1)
        net.add_edge(
            rel['cause'],
            rel['effect'],
            label=rel['label'],
            color=edge_color,
            width=width,
            title=f"{rel['label']} (confidence: {rel['confidence']})"
        )

    net.save_graph(output_path)
    with open(output_path, "r") as f:
        html_str = f.read()
    return net, html_str


def export_causal_relations(causal_relations: list, path: str = "causal_relations.json", fmt: str = "json") -> None:
    """Save causal relations to disk for debugging, reporting, or cross-paper comparison."""
    if fmt == "json":
        with open(path, "w") as f:
            json.dump(causal_relations, f, indent=2)
    elif fmt == "csv":
        if not causal_relations:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=causal_relations[0].keys())
            writer.writeheader()
            writer.writerows(causal_relations)
    else:
        raise ValueError(f"Unsupported fmt: {fmt!r}, expected 'json' or 'csv'")


def run_causal_discovery(
    dataset_df,
    mapping: dict,
    methods: list = ('pc',),
    pc_alpha: float = 0.05,
    tau_max: int = 5,
) -> dict:
    """
    Runs one or more statistical causal-discovery methods on a real uploaded
    dataset to build a causal graph directly from the data — independent of
    any LLM-derived causal graph.

    mapping: {kg_node: dataset_column_or_None} — built by
             causal_discovery.suggest_variable_mapping(kg_nodes, dataset_columns)
             and confirmed/edited by the user before this is called.
    methods: any of 'pc', 'pcmci', 'pcmci_plus', 'tcdf', 'lingam', 'fci', 'cdnod', 'daggnn',
             'lpcmci', 'ges'.

    Returns: {
        'columns_used': [...],
        'statistical_edges': {method_name: [edge dicts in dataset-column space]},
        'kg_edges': {method_name: [edge dicts in KG-node space, ready for
                     generate_causal_graph()]},
    }
    """
    from causal_discovery import (
        run_pc, run_pcmci, run_pcmci_plus, run_tcdf, run_lingam, run_fci, run_cdnod, run_dag_gnn,
        run_lpcmci, run_ges, edges_to_kg_space
    )

    mapped_columns = sorted({c for c in mapping.values() if c})
    if len(mapped_columns) < 2:
        raise ValueError("Need at least 2 KG nodes mapped to dataset columns to run causal discovery.")

    runners = {
        'pc': lambda: run_pc(dataset_df, mapped_columns, alpha=pc_alpha),
        'pcmci': lambda: run_pcmci(dataset_df, mapped_columns, tau_max=tau_max, alpha=pc_alpha),
        'pcmci_plus': lambda: run_pcmci_plus(dataset_df, mapped_columns, tau_max=tau_max, alpha=pc_alpha),
        'tcdf': lambda: run_tcdf(dataset_df, mapped_columns, tau_max=tau_max),
        'lingam': lambda: run_lingam(dataset_df, mapped_columns),
        'fci': lambda: run_fci(dataset_df, mapped_columns, alpha=pc_alpha),
        'cdnod': lambda: run_cdnod(dataset_df, mapped_columns, alpha=pc_alpha),
        'daggnn': lambda: run_dag_gnn(dataset_df, mapped_columns),
        # capped below run_pcmci/run_pcmci_plus's shared tau_max -- LPCMCI's
        # runtime scales badly with it (~5s at tau_max=2 vs. ~104s at
        # tau_max=5 on a 6-var/1826-row dataset, verified in
        # cd_algorithm/LPCMCI.ipynb), so a user picking tau_max=5 for
        # PCMCI+/TCDF shouldn't also make LPCMCI take minutes unasked.
        'lpcmci': lambda: run_lpcmci(dataset_df, mapped_columns, tau_max=min(tau_max, 2), alpha=pc_alpha),
        'ges': lambda: run_ges(dataset_df, mapped_columns),
    }

    statistical_edges = {}
    kg_edges = {}
    print(f"\n📊 Causal Discovery: {len(methods)} method(s) on {len(mapped_columns)} mapped variable(s): {mapped_columns}")
    for method in methods:
        if method not in runners:
            raise ValueError(f"Unknown method: {method!r}, expected one of {list(runners)}")
        print(f"\n▶ Starting {method.upper()}...")
        t0 = time.time()
        edges = runners[method]()
        elapsed = time.time() - t0
        print(f"✅ {method.upper()} done in {elapsed:.1f}s — {len(edges)} edge(s) found")
        statistical_edges[method] = edges
        kg_edges[method] = edges_to_kg_space(edges, mapping)

    return {
        'columns_used': mapped_columns,
        'statistical_edges': statistical_edges,
        'kg_edges': kg_edges,
    }
