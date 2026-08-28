"""
Statistical causal discovery on a real uploaded dataset — an independent,
data-driven counterpart to the LLM-derived causal graph produced by
causal_graph.py.

Four groups of functionality:
  1. Dataset loading and inspection (load_dataset, detect_time_column,
     numeric_columns).
  2. KG-node <-> dataset-column name matching (suggest_variable_mapping and
     its supporting alias/fuzzy-matching helpers).
  3. Display-only name formatting (prettify_node_name, prettify_column_name)
     — never used on values that serve as dict keys or DataFrame column
     references, only on text rendered to the screen.
  4. Ten causal-discovery method implementations (run_pc, run_fci, run_cdnod,
     run_lingam, run_pcmci, run_pcmci_plus, run_tcdf, run_dag_gnn,
     run_lpcmci, run_ges), each returning edges in dataset-column space, plus
     edges_to_kg_space() to map a method's result back to KG-node space.
     run_pcmci_full / run_pcmci_plus_full run the identical computation as
     run_pcmci / run_pcmci_plus but additionally return the raw Tigramite
     graph/val_matrix/p_matrix arrays, for callers that want to render the
     full multivariate causal graph (every link type, not just the resolved
     edges) via plot_full_causal_graph().

Kept in its own file so the existing KG / Q&A / Causal Graph pipeline
(keywords_extraction.py, neo4j_storage.py, qa_module.py, causal_graph.py's
existing functions) is never touched by this module. Heavy/optional
dependencies (causallearn, tigramite, torch, lingam) are imported lazily
inside each run_* function, so a missing or broken package only disables
that one method instead of breaking the rest of the app.
"""
import re
import time
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

try:
    import wordninja
except ImportError:
    wordninja = None

# ── Algorithm Glossary ──────────────────────────────────────────────────
# One line per causal-discovery method implemented in this file, its
# category, and its source citation. This is the canonical reference for
# these ten methods across the whole Causal_graph module — other files
# (e.g. causal_graph.py's run_causal_discovery()) point back here instead
# of repeating this list.
#
#   PC        constraint-based   Spirtes & Glymour, 1991 (i.i.d.)
#   FCI       constraint-based   Spirtes, Meek & Richardson, 1999 (i.i.d.,
#                                tolerates unmeasured confounders)
#   CD-NOD    constraint-based   Huang et al., 2020 (i.i.d., flags
#                                nonstationary variables)
#   LiNGAM    functional         Shimizu et al., 2006 ("DirectLiNGAM"
#                                variant used here; i.i.d.)
#   GES       score-based        Chickering, 2002 (i.i.d.)
#   DAG-GNN   continuous-opt.    Yu et al., ICML 2019 (i.i.d.)
#   PCMCI     constraint-based   Runge et al., Science Advances 2019
#                                (time series, lagged links only)
#   PCMCI+    constraint-based   Runge, UAI 2020 (time series, lagged +
#                                contemporaneous links)
#   TCDF      neural             Nauta, Bucur & Seifert, 2019 (time
#                                series, depthwise-causal CNN + attention)
#   LPCMCI    constraint-based   Gerhardus & Runge, NeurIPS 2020 (time
#                                series, tolerates unmeasured confounders —
#                                the lagged counterpart to FCI)
#
# Common parameters across most methods below:
#   pc_alpha / alpha   significance threshold for the conditional-
#                      independence tests the constraint-based methods
#                      (PC/FCI/CD-NOD/PCMCI family) run — lower values
#                      require stronger evidence before keeping an edge.
#   tau_max            (PCMCI family, TCDF, LPCMCI only) the maximum time
#                      lag, in rows, considered as a possible cause — e.g.
#                      tau_max=5 tests whether each variable up to 5 rows
#                      earlier causally affects the current row.
# ── Dataset loading & inspection ────────────────────────────────────────

def load_dataset(uploaded_file) -> pd.DataFrame:
    """Reads an uploaded CSV into a DataFrame. Raises ValueError on bad input."""
    df = pd.read_csv(uploaded_file)
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")
    if len(df.select_dtypes(include=[np.number]).columns) < 2:
        raise ValueError("Dataset needs at least 2 numeric columns for causal discovery.")
    return df


def detect_time_column(df: pd.DataFrame) -> str | None:
    """Best-effort detection of a date/time column — determines whether
    time-series methods (PCMCI, PCMCI+) are offered as options."""
    candidates = [c for c in df.columns if re.search(r'date|time|timestamp|year|month|day', c, re.IGNORECASE)]
    for c in candidates:
        try:
            pd.to_datetime(df[c])
            return c
        except (ValueError, TypeError):
            continue
    return None


def numeric_columns(df: pd.DataFrame) -> list:
    """Column names with a numeric dtype — the only columns a user can map
    to a KG node, since every causal-discovery method here requires numeric
    input."""
    return list(df.select_dtypes(include=[np.number]).columns)


# ── Variable mapping (KG/CG node <-> dataset column) ───────────────────────

_LEADING_ARTICLE_RE = re.compile(r'^(the|a|an)\s+', re.IGNORECASE)

# Below this length, a spaceless token is treated as a legitimate short word
# or acronym (e.g. 'sst', 'co2') and never sent through _resplit_if_squished —
# wordninja has no dictionary entry for most domain acronyms and will force a
# bad split on them (e.g. 'prectotcorr' -> ['pre', 'c', 'to', 't', 'corr']).
_MIN_RESPLIT_LEN = 10


def _resplit_if_squished(t: str) -> str:
    """
    pdfplumber (keywords_extraction.py's PDF text extraction) occasionally
    loses the space between adjacent words, so a raw KG node can arrive here
    as 'neuralnetwork' or 'seaiceextent' instead of 'neural network' / 'sea
    ice extent', which also breaks fuzzy-matching against dataset columns.
    Re-split spaceless text with wordninja's frequency-weighted segmenter,
    but only keep the result if every fragment is at least 2 characters — a
    single-letter fragment means wordninja was forced into a low-confidence
    split on text it doesn't recognize, so the original is kept as-is rather
    than risking a worse mangling. Deliberately duplicated from
    causal_graph._resplit_if_squished rather than imported, to keep this
    module independent of causal_graph.py (see module docstring).
    """
    if wordninja is None or ' ' in t or len(t) < _MIN_RESPLIT_LEN:
        return t
    parts = wordninja.split(t)
    if len(parts) < 2 or any(len(p) < 2 for p in parts):
        return t
    return ' '.join(parts)


def clean_node_name(text: str) -> str:
    """
    Canonical display form for a raw KG node name — strips leading articles,
    collapses whitespace, lowercases. Raw KG nodes (keywords_extraction.py)
    only get .lower().strip(), so without this, "the Sea Ice Extent" and
    "sea ice  extent" show up as two different entries in the variable-mapping
    dropdown instead of one. Also re-splits spaceless text lost to PDF-
    extraction artifacts (see _resplit_if_squished). Deliberately duplicated
    from causal_graph._clean_node rather than imported, to keep this module
    independent of causal_graph.py (see module docstring).
    """
    t = text.strip().lower()
    t = _LEADING_ARTICLE_RE.sub('', t)
    t = re.sub(r'\s+', ' ', t)
    t = _resplit_if_squished(t.strip())
    return t.strip()


# Common climate/meteorological variable abbreviations -> full-name phrases.
# Checked before plain fuzzy matching: acronyms rarely fuzzy-match their own
# full-name phrase well on raw string similarity alone (e.g. "T2M" against
# "surface temperature", or "SIC" against "sea ice concentration").
_VARIABLE_ALIASES = {
    't2m': ['surface temperature', 'air temperature', '2m temperature', 'temperature'],
    'sst': ['sea surface temperature'],
    'sic': ['sea ice concentration'],
    'sie': ['sea ice extent'],
    'rh2m': ['relative humidity', '2m relative humidity', 'humidity'],
    'ws2m': ['wind speed', '2m wind speed'],
    'prectotcorr': ['precipitation', 'total precipitation'],
    'slp': ['sea level pressure'],
    'ohc': ['ocean heat content'],
    'aod': ['aerosol optical depth'],
    'co2': ['carbon dioxide', 'co2 concentration'],
    'ch4': ['methane', 'ch4 concentration'],

    # Additional common Earth-science / reanalysis dataset short names
    # (ERA5, NASA POWER, NOAA-style column headers), so a wider range of
    # dataset uploads has a chance of auto-mapping without a new alias
    # entry being added on the spot. Not exhaustive — extend as new
    # datasets surface unmatched short names.

    # Atmospheric / reanalysis
    'lw_down': ['longwave radiation', 'downward longwave radiation', 'incoming longwave radiation'],
    'sw_down': ['shortwave radiation', 'downward shortwave radiation', 'incoming shortwave radiation'],
    'ssrd': ['surface solar radiation downwards', 'shortwave radiation'],
    'strd': ['surface thermal radiation downwards', 'longwave radiation'],
    'ssr': ['surface net solar radiation', 'net shortwave radiation'],
    'str': ['surface net thermal radiation', 'net longwave radiation'],
    'sshf': ['surface sensible heat flux', 'sensible heat flux'],
    'slhf': ['surface latent heat flux', 'latent heat flux'],
    'd2m': ['dew point temperature', '2m dew point temperature'],
    'u10': ['10m wind u component', 'eastward wind', 'zonal wind'],
    'v10': ['10m wind v component', 'northward wind', 'meridional wind'],
    'si10': ['10m wind speed', 'wind speed'],
    'wd10m': ['wind direction'],
    # Root form recovered by _strip_height_suffix() from columns like
    # "wind_10m" -- lets a sensor-height-tagged column still hit this
    # alias even though the literal column string is never in this dict.
    'wind': ['wind velocity', 'wind speed'],
    'sp': ['surface pressure'],
    'ps': ['surface pressure'],
    'msl': ['mean sea level pressure'],
    'tp': ['total precipitation', 'precipitation'],
    'tcc': ['total cloud cover', 'cloud cover'],
    'tcwv': ['total column water vapour', 'water vapour', 'precipitable water'],
    'qv2m': ['specific humidity', '2m specific humidity'],

    # Ocean / marine
    'sss': ['sea surface salinity'],
    'ssh': ['sea surface height', 'sea level'],
    'mld': ['mixed layer depth'],
    'sit': ['sea ice thickness'],

    # Cryosphere / land
    'swe': ['snow water equivalent'],
    'sm': ['soil moisture'],

    # Carbon cycle / atmospheric composition (co2/ch4 above)
    'n2o': ['nitrous oxide'],
    'o3': ['ozone'],
    'pm25': ['particulate matter 2.5', 'pm2.5'],
    'pm10': ['particulate matter 10'],

    # Vegetation / land surface
    'ndvi': ['normalized difference vegetation index', 'vegetation index'],
    'evi': ['enhanced vegetation index'],
    'lai': ['leaf area index'],
    'gpp': ['gross primary production', 'gross primary productivity'],
    'npp': ['net primary production', 'net primary productivity'],
    'et': ['evapotranspiration'],
    'pet': ['potential evapotranspiration'],

    # Climate indices / large-scale modes
    'nao': ['north atlantic oscillation'],
    'oni': ['oceanic nino index', 'el nino southern oscillation', 'enso'],
    'pdo': ['pacific decadal oscillation'],
    'amo': ['atlantic multidecadal oscillation'],
    'ao': ['arctic oscillation'],
    'sam': ['southern annular mode'],
    'iod': ['indian ocean dipole'],
}


# ── Display-only formatting (never call on values used for matching/lookups) ──

# A few _VARIABLE_ALIASES keys are plain English words, not real acronyms --
# they exist purely so _strip_height_suffix()'d columns (e.g. "wind_10m" ->
# "wind") or short ambiguous codes still hit the alias dictionary. They must
# NOT be force-uppercased for display, or a legitimate node like "Wind
# Velocity" would render as "WIND Velocity".
_ACRONYM_DISPLAY_EXCLUDE = {'wind', 'sm', 'et', 'sp', 'ps', 'tp'}

_KNOWN_ACRONYMS = (frozenset(_VARIABLE_ALIASES.keys()) - _ACRONYM_DISPLAY_EXCLUDE) | {
    'enso', 'ipcc', 'noaa', 'nasa', 'pm2.5', 'pm25',
}


def prettify_node_name(name: str) -> str:
    """
    Title-cases a canonical (lowercase, space-separated) KG/CG node name for
    display — 'sea ice extent' -> 'Sea Ice Extent'. Known acronyms (from
    _VARIABLE_ALIASES' keys, e.g. 'co2', 'sst') and any word containing a
    digit are fully upper-cased instead of title-cased, since naive
    .capitalize() would otherwise produce 'Co2'/'Sst'. Pure display
    transform — never touches the underlying canonical string, so it's only
    safe to call at render time (widget labels, tags, graph node labels),
    never on a value used as a dict key, session-state value, or lookup.
    """
    if not name:
        return name
    words = []
    for w in name.split(' '):
        if w.lower() in _KNOWN_ACRONYMS or any(ch.isdigit() for ch in w):
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return ' '.join(words)


def prettify_column_name(col: str) -> str:
    """
    Display-only formatting for a raw dataset column header, e.g.
    'sea_ice_extent' -> 'Sea Ice Extent', 'T2M' stays 'T2M'. Splits on
    underscores/hyphens (common in CSV headers, unlike KG node names which
    are already space-separated) in addition to the acronym/digit rule
    prettify_node_name uses; a column already fully upper-case (e.g. 'SIC')
    is assumed to be an intentional code and left untouched. Never touches
    the actual column name used to index the DataFrame — call only at
    render time (e.g. via st.selectbox's format_func).
    """
    if not col or col == "(none)":
        return col
    words = []
    for w in re.split(r'[_\-\s]+', col.strip()):
        if not w:
            continue
        if w.isupper() or w.lower() in _KNOWN_ACRONYMS or any(ch.isdigit() for ch in w):
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return ' '.join(words)


_ALIAS_MIN_SCORE = 80  # aliases are curated equivalences — only trust a strong match

# partial_ratio scores by best-matching substring, so a short generic word
# (e.g. "ice") scores ~100 against any column that merely contains it (e.g.
# "sea_ice_extent"), even though "ice" alone is a far broader/vaguer concept
# than that specific column — this is the direct cause of two different KG
# nodes ("Ice" and "Sea Ice Extent") both being suggested for the same
# column. Only trust partial_ratio when the shorter string is at least this
# fraction of the longer one's length; token_sort_ratio (not susceptible to
# this failure mode the same way) is always considered regardless.
_MIN_PARTIAL_LEN_RATIO = 0.5

# Reanalysis/observational datasets commonly tag a sensor's measurement
# height onto the column name (e.g. "wind_10m", ERA5/NASA POWER style).
# Stripping it recovers a root ("wind") that can still hit the alias
# dictionary or fuzzy-match a height-agnostic phrase like "wind velocity".
# Requires an underscore before the digits, so glued-together whole-word
# abbreviations with no separator (e.g. "t2m", "ws2m", "rh2m") are left
# untouched and keep matching via their own exact alias-dictionary entries.
_HEIGHT_SUFFIX_RE = re.compile(r'_\d+(?:\.\d+)?(?:m|cm|km|mm)$', re.IGNORECASE)


def _strip_height_suffix(col_clean: str) -> str:
    """Removes a trailing sensor-height tag (e.g. '_10m' in 'wind_10m') so the
    remaining root ('wind') can still hit the alias dictionary or fuzzy-match
    a height-agnostic phrase — see _HEIGHT_SUFFIX_RE above for what does and
    doesn't qualify as a height suffix."""
    return _HEIGHT_SUFFIX_RE.sub('', col_clean)


def _match_score(a: str, b: str) -> int:
    """Best of two fuzzy metrics — token_sort_ratio handles reordered/full-phrase
    matches, partial_ratio handles one string being a substring/fragment of the
    other (common with abbreviation expansions). partial_ratio's contribution
    is only trusted when the shorter input is a large-enough fraction of the
    longer one's length (_MIN_PARTIAL_LEN_RATIO) — see its docstring for why."""
    token_score = fuzz.token_sort_ratio(a, b)
    shorter_len, longer_len = sorted((len(a), len(b)))
    if longer_len and shorter_len / longer_len >= _MIN_PARTIAL_LEN_RATIO:
        return max(token_score, fuzz.partial_ratio(a, b))
    return token_score


def suggest_variable_mapping(kg_nodes: list, dataset_columns: list, min_score: int = 55) -> dict:
    """
    Best-guess match between canonical KG/CG node names (e.g. 'sea ice extent')
    and real dataset column names (e.g. 'SIC'). Returns {kg_node: best_column_or_None}.

    Checks the curated abbreviation dictionary in both directions (column-as-
    abbreviation and node-as-abbreviation, plus a height-suffix-stripped form
    of the column, e.g. "wind_10m" -> "wind") before falling back to plain
    fuzzy matching, so known acronym/full-name pairs match even when the raw
    strings don't look alike. Alias matches require a high score
    (_ALIAS_MIN_SCORE) to count, since partial_ratio against a short generic
    alias word (e.g. "temperature") can otherwise clear the regular min_score
    by coincidence against an unrelated phrase.

    Global assignment, not independent per-node best-match: every (node,
    column) score is computed first, then assigned strongest-match-first, so
    a column already claimed by a higher-scoring node can't also be handed to
    a weaker one (previously, a spuriously high-scoring short node like "ice"
    could "tie" or beat the real "sea ice extent" node for the same column —
    both would then independently pick it, producing a duplicate mapping).
    The weaker competitor is left unmapped (None) rather than duplicated.

    This is a SUGGESTION only — the frontend should let the user confirm/
    override every mapping before anything is run against it.
    """
    scores = {}  # (node, col) -> score
    for node in kg_nodes:
        node_clean = node.lower().strip()
        node_aliases = _VARIABLE_ALIASES.get(node_clean, [])

        for col in dataset_columns:
            col_clean = col.lower().strip()
            col_stripped = _strip_height_suffix(col_clean)

            score = _match_score(node_clean, col_clean)
            if col_stripped != col_clean:
                score = max(score, _match_score(node_clean, col_stripped))

            col_aliases = list(_VARIABLE_ALIASES.get(col_clean, []))
            if col_stripped != col_clean:
                col_aliases += _VARIABLE_ALIASES.get(col_stripped, [])

            for alias in col_aliases:
                alias_score = _match_score(node_clean, alias)
                if alias_score >= _ALIAS_MIN_SCORE:
                    score = max(score, alias_score)
            for alias in node_aliases:
                alias_score = _match_score(alias, col_clean)
                if alias_score >= _ALIAS_MIN_SCORE:
                    score = max(score, alias_score)

            scores[(node, col)] = score

    # Greedy global assignment: strongest match wins first. Sorted
    # descending, so once a score drops below min_score every remaining
    # candidate does too — safe to stop there.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    mapping = {node: None for node in kg_nodes}
    used_columns = set()
    for (node, col), score in ranked:
        if score < min_score:
            break
        if mapping[node] is not None or col in used_columns:
            continue
        mapping[node] = col
        used_columns.add(col)

    return mapping


# ── Statistical causal discovery ────────────────────────────────────────

def run_pc(df: pd.DataFrame, columns: list, alpha: float = 0.05) -> list:
    """
    Constraint-based PC algorithm (Spirtes & Glymour, 1991; causal-learn's
    implementation). Treats rows as i.i.d. samples — no time/lag awareness.
    Many true causal edges come back undirected (PC can only
    orient edges that form a v-structure/collider) — those are intentionally
    excluded here since we only want directed claims to compare against the LLM CG.

    Returns list of {cause, effect, label, confidence} with cause/effect as
    DATASET COLUMN names.
    """
    from causallearn.search.ConstraintBased.PC import pc as causallearn_pc

    data = df[columns].dropna().to_numpy()
    if data.shape[0] < 20:
        raise ValueError(f"Not enough complete rows ({data.shape[0]}) to run PC reliably (need >= 20).")

    print(f"  [PC] {data.shape[0]} rows x {len(columns)} variables, alpha={alpha}")
    cg = causallearn_pc(data, alpha=alpha, node_names=columns, show_progress=False)
    graph = cg.G.graph  # graph[i, j] = endpoint mark at node i on the i-j edge (-1=TAIL, 1=ARROW, 0=no edge)

    edges = []
    undirected_count = 0
    n = len(columns)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if graph[i, j] == -1 and graph[j, i] == 1:  # tail at i, arrow at j => i -> j
                edges.append({
                    'cause': columns[i],
                    'effect': columns[j],
                    'label': 'PC_CAUSES',
                    'confidence': 1.0,
                })
            elif graph[i, j] == -1 and graph[j, i] == -1 and i < j:  # undirected, count once per pair
                undirected_count += 1

    print(f"  [PC] {len(edges)} directed edge(s) kept"
          + (f", {undirected_count} undirected pair(s) found but not orientable — dropped" if undirected_count else ""))
    return edges


def run_fci(df: pd.DataFrame, columns: list, alpha: float = 0.05) -> list:
    """
    FCI (Fast Causal Inference; Spirtes, Meek & Richardson, 1999;
    causal-learn's implementation). Like PC, treats rows as i.i.d.
    samples — no time/lag awareness. Unlike PC, FCI additionally tolerates
    unmeasured confounders: PC/CPDAG has no way to represent "these two share
    a hidden common cause", while FCI's PAG output does, via a bidirected
    (<->) mark.

    Only fully-oriented tail->arrow edges are kept as directed cause/effect
    claims here — same policy as run_pc (undirected/circle marks are dropped
    since we only want directed claims to compare against the LLM CG).
    Bidirected (<->) edges are reported to the terminal but also dropped:
    they explicitly assert "no direct causal claim can be made here", so
    keeping them as a fabricated direction would misrepresent what FCI
    actually found.

    Returns list of {cause, effect, label, confidence} with cause/effect as
    DATASET COLUMN names.
    """
    from causallearn.search.ConstraintBased.FCI import fci as causallearn_fci
    from causallearn.graph.Endpoint import Endpoint

    data = df[columns].dropna().to_numpy()
    if data.shape[0] < 20:
        raise ValueError(f"Not enough complete rows ({data.shape[0]}) to run FCI reliably (need >= 20).")

    print(f"  [FCI] {data.shape[0]} rows x {len(columns)} variables, alpha={alpha}")
    _, fci_edges = causallearn_fci(
        data,
        independence_test_method="fisherz",
        alpha=alpha,
        show_progress=False,
        node_names=columns
    )

    edges = []
    bidirected_count = 0
    circle_count = 0
    for edge in fci_edges:
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()
        ep1 = edge.get_endpoint1()
        ep2 = edge.get_endpoint2()

        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            edges.append({'cause': n1, 'effect': n2, 'label': 'FCI_CAUSES', 'confidence': 1.0})
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            edges.append({'cause': n2, 'effect': n1, 'label': 'FCI_CAUSES', 'confidence': 1.0})
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.ARROW:
            bidirected_count += 1
        else:
            circle_count += 1

    print(f"  [FCI] {len(edges)} directed edge(s) kept"
          + (f", {bidirected_count} bidirected pair(s) (possible latent confounder) — dropped" if bidirected_count else "")
          + (f", {circle_count} undirected/ambiguous pair(s) — dropped" if circle_count else ""))
    return edges


def run_cdnod(df: pd.DataFrame, columns: list, alpha: float = 0.05) -> list:
    """
    CD-NOD (Constraint-based causal Discovery from Nonstationary/
    heterogeneous Data; Huang et al., 2020; causal-learn's implementation)
    — constraint-based discovery like PC, but augments the dataset with a
    context index (c_indx) representing possible heterogeneity/
    nonstationarity across rows, and can additionally flag which
    variables' causal mechanisms appear to depend on that index. Treats
    rows as i.i.d. samples otherwise — no time/lag awareness beyond the
    context index itself.

    c_indx is row order (np.arange) rather than a real timestamp, matching
    the convention PCMCI/TCDF already use elsewhere in this module when no
    timestamp column is threaded through separately. This is a proxy for
    "position in the sequence," not an actual measured time value; the
    directed edges CD-NOD returns don't depend on which proxy is used for
    c_indx, only the nonstationarity diagnostic below would (a real
    timestamp could reveal nonstationarity that pure row order misses if
    rows aren't evenly spaced in time).

    Only fully-oriented tail->arrow edges are kept as directed cause/effect
    claims — same "drop what isn't orientable" policy as run_pc. Variables
    found to be adjacent to the context index itself are not turned into
    edges (the index isn't a real variable) but are logged to the terminal
    as CD-NOD's distinguishing diagnostic over plain PC.

    Returns list of {cause, effect, label, confidence} with cause/effect as
    DATASET COLUMN names.
    """
    from causallearn.search.ConstraintBased.CDNOD import cdnod as causallearn_cdnod

    data = df[columns].dropna().to_numpy()
    if data.shape[0] < 20:
        raise ValueError(f"Not enough complete rows ({data.shape[0]}) to run CD-NOD reliably (need >= 20).")

    c_indx = np.arange(data.shape[0]).reshape(-1, 1).astype(float)

    print(f"  [CD-NOD] {data.shape[0]} rows x {len(columns)} variables, alpha={alpha}")
    cg = causallearn_cdnod(data, c_indx, alpha=alpha, indep_test="fisherz", show_progress=False)
    graph = cg.G.graph  # (n+1)x(n+1): same endpoint-mark convention as run_pc, last row/col is c_indx

    n = len(columns)
    edges = []
    undirected_count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if graph[i, j] == -1 and graph[j, i] == 1:  # tail at i, arrow at j => i -> j
                edges.append({
                    'cause': columns[i],
                    'effect': columns[j],
                    'label': 'CDNOD_CAUSES',
                    'confidence': 1.0,
                })
            elif graph[i, j] == -1 and graph[j, i] == -1 and i < j:  # undirected, count once per pair
                undirected_count += 1

    changing_vars = [columns[i] for i in range(n) if graph[i, n] != 0 or graph[n, i] != 0]

    print(f"  [CD-NOD] {len(edges)} directed edge(s) kept"
          + (f", {undirected_count} undirected pair(s) found but not orientable — dropped" if undirected_count else ""))
    print(f"  [CD-NOD] variable(s) flagged as related to the context index (possible nonstationarity): "
          f"{changing_vars if changing_vars else 'none'}")
    return edges


def run_lingam(df: pd.DataFrame, columns: list, min_coefficient: float = 0.1) -> list:
    """
    DirectLiNGAM (Linear Non-Gaussian Acyclic Model; Shimizu et al., 2006;
    the `lingam` package's implementation) — full DAG orientation from
    i.i.d. data, complementary to PC: PC only orients v-structure/collider edges
    and often returns nothing for simple chains (e.g. X->Y->Z); LiNGAM uses the
    assumption that noise is non-Gaussian to fully orient every edge instead.
    No time/lag awareness (like PC) — treats rows as i.i.d. samples.

    LiNGAM assigns a coefficient to nearly every variable pair in its fitted
    causal order rather than doing sparse structure learning, so keeping every
    nonzero coefficient over-connects badly: on a benchmark run against the
    Sachs protein-signaling dataset (a standard causal-discovery benchmark
    with 11 variables and 18 known true edges), a near-zero threshold of
    1e-3 kept 36 edges — twice the true count. min_coefficient=0.1 was
    tuned against that same benchmark (20 edges vs. 18 true) as a much
    closer default.

    Returns list of {cause, effect, label, confidence} with cause/effect as
    DATASET COLUMN names.
    """
    import lingam
    from sklearn.preprocessing import StandardScaler

    data = df[columns].dropna().to_numpy()
    if data.shape[0] < 20:
        raise ValueError(f"Not enough complete rows ({data.shape[0]}) to run LiNGAM reliably (need >= 20).")

    print(f"  [LiNGAM] {data.shape[0]} rows x {len(columns)} variables, min_coefficient={min_coefficient}")

    data_scaled = StandardScaler().fit_transform(data)
    model = lingam.DirectLiNGAM()
    model.fit(data_scaled)

    adj = model.adjacency_matrix_  # adj[i, j] = coefficient of column j (cause) on row i (effect)

    edges = []
    n = len(columns)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            coef = adj[i, j]
            if abs(coef) >= min_coefficient:
                edges.append({
                    'cause': columns[j],
                    'effect': columns[i],
                    'label': 'LINGAM_CAUSES',
                    'confidence': round(min(1.0, abs(coef)), 3),
                })

    print(f"  [LiNGAM] {len(edges)} directed edge(s) kept (min_coefficient={min_coefficient})")
    return edges


def _run_pcmci_family(
    df: pd.DataFrame,
    columns: list,
    tau_min: int,
    tau_max: int,
    alpha: float,
    plus: bool,
) -> list:
    """
    Shared implementation behind run_pcmci()/run_pcmci_plus() (Runge et
    al., Science Advances 2019 / Runge, UAI 2020; Tigramite's
    implementation) — time-lagged causal discovery: for each ordered pair
    of variables (source, target) and each lag from 1 (or 0 for PCMCI+) up
    to tau_max, tests whether "source at time t-lag" is conditionally
    independent of "target at time t" given the rest of the system. A
    dependency that survives this conditioning is kept as a directed
    edge, labeled with the lag it was found at. PCMCI+ (plus=True)
    additionally tests lag-0 (contemporaneous/same-timestep) links, which
    plain PCMCI (plus=False, tau_min forced to >=1) never considers —
    that is the entire difference between the two.

    Two independence-test choices, both standard Tigramite defaults kept
    as-is here rather than tuned: ParCorr (partial correlation) as the
    conditional-independence test, appropriate for linear-Gaussian
    relationships (Tigramite also offers nonlinear tests like GPDC/CMI,
    not used here); and fdr_method="fdr_bh" (Benjamini-Hochberg
    false-discovery-rate correction) applied across all the simultaneous
    per-pair-per-lag tests, controlling the *expected proportion* of
    false-positive edges among all edges kept — a more practical choice
    than Bonferroni correction (which controls the probability of *any*
    false positive but becomes very conservative as the number of
    lag/pair combinations grows with tau_max and variable count).

    tau_max is capped at min(requested, num_rows // 10) — Tigramite's own
    conditional-independence tests need enough rows per tested lag to have
    statistical power; requesting a lag depth close to the dataset's own
    length leaves too few effective samples per test and produces
    unreliable results, so this caps it proportionally rather than
    trusting an arbitrarily large user-requested value. The 30-row minimum
    below is the same kind of guard, just an absolute floor rather than a
    proportional one.

    Output:
        {
            "cause": source column,
            "effect": target column,
            "label": "PCMCI_LAGk" or "PCMCI_PLUS_LAGk",
            "confidence": absolute partial-correlation strength
        }
    """
    from sklearn.preprocessing import StandardScaler

    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    method_name = "PCMCI+" if plus else "PCMCI"

    # ── Input validation ──────────────────────────────────────────────────

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    columns = list(columns)

    if len(columns) < 2:
        raise ValueError(
            f"Need at least 2 columns to run {method_name}."
        )

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing_columns}"
        )

    raw_frame = df.loc[:, columns].copy()

    nonnumeric_columns = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(raw_frame[column])
    ]

    if nonnumeric_columns:
        raise TypeError(
            f"{method_name} requires numeric columns: "
            f"{nonnumeric_columns}"
        )

    # Interpolate rather than drop incomplete rows: dropna() silently splices
    # non-adjacent timepoints together across any gap, which breaks the
    # fixed time-lag assumption PCMCI-family methods rely on for every row.
    data_frame = raw_frame.interpolate(
        method="linear",
        limit_direction="both",
    )

    all_nan_columns = [
        column
        for column in columns
        if data_frame[column].isna().all()
    ]

    if all_nan_columns:
        raise ValueError(
            f"Column(s) with no valid values to interpolate from: "
            f"{all_nan_columns}"
        )

    data = data_frame.to_numpy(dtype=float)

    # 30 rows is an absolute floor below which the conditional-independence
    # tests below have too little statistical power to be meaningful,
    # independent of how large tau_max ends up being -- see this
    # function's docstring for how the tau_max cap right below handles
    # the proportional version of the same concern.
    if data.shape[0] < 30:
        raise ValueError(
            f"Not enough complete rows ({data.shape[0]}) "
            f"to run {method_name} reliably (need >= 30)."
        )

    standard_deviations = data.std(axis=0, ddof=0)

    constant_columns = [
        columns[index]
        for index, value in enumerate(standard_deviations)
        if value <= 1e-12
    ]

    if constant_columns:
        raise ValueError(
            f"Constant or nearly constant column(s): {constant_columns}"
        )

    # Capped proportionally to dataset length (rows // 10) -- see this
    # function's docstring for why an arbitrarily large user-requested
    # tau_max isn't trusted directly.
    tau_max = max(
        1,
        min(
            int(tau_max),
            data.shape[0] // 10,
        ),
    )

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    # ── Standardization ────────────────────────────────────────────────────

    data_scaled = (
        StandardScaler()
        .fit_transform(data)
    )

    if not np.isfinite(data_scaled).all():
        raise ValueError(
            f"{method_name} input contains NaN or infinite values "
            "after preprocessing."
        )

    print(
        f"  [{method_name}] "
        f"{data_scaled.shape[0]} rows x "
        f"{len(columns)} variables, "
        f"tau_max={tau_max}, alpha={alpha}, "
        "fdr_method=fdr_bh"
    )

    tigramite_df = pp.DataFrame(
        data_scaled,
        var_names=columns,
    )

    pcmci = PCMCI(
        dataframe=tigramite_df,
        cond_ind_test=ParCorr(
            significance="analytic"
        ),
        verbosity=0,
    )

    # ── Original Tigramite PCMCI/PCMCI+ calls ─────────────────────────────

    if plus:
        results = pcmci.run_pcmciplus(
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=alpha,
            fdr_method="fdr_bh",
        )
    else:
        results = pcmci.run_pcmci(
            tau_min=max(tau_min, 1),
            tau_max=tau_max,
            pc_alpha=alpha,
            alpha_level=alpha,
            fdr_method="fdr_bh",
        )

    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    # ── Preserve the app's one-edge-per-ordered-pair convention ───────────

    best = {}
    n_variables = len(columns)

    first_lag = max(
        tau_min,
        0 if plus else 1,
    )

    for source_index in range(n_variables):
        for target_index in range(n_variables):
            if source_index == target_index:
                continue

            for lag in range(
                first_lag,
                tau_max + 1,
            ):
                if (
                    graph[
                        source_index,
                        target_index,
                        lag,
                    ]
                    != "-->"
                ):
                    continue

                effect_strength = float(
                    val_matrix[
                        source_index,
                        target_index,
                        lag,
                    ]
                )

                key = (
                    columns[source_index],
                    columns[target_index],
                )

                if (
                    key not in best
                    or abs(effect_strength)
                    > abs(best[key][1])
                ):
                    best[key] = (
                        round(
                            float(
                                p_matrix[
                                    source_index,
                                    target_index,
                                    lag,
                                ]
                            ),
                            4,
                        ),
                        effect_strength,
                        lag,
                    )

    edges = []

    for (
        source,
        target,
    ), (
        p_value,
        effect_strength,
        lag,
    ) in best.items():
        edges.append(
            {
                "cause": source,
                "effect": target,
                "label": (
                    f"PCMCI"
                    f"{'_PLUS' if plus else ''}"
                    f"_LAG{lag}"
                ),
                "confidence": round(
                    min(
                        1.0,
                        abs(effect_strength),
                    ),
                    3,
                ),
            }
        )

    print(
        f"  [{method_name}] "
        f"{len(edges)} directed edge(s) kept "
        "after FDR correction"
    )

    return edges, graph, val_matrix, p_matrix, columns


def run_pcmci(
    df: pd.DataFrame,
    columns: list,
    tau_max: int = 5,
    alpha: float = 0.05,
) -> list:
    """
    PCMCI (Runge et al., Science Advances 2019) — lagged-only causal
    discovery: tests whether each variable's past values (up to tau_max
    rows back) causally affect each other variable's current value, using
    partial correlation with FDR correction. See _run_pcmci_family()'s
    docstring above for the full algorithm/parameter explanation this and
    run_pcmci_plus() share. tau_min is forced to >=1 here (no lag-0
    testing) -- that's the one difference from run_pcmci_plus() below.

    The dataset rows must be in chronological order.
    """
    edges, _graph, _val_matrix, _p_matrix, _columns = _run_pcmci_family(
        df,
        columns,
        tau_min=1,
        tau_max=tau_max,
        alpha=alpha,
        plus=False,
    )
    return edges


def run_pcmci_plus(
    df: pd.DataFrame,
    columns: list,
    tau_max: int = 5,
    alpha: float = 0.05,
) -> list:
    """
    PCMCI+ (Runge, UAI 2020) — same lagged causal discovery as PCMCI
    above, plus lag-0 (contemporaneous/same-timestep) links, which plain
    PCMCI never tests (tau_min=0 here vs. run_pcmci()'s forced tau_min=1).
    Use this over plain PCMCI whenever two variables might influence each
    other within the same observation interval, not just across
    intervals. See _run_pcmci_family()'s docstring above for the full
    algorithm/parameter explanation.

    The dataset rows must be in chronological order.
    """
    edges, _graph, _val_matrix, _p_matrix, _columns = _run_pcmci_family(
        df,
        columns,
        tau_min=0,
        tau_max=tau_max,
        alpha=alpha,
        plus=True,
    )
    return edges


def run_pcmci_plus_full(
    df: pd.DataFrame,
    columns: list,
    tau_max: int = 5,
    alpha: float = 0.05,
) -> dict:
    """
    Same computation as run_pcmci_plus(), run exactly once, but also returns
    the raw Tigramite artifacts needed to draw the full multivariate causal
    graph (every link type Tigramite found, not just the resolved KG edges).

    Returns:
        {
            "edges": [...],       # identical to run_pcmci_plus()'s return value
            "graph": np.ndarray,      # shape [N, N, tau_max+1], all link marks
            "val_matrix": np.ndarray, # cross-MCI / auto-MCI strengths
            "p_matrix": np.ndarray,
            "var_names": [...],       # == columns, axis order for the arrays above
        }
    """
    edges, graph, val_matrix, p_matrix, var_names = _run_pcmci_family(
        df,
        columns,
        tau_min=0,
        tau_max=tau_max,
        alpha=alpha,
        plus=True,
    )
    return {
        "edges": edges,
        "graph": graph,
        "val_matrix": val_matrix,
        "p_matrix": p_matrix,
        "var_names": var_names,
    }


def run_pcmci_full(
    df: pd.DataFrame,
    columns: list,
    tau_max: int = 5,
    alpha: float = 0.05,
) -> dict:
    """
    Same computation as run_pcmci(), run exactly once, but also returns the
    raw Tigramite artifacts needed to draw the full lagged causal graph.
    See run_pcmci_plus_full() for the returned dict shape.
    """
    edges, graph, val_matrix, p_matrix, var_names = _run_pcmci_family(
        df,
        columns,
        tau_min=1,
        tau_max=tau_max,
        alpha=alpha,
        plus=False,
    )
    return {
        "edges": edges,
        "graph": graph,
        "val_matrix": val_matrix,
        "p_matrix": p_matrix,
        "var_names": var_names,
    }


def plot_full_causal_graph(
    graph,
    val_matrix,
    var_names: list,
    title: str = None,
):
    """
    Render the full Tigramite multivariate causal graph (every link type:
    '-->', 'x-x', 'o-o', etc. — not just the resolved edges the app's
    edge-dict output keeps) for display, e.g. via Streamlit's st.pyplot().

    Returns the (fig, ax) tuple tigramite.plotting.plot_graph() already
    produces, so the caller gets a ready-to-render matplotlib Figure.
    """
    from tigramite import plotting as tp

    fig, ax = tp.plot_graph(
        val_matrix=val_matrix,
        graph=graph,
        var_names=var_names,
        link_colorbar_label="cross-MCI",
        node_colorbar_label="auto-MCI",
        figsize=(10, 6),
    )
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig, ax


def run_tcdf(
    df: pd.DataFrame,
    columns: list,
    tau_max: int = 7,
    epochs: int = 500,
    significance: float = 0.8,
    kernel_size: int = 4,
    num_levels: int = 1,
    dilation_c: int = 2,
    lr: float = 0.01,
    seed: int = 42,
) -> list:
    """
    TCDF (Temporal Causal Discovery Framework; Nauta, Bucur & Seifert,
    2019) — time-series causal discovery via a small neural network per
    target variable, rather than a statistical conditional-independence
    test like the PCMCI family above. For each target column, trains an
    attention-augmented depthwise-causal CNN (the nested _ADDSTCN class
    below) to predict that column from every column's own recent history
    (up to tau_max lags back); the network's learned per-input attention
    weights (fs_attention) indicate which source columns it actually
    relied on. Candidate causes are then validated via PIVM (Permutation
    Importance Validation Method): each candidate source's values are
    shuffled and the trained network is re-scored — a genuine cause's
    removal should measurably hurt prediction loss, while a spurious one
    shouldn't (see _run_single_replica's PIVM step below). The specific
    time lag for each validated cause is then read off via kernel-based
    delay extraction (which input timestep the network's convolution
    kernel weighted most heavily).

    Known limitation, not fixed in this port: PIVM runs one permutation
    test per (candidate source, target) pair with no multiple-testing
    correction across those tests — unlike the PCMCI family above, which
    applies Benjamini-Hochberg FDR correction across all its simultaneous
    tests. This means TCDF is more prone to false-positive edges as the
    number of variables grows, a property of the original TCDF method
    itself, not specific to this implementation.

    Stability improvements over a single vanilla TCDF run, on top of the
    original method:
      1. Detects strong non-stationarity and uses first-difference
         standardization when necessary (a trend can otherwise dominate
         the network's loss and drown out genuine short-lag causal
         signal).
      2. Trains two independent replicas (different random seeds) and
         keeps only causal pairs found by BOTH, reducing seed-specific
         false positives from the neural network's stochastic training.
      3. Limits CPU thread overhead for faster small-model execution.

    Default hyperparameters (matching the original TCDF paper's own
    defaults, not independently tuned against a benchmark the way e.g.
    LiNGAM's min_coefficient was): tau_max=7 (lags considered),
    epochs=500 (training length per replica), significance=0.8 (the PIVM
    permutation-importance threshold a candidate source must clear to be
    kept), kernel_size=4 and dilation_c=2 (control the CNN's effective
    receptive field, i.e. how far back in time one convolution layer can
    see), num_levels=1 (number of stacked causal-CNN layers), lr=0.01
    (training learning rate).

    Application interface is unchanged from the original TCDF port: same
    function name/parameters/defaults/return type; same four output
    fields (cause, effect, label, confidence); self-loops excluded to
    match the rest of this module's convention.

    Rows of df must be ordered chronologically.
    """
    import heapq
    import random
    from collections import Counter, defaultdict

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from sklearn.preprocessing import StandardScaler

    # ── Input checks ──────────────────────────────────────────────────────

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    columns = list(columns)

    if len(columns) < 2:
        raise ValueError("Need at least 2 columns to run TCDF.")

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing_columns}"
        )

    data_frame = df.loc[:, columns].dropna().copy()

    nonnumeric_columns = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(data_frame[column])
    ]

    if nonnumeric_columns:
        raise TypeError(
            f"TCDF requires numeric columns: {nonnumeric_columns}"
        )

    raw_data = data_frame.to_numpy(dtype=np.float32)
    original_n_rows, n_vars = raw_data.shape

    if original_n_rows < 30:
        raise ValueError(
            f"Not enough complete rows ({original_n_rows}) "
            "to run TCDF reliably (need >= 30)."
        )

    column_stds = raw_data.std(axis=0, ddof=0)

    constant_columns = [
        columns[index]
        for index, value in enumerate(column_stds)
        if value <= 1e-12
    ]

    if constant_columns:
        raise ValueError(
            f"Constant or nearly constant column(s): {constant_columns}"
        )

    # ── Stationarity-aware preprocessing ──────────────────────────────────

    time_index = np.arange(
        original_n_rows,
        dtype=np.float64,
    )

    trend_correlations = []

    for variable_index in range(n_vars):
        series = raw_data[:, variable_index].astype(np.float64)

        correlation = np.corrcoef(
            time_index,
            series,
        )[0, 1]

        if np.isfinite(correlation):
            trend_correlations.append(
                abs(float(correlation))
            )
        else:
            trend_correlations.append(0.0)

    median_trend_strength = float(
        np.median(trend_correlations)
    )

    # Strong common trend can dominate TCDF attention. First differencing
    # removes that trend while preserving short-term temporal dependence.
    use_first_difference = (
        median_trend_strength >= 0.30
        and original_n_rows >= 31
    )

    if use_first_difference:
        model_data = np.diff(
            raw_data,
            axis=0,
        )

        preprocessing_name = (
            "first-difference + standardization"
        )
    else:
        model_data = raw_data.copy()
        preprocessing_name = "standardization"

    data_scaled = (
        StandardScaler()
        .fit_transform(model_data)
        .astype(np.float32)
    )

    if not np.isfinite(data_scaled).all():
        raise ValueError(
            "TCDF input contains NaN or infinite values "
            "after preprocessing."
        )

    n_rows = data_scaled.shape[0]

    tau_max = max(
        1,
        min(
            int(tau_max),
            n_rows // 10,
        ),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    cuda = device.type == "cuda"

    old_thread_count = torch.get_num_threads()

    if not cuda:
        torch.set_num_threads(1)

    # ── TCDF network ──────────────────────────────────────────────────────

    class _Chomp1d(nn.Module):
        """
        Causal padding trim. A dilated 1D convolution needs symmetric
        padding to keep its output the same length as its input, but
        symmetric padding lets each output timestep see FUTURE input
        timesteps too -- not allowed for causal (time-respecting)
        prediction. This layer chops off the extra `chomp_size` timesteps
        from the END of the convolution's output, leaving only the
        causal (past-only) portion. Used inside every block below.
        """

        def __init__(self, chomp_size):
            super().__init__()
            self.chomp_size = int(chomp_size)

        def forward(self, x):
            if self.chomp_size == 0:
                return x

            return x[
                :,
                :,
                :-self.chomp_size,
            ].contiguous()

    class _FirstBlock(nn.Module):
        """
        First layer of the causal-CNN stack (_DepthwiseCausalNet below).
        A depthwise causal convolution (groups=n_inputs makes each input
        channel/column convolved independently of the others -- no
        cross-channel mixing happens in this layer) followed by the
        _Chomp1d causal trim above and a PReLU activation. No residual
        connection, unlike _TemporalBlock/_LastBlock below, since this is
        the network's first transformation of the raw input.
        """

        def __init__(
            self,
            n_inputs,
            current_kernel_size,
            dilation,
            padding,
        ):
            super().__init__()

            self.conv1 = nn.Conv1d(
                n_inputs,
                n_inputs,
                current_kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=n_inputs,
            )

            self.net = nn.Sequential(
                self.conv1,
                _Chomp1d(padding),
            )

            self.relu = nn.PReLU(n_inputs)

            self.conv1.weight.data.normal_(
                0.0,
                0.1,
            )

        def forward(self, x):
            return self.relu(
                self.net(x)
            )

    class _TemporalBlock(nn.Module):
        """
        Middle layer(s) of the causal-CNN stack -- identical structure to
        _FirstBlock above (depthwise causal conv + chomp + PReLU), but
        with a residual connection (self.net(x) + x in forward()) so
        gradients can flow directly through the block during training,
        the standard trick that lets deeper temporal-CNN stacks train
        stably. Only used when num_levels > 2 (one _FirstBlock, some
        number of _TemporalBlocks, one _LastBlock).
        """

        def __init__(
            self,
            n_inputs,
            current_kernel_size,
            dilation,
            padding,
        ):
            super().__init__()

            self.conv1 = nn.Conv1d(
                n_inputs,
                n_inputs,
                current_kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=n_inputs,
            )

            self.net = nn.Sequential(
                self.conv1,
                _Chomp1d(padding),
            )

            self.relu = nn.PReLU(n_inputs)

            self.conv1.weight.data.normal_(
                0.0,
                0.1,
            )

        def forward(self, x):
            return self.relu(
                self.net(x) + x
            )

    class _LastBlock(nn.Module):
        """
        Final layer of the causal-CNN stack. Same depthwise causal conv +
        chomp + residual pattern as _TemporalBlock above, but the output
        is passed through an extra Linear layer instead of an activation
        -- this produces the network's final per-timestep prediction, so
        it needs a layer that can freely rescale/mix values rather than a
        bounded nonlinearity. forward() transposes to (batch, time,
        channels) before the Linear layer (which operates on the last
        dimension) and transposes back after, since Conv1d and Linear
        expect channels in different positions.
        """

        def __init__(
            self,
            n_inputs,
            current_kernel_size,
            dilation,
            padding,
        ):
            super().__init__()

            self.conv1 = nn.Conv1d(
                n_inputs,
                n_inputs,
                current_kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=n_inputs,
            )

            self.net = nn.Sequential(
                self.conv1,
                _Chomp1d(padding),
            )

            self.linear = nn.Linear(
                n_inputs,
                n_inputs,
            )

            self.linear.weight.data.normal_(
                0.0,
                0.01,
            )

        def forward(self, x):
            output = self.net(x)

            return self.linear(
                (output + x).transpose(1, 2)
            ).transpose(1, 2)

    class _DepthwiseCausalNet(nn.Module):
        """
        Stacks current_num_levels of the blocks above (_FirstBlock, then
        any _TemporalBlocks, then _LastBlock) with exponentially growing
        dilation (current_dilation_c ** level) at each level -- the
        standard dilated-causal-CNN trick for covering a long time window
        with few layers: each added level roughly multiplies how far back
        in time the network can "see" (its receptive field) by
        current_dilation_c, rather than only adding current_kernel_size-1
        more timesteps the way an undilated stack would.
        """

        def __init__(
            self,
            num_inputs,
            current_num_levels,
            current_kernel_size,
            current_dilation_c,
        ):
            super().__init__()

            layers = []

            for level in range(current_num_levels):
                dilation = (
                    current_dilation_c ** level
                )

                padding = (
                    current_kernel_size - 1
                ) * dilation

                if level == 0:
                    layers.append(
                        _FirstBlock(
                            num_inputs,
                            current_kernel_size,
                            dilation,
                            padding,
                        )
                    )

                elif level == current_num_levels - 1:
                    layers.append(
                        _LastBlock(
                            num_inputs,
                            current_kernel_size,
                            dilation,
                            padding,
                        )
                    )

                else:
                    layers.append(
                        _TemporalBlock(
                            num_inputs,
                            current_kernel_size,
                            dilation,
                            padding,
                        )
                    )

            self.network = nn.Sequential(
                *layers
            )

        def forward(self, x):
            return self.network(x)

    class _ADDSTCN(nn.Module):
        """
        Full per-target network: attention-gated depthwise causal CNN.
        One instance of this is trained per target column (see
        _run_single_replica below) to predict that column from every
        column's own recent history.

        fs_attention is a learnable weight per INPUT column (not a
        function of the data, unlike typical transformer-style attention
        -- it's a single global set of weights learned during training),
        passed through softmax so all weights sum to 1. The input is
        elementwise-scaled by these weights BEFORE entering the causal
        CNN (self.dwn, a _DepthwiseCausalNet) -- so an input column the
        network learns to weight near 0 is effectively suppressed from
        influencing the prediction at all. After the causal CNN, a
        pointwise (1x1) convolution collapses the per-column hidden
        channels down to the network's single scalar prediction per
        timestep. It is these learned fs_attention weights, once
        training converges, that identify which source columns the
        network actually relied on -- the basis for PIVM's permutation
        test in _run_single_replica below.
        """

        def __init__(
            self,
            input_size,
            current_num_levels,
            current_kernel_size,
            current_dilation_c,
        ):
            super().__init__()

            self.dwn = _DepthwiseCausalNet(
                input_size,
                current_num_levels,
                current_kernel_size,
                current_dilation_c,
            )

            self.pointwise = nn.Conv1d(
                input_size,
                1,
                kernel_size=1,
            )

            self.fs_attention = nn.Parameter(
                torch.ones(input_size, 1)
            )

        def forward(self, x):
            attention = F.softmax(
                self.fs_attention,
                dim=0,
            )

            hidden = self.dwn(
                x * attention
            )

            return self.pointwise(
                hidden
            ).transpose(1, 2)

    # ── TCDF helper functions ─────────────────────────────────────────────

    def _set_seed(current_seed):
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        random.seed(current_seed)

        if cuda:
            torch.cuda.manual_seed(current_seed)
            torch.cuda.manual_seed_all(
                current_seed
            )

    def _prepare_data(
        array,
        target_index,
    ):
        total_rows = array.shape[0]

        x_array = array.copy()
        target = array[
            :,
            target_index,
        ].copy()

        shifted_target = np.zeros(
            total_rows,
            dtype=np.float32,
        )

        shifted_target[1:] = array[
            :-1,
            target_index,
        ]

        x_array[
            :,
            target_index,
        ] = shifted_target

        x_tensor = torch.from_numpy(
            x_array.T[
                np.newaxis,
                :,
                :,
            ]
        )

        y_tensor = torch.from_numpy(
            target[
                np.newaxis,
                :,
                np.newaxis,
            ]
        )

        return x_tensor, y_tensor

    def _select_potentials_by_gap(
        scores,
    ):
        """
        Picks which input columns are plausible causal candidates for a
        target, from their learned fs_attention scores (see _ADDSTCN
        above), WITHOUT a fixed cutoff threshold. `scores[i] > 1.0` is
        the baseline check for "attended to more than an average/
        uninformative input would be" -- with 5 or fewer candidates that
        alone decides it.

        With more than 5 candidates, this instead looks for a natural
        "elbow": sort scores descending, compute the gap between each
        consecutive pair (only among scores still above the 1.0
        baseline), and take the LARGEST such gap that falls within the
        first half of the ranking (excluding the very first gap, which
        would trivially always be "largest" if the top score is an
        outlier) as the split point between "genuinely relevant" and
        "not relevant" candidates. This is a heuristic, not a
        statistically derived cutoff: real causal signals are assumed to
        separate cleanly from noise in the attention ranking, and this
        finds that separation adaptively per target rather than trusting
        one fixed threshold to work for every dataset. If no qualifying
        gap is found, falls back to keeping only the single
        highest-scoring candidate.
        """
        sorted_scores = sorted(
            scores.tolist(),
            reverse=True,
        )

        indices = np.argsort(
            -scores
        ).tolist()

        if len(sorted_scores) <= 5:
            return [
                index
                for index in indices
                if scores[index] > 1.0
            ]

        gaps = []

        for index in range(
            len(sorted_scores) - 1
        ):
            if sorted_scores[index] < 1.0:
                break

            gaps.append(
                sorted_scores[index]
                - sorted_scores[index + 1]
            )

        if not gaps:
            return []

        sorted_gaps = sorted(
            gaps,
            reverse=True,
        )

        split_index = -1

        for gap in sorted_gaps:
            gap_index = gaps.index(gap)

            if (
                0 < gap_index
                < (len(sorted_scores) - 1) / 2
            ):
                split_index = gap_index
                break

        if split_index < 0:
            split_index = 0

        return indices[
            : split_index + 1
        ]

    def _run_single_replica(
        replica_seed,
    ):
        _set_seed(replica_seed)

        replica_edges = []

        for target_index in range(n_vars):
            # Resetting for each target prevents target order from changing
            # model initialization and attention behavior.
            _set_seed(replica_seed)

            x_train, y_train = _prepare_data(
                data_scaled,
                target_index,
            )

            x_train = x_train.to(device)
            y_train = y_train.to(device)

            model = _ADDSTCN(
                n_vars,
                num_levels,
                kernel_size,
                dilation_c,
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
            )

            first_loss = None
            last_loss = None

            for epoch in range(
                1,
                epochs + 1,
            ):
                model.train()
                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction = model(
                    x_train
                )

                loss = F.mse_loss(
                    prediction,
                    y_train,
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "TCDF produced a non-finite "
                        f"loss for target "
                        f"'{columns[target_index]}'."
                    )

                loss.backward()
                optimizer.step()

                current_loss = float(
                    loss.detach()
                    .cpu()
                    .item()
                )

                if epoch == 1:
                    first_loss = current_loss

                last_loss = current_loss

            attention_scores = (
                model.fs_attention
                .detach()
                .view(-1)
                .cpu()
                .numpy()
            )

            potential_causes = (
                _select_potentials_by_gap(
                    attention_scores
                )
            )

            if not potential_causes:
                continue

            validated_causes = list(
                potential_causes
            )

            training_improvement = (
                first_loss - last_loss
            )

            # PIVM (Permutation Importance Validation Method): for each
            # attention-flagged candidate source, shuffle just that
            # source's values (destroying its real temporal signal while
            # leaving everything else intact) and re-score the ALREADY-
            # TRAINED model on this corrupted input. test_improvement
            # measures how much of the model's training-time gain
            # (training_improvement, first-epoch loss minus final loss)
            # still shows up even with that source scrambled. If the
            # model barely notices (test_improvement is still a large
            # fraction -- `significance` -- of training_improvement), the
            # source wasn't actually load-bearing for the prediction and
            # is discarded as a false positive from the attention step
            # above; only sources whose removal meaningfully hurts
            # performance survive as validated causes.
            #
            # No correction for multiple comparisons is applied here: one
            # independent shuffle test runs per (candidate source, target)
            # pair, unlike the PCMCI family above which applies
            # Benjamini-Hochberg FDR correction across all its
            # simultaneous tests. This is a property of the original TCDF
            # method (see run_tcdf's docstring), not something this port
            # adds or fixes.
            for source_index in potential_causes:
                random.seed(replica_seed)

                shuffled_array = (
                    x_train.detach()
                    .cpu()
                    .clone()
                    .numpy()
                )

                random.shuffle(
                    shuffled_array[
                        :,
                        source_index,
                        :,
                    ][0]
                )

                shuffled_input = (
                    torch.from_numpy(
                        shuffled_array
                    )
                    .to(device)
                )

                model.eval()

                with torch.no_grad():
                    shuffled_loss = float(
                        F.mse_loss(
                            model(shuffled_input),
                            y_train,
                        )
                        .detach()
                        .cpu()
                        .item()
                    )

                test_improvement = (
                    first_loss
                    - shuffled_loss
                )

                if (
                    test_improvement
                    > training_improvement
                    * significance
                ):
                    validated_causes.remove(
                        source_index
                    )

            if not validated_causes:
                continue

            convolution_weights = []

            for level in range(num_levels):
                level_weight = (
                    model.dwn
                    .network[level]
                    .net[0]
                    .weight
                    .detach()
                    .abs()
                )

                convolution_weights.append(
                    level_weight.view(
                        level_weight.size(0),
                        level_weight.size(2),
                    )
                )

            normalized_attention = (
                F.softmax(
                    torch.tensor(
                        attention_scores
                    ),
                    dim=0,
                )
                .numpy()
            )

            for source_index in validated_causes:
                # Preserve the existing app convention.
                if source_index == target_index:
                    continue

                total_delay = 0

                for level, weight in enumerate(
                    convolution_weights
                ):
                    row = weight[
                        source_index
                    ].tolist()

                    largest, second_largest = (
                        heapq.nlargest(
                            2,
                            row,
                        )
                    )

                    if largest > second_largest:
                        maximum_index = max(
                            range(len(row)),
                            key=row.__getitem__,
                        )

                        delay_index = (
                            len(row)
                            - 1
                            - maximum_index
                        )
                    else:
                        delay_index = 0

                    total_delay += (
                        delay_index
                        * (
                            dilation_c ** level
                        )
                    )

                # Preserve the current app's positive-lag label convention.
                lag = max(
                    1,
                    min(
                        int(total_delay),
                        tau_max,
                    ),
                )

                replica_edges.append(
                    {
                        "cause": columns[
                            source_index
                        ],
                        "effect": columns[
                            target_index
                        ],
                        "label": (
                            f"TCDF_LAG{lag}"
                        ),
                        "confidence": round(
                            min(
                                1.0,
                                float(
                                    normalized_attention[
                                        source_index
                                    ]
                                ),
                            ),
                            3,
                        ),
                    }
                )

        return replica_edges

    # ── Two-seed stability consensus ──────────────────────────────────────

    second_seed = (
        1111
        if seed != 1111
        else 42
    )

    replica_seeds = [
        int(seed),
        int(second_seed),
    ]

    print(
        f"  [TCDF] {n_rows} processed rows x "
        f"{n_vars} variables, "
        f"{epochs} epochs/target/replica, "
        f"device={device}, tau_max={tau_max}"
    )

    print(
        f"  [TCDF] preprocessing="
        f"{preprocessing_name}, "
        f"median trend={median_trend_strength:.3f}, "
        f"stability seeds={replica_seeds}"
    )

    try:
        all_replica_edges = []

        total_start = time.time()

        for replica_number, replica_seed in enumerate(
            replica_seeds,
            start=1,
        ):
            replica_start = time.time()

            print(
                f"  [TCDF] Replica "
                f"{replica_number}/"
                f"{len(replica_seeds)}, "
                f"seed={replica_seed}"
            )

            replica_edges = (
                _run_single_replica(
                    replica_seed
                )
            )

            all_replica_edges.append(
                replica_edges
            )

            print(
                f"    replica found "
                f"{len(replica_edges)} edge(s) "
                f"in "
                f"{time.time() - replica_start:.1f}s"
            )

        pair_occurrences = defaultdict(list)

        for replica_index, replica_edges in enumerate(
            all_replica_edges
        ):
            seen_pairs = set()

            for edge in replica_edges:
                pair = (
                    edge["cause"],
                    edge["effect"],
                )

                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)

                pair_occurrences[
                    pair
                ].append(edge)

        stable_edges = []

        number_of_replicas = len(
            replica_seeds
        )

        for pair, matching_edges in (
            pair_occurrences.items()
        ):
            # Strict 2-of-2 consensus removes seed-specific edges.
            if (
                len(matching_edges)
                != number_of_replicas
            ):
                continue

            lags = [
                int(
                    edge["label"].split(
                        "TCDF_LAG"
                    )[-1]
                )
                for edge in matching_edges
            ]

            selected_lag = (
                Counter(lags)
                .most_common(1)[0][0]
            )

            average_confidence = float(
                np.mean(
                    [
                        edge["confidence"]
                        for edge in matching_edges
                    ]
                )
            )

            stable_edges.append(
                {
                    "cause": pair[0],
                    "effect": pair[1],
                    "label": (
                        f"TCDF_LAG"
                        f"{selected_lag}"
                    ),
                    "confidence": round(
                        min(
                            1.0,
                            average_confidence,
                        ),
                        3,
                    ),
                }
            )

        stable_edges.sort(
            key=lambda edge: (
                columns.index(
                    edge["effect"]
                ),
                columns.index(
                    edge["cause"]
                ),
            )
        )

        print(
            f"  [TCDF] "
            f"{len(stable_edges)} stable "
            f"directed edge(s) found "
            f"in "
            f"{time.time() - total_start:.1f}s"
        )

        return stable_edges

    finally:
        if not cuda:
            torch.set_num_threads(
                old_thread_count
            )


def run_dag_gnn(
    df,
    columns,
    hidden_dims=32,
    z_dims=1,
    lr=3e-3,
    tau_a=0.0,
    trace_penalty=100.0,
    graph_threshold=0.3,
    inner_epochs=50,
    k_max_iter=20,
    c_init=1.0,
    c_max=1e20,
    h_tol=1e-6,
    max_total_blocks=40,
    seed=42,
    batch_size=512,
    preprocess="none",
    enforce_dag=True,
    gradient_clip=5.0,
):
    """DAG-GNN (Yu et al., ICML 2019) on i.i.d. tabular observations.

    A single learned d x d adjacency ``A`` routes every variable's data
    through a shared encoder/decoder, trained with an augmented-Lagrangian
    schedule against a polynomial acyclicity constraint
    ``h(A) = tr[(I + A∘A/d)^d] - d``. No time/lag awareness (like PC/LiNGAM)
    — treats rows as i.i.d. samples.

    Architecture notes
    ------------------
    * The adjacency is reparameterized as ``sinh(3*raw_A)``, which amplifies
      small values and accelerates convergence (the reference implementation's
      own rationale, github.com/fishmoon1234/DAG-GNN).
    * Despite the encoder/decoder naming, this is a deterministic SEM
      autoencoder, not a real VAE: there is no logvar/sampling step, so the
      "KL" term is an L2 pull toward zero rather than a true posterior KL
      divergence, and the reconstruction loss uses a fixed variance,
      collapsing it to plain scaled MSE.
    * A shared per-node offset ``Wa`` is added before the graph operation and
      subtracted after, in both encoder and decoder.
    * ``h(A) == 0`` is a property of the continuous-relaxation weight matrix,
      not a guarantee that the thresholded *binary* graph is acyclic
      (a known weakness of the whole NOTEARS/DAG-GNN continuous-relaxation
      family) — this is why ``enforce_dag`` exists as a separate, explicit
      post-thresholding step rather than being assumed automatic.

    This is a backward-compatible replacement for the app's previous
    ``run_dag_gnn`` function.

    Compatibility guarantees
    ------------------------
    * The original function name is unchanged.
    * The original argument order through ``seed`` is unchanged.
    * ``trace_penalty`` is still accepted so old callers do not fail.
    * The return value remains a list of dictionaries containing only:
      ``cause``, ``effect``, ``label``, and ``confidence``.

    Parameters added at the end
    ---------------------------
    batch_size : int or None
        Mini-batch size. The tested benchmark setting is 100. Use None for
        full-batch training.
    preprocess : {'none', 'center', 'standardize'}
        ``none`` is the validated setting for synthetic_iid.csv.
    enforce_dag : bool
        Remove the weakest edge from cycles after thresholding.
    gradient_clip : float or None
        Gradient clipping threshold.

    Hyperparameter provenance
    --------------------------
    ``graph_threshold=0.3`` is the cutoff applied to the learned continuous
    adjacency matrix's edge weights after training -- an edge survives only
    if its |weight| exceeds this value, so it directly controls the
    precision/recall tradeoff of the final graph (lower = more edges kept,
    including weaker/noisier ones). ``0.3`` matches the reference
    implementation's own reported operating point, not a value independently
    tuned against a benchmark in this codebase.

    Inside the augmented-Lagrangian training loop, the schedule doubles its
    penalty coefficient ``c_A`` whenever the acyclicity violation ``h(A)``
    hasn't shrunk enough between outer steps -- specifically when
    ``h_new > 0.25 * h_old`` (i.e. it dropped by less than 75%). ``0.25`` is
    the reference implementation's own convergence-rate threshold for
    deciding "not shrinking fast enough, escalate the penalty."

    The remaining hyperparameters (``hidden_dims``, ``z_dims``, ``lr``,
    ``tau_a``, ``inner_epochs``, ``k_max_iter``, ``c_init``, ``c_max``,
    ``h_tol``) are not independently tuned in this codebase -- they match
    the reference implementation's own argparse defaults (train.py) exactly,
    a deliberate choice: re-tuning any of them against a benchmark's known
    ground truth would itself be a form of test-set leakage into model
    selection. ``max_total_blocks=40`` is the one addition not in the
    reference: a runtime safety cap on the total number of ``inner_epochs``
    training blocks, since the reference's own ``k_max_iter`` has no
    equivalent hard stop; in practice every run here converges well before
    this cap is reached. ``gradient_clip=5.0`` is likewise not in the
    reference and was added defensively against occasional loss spikes
    during the early augmented-Lagrangian steps, not because the reference
    needed it.

    Notes
    -----
    ``trace_penalty`` is retained only for compatibility. Because the
    adjacency diagonal is hard-masked to zero, the old
    trace(A elementwise-multiplied by A) term is always zero and is therefore
    intentionally not added to the loss. (In the original reference
    implementation this term penalized self-loop magnitude specifically, not
    general sparsity — trace of an elementwise square only sums the diagonal.)
    """
    import math
    import random

    import networkx as nx
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Kept only so existing app calls that pass this argument remain valid.
    _ = trace_penalty

    # ── Validate input without changing the app-facing API ────────────────
    columns = list(columns)
    if len(columns) < 2:
        raise ValueError("DAG-GNN requires at least two variables.")

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Column(s) not found: {missing_columns}")

    data = df.loc[:, columns].dropna().copy()
    if data.shape[0] < 20:
        raise ValueError(
            f"Not enough complete rows ({data.shape[0]}) to run DAG-GNN "
            "reliably (need >= 20)."
        )

    nonnumeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if nonnumeric:
        raise TypeError(f"DAG-GNN requires numeric columns: {nonnumeric}")

    standard_deviations = data.std(axis=0, ddof=0)
    constant_columns = standard_deviations[
        standard_deviations <= 1e-12
    ].index.tolist()
    if constant_columns:
        raise ValueError(
            f"Constant or nearly constant column(s): {constant_columns}"
        )

    preprocess = str(preprocess).lower()
    if preprocess == "none":
        prepared_data = data
    elif preprocess == "center":
        prepared_data = data - data.mean(axis=0)
    elif preprocess == "standardize":
        prepared_data = (
            data - data.mean(axis=0)
        ) / standard_deviations
    else:
        raise ValueError(
            "preprocess must be 'none', 'center', or 'standardize'."
        )

    # ── Reproducibility ───────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = len(columns)
    n_samples = len(prepared_data)

    if batch_size is None:
        effective_batch_size = n_samples
    else:
        effective_batch_size = min(max(1, int(batch_size)), n_samples)

    print(
        f"  [DAG-GNN] {n_samples} rows x {d} variables, "
        f"device={device}, batch={effective_batch_size}, "
        f"preprocess={preprocess}, seed={seed}"
    )

    values = prepared_data.to_numpy(dtype=np.float32)
    cpu_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(cpu_tensor),
        batch_size=effective_batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
    )

    eye = torch.eye(d, dtype=torch.float32, device=device)
    offdiag_mask = 1.0 - eye

    # ── DAG-GNN encoder and decoder ───────────────────────────────────────
    class _GraphEncoder(nn.Module):
        """
        Maps each variable's raw scalar value to a z_dim-dimensional
        latent representation (2-layer MLP, fc1 -> ReLU -> fc2), then
        applies the learned graph_operator (a function of the adjacency
        matrix A -- see the acyclicity-constrained training loop below)
        to mix information according to the current causal-graph
        hypothesis. Wa is the shared per-node offset mentioned in
        run_dag_gnn's docstring above: added before the graph operation,
        subtracted after, matching the reference implementation.
        """
        def __init__(self, hidden, z_dim):
            super().__init__()
            self.fc1 = nn.Linear(1, hidden)
            self.fc2 = nn.Linear(hidden, z_dim)
            self.Wa = nn.Parameter(torch.zeros(z_dim))

            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, inputs, graph_operator):
            hidden = torch.relu(self.fc1(inputs))
            latent = self.fc2(hidden)
            return torch.matmul(
                graph_operator,
                latent + self.Wa,
            ) - self.Wa

    class _GraphDecoder(nn.Module):
        """
        Inverse of _GraphEncoder above: un-mixes the latent representation
        via inverse_operator (undoing the graph operation applied during
        encoding) and maps it back down to a reconstructed scalar value
        per variable (2-layer MLP, fc1 -> ReLU -> fc2). Comparing this
        reconstruction against the real input is what drives the training
        loss described in run_dag_gnn's docstring above (a deterministic
        SEM autoencoder, not a sampling-based VAE).
        """
        def __init__(self, hidden, z_dim):
            super().__init__()
            self.fc1 = nn.Linear(z_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)

            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, latent, inverse_operator, wa):
            transformed = torch.matmul(
                inverse_operator,
                latent + wa,
            ) - wa
            return self.fc2(torch.relu(self.fc1(transformed)))

    # Reference-style zero adjacency initialization.
    raw_adj = nn.Parameter(
        torch.zeros(d, d, dtype=torch.float32, device=device)
    )
    encoder = _GraphEncoder(hidden_dims, z_dims).to(device)
    decoder = _GraphDecoder(hidden_dims, z_dims).to(device)

    trainable_parameters = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + [raw_adj]
    )
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr)

    def _current_adjacency():
        # A[i, j] is interpreted as columns[i] -> columns[j].
        return torch.sinh(3.0 * raw_adj) * offdiag_mask

    def _h_acyclic(adjacency):
        matrix = eye + (adjacency * adjacency) / d
        return torch.trace(torch.linalg.matrix_power(matrix, d)) - d

    def _update_optimizer(c_a):
        # Reference DAG-GNN learning-rate schedule.
        estimated_lr = lr / (math.log10(c_a) + 1e-10)
        current_lr = min(max(estimated_lr, 1e-4), 1e-2)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr
        return current_lr

    def _soft_threshold(parameter, threshold):
        return torch.sign(parameter) * torch.clamp(
            torch.abs(parameter) - threshold,
            min=0.0,
        )

    # ── Augmented-Lagrangian training ─────────────────────────────────────
    c_a = float(c_init)
    lambda_a = 0.0
    h_old = float("inf")
    h_new = 1.0
    total_blocks = 0
    adjacency = _current_adjacency().detach()

    for _outer_iteration in range(k_max_iter):
        while c_a < c_max:
            current_lr = _update_optimizer(c_a)
            last_loss = float("nan")

            for _ in range(inner_epochs):
                for (batch_cpu,) in loader:
                    batch = batch_cpu.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    adjacency = _current_adjacency()
                    graph_operator = eye - adjacency.transpose(0, 1)
                    latent = encoder(batch, graph_operator)

                    try:
                        inverse_operator = torch.linalg.inv(graph_operator)
                    except RuntimeError:
                        inverse_operator = torch.linalg.pinv(graph_operator)

                    predictions = decoder(
                        latent,
                        inverse_operator,
                        encoder.Wa,
                    )

                    loss_nll = (
                        ((predictions - batch) ** 2) / 2.0
                    ).sum() / batch.shape[0]
                    loss_kl = 0.5 * (
                        latent * latent
                    ).sum() / batch.shape[0]
                    h_value = _h_acyclic(adjacency)
                    sparse_loss = tau_a * torch.sum(torch.abs(adjacency))

                    loss = (
                        loss_kl
                        + loss_nll
                        + lambda_a * h_value
                        + 0.5 * c_a * h_value * h_value
                        + sparse_loss
                    )

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "DAG-GNN produced a non-finite loss. "
                            "Try preprocess='center', reduce lr, or inspect "
                            "the variable scales."
                        )

                    loss.backward()
                    if gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters,
                            float(gradient_clip),
                        )
                    optimizer.step()

                    with torch.no_grad():
                        if tau_a > 0.0:
                            raw_adj.copy_(
                                _soft_threshold(
                                    raw_adj,
                                    tau_a * current_lr,
                                )
                            )
                        raw_adj.mul_(offdiag_mask)

                    last_loss = float(loss.detach().item())

            total_blocks += 1

            with torch.no_grad():
                adjacency = _current_adjacency()
                h_new = float(_h_acyclic(adjacency).item())

            if total_blocks >= max_total_blocks:
                break

            # 0.25 is the reference implementation's own convergence-rate
            # threshold -- see run_dag_gnn's "Hyperparameter provenance"
            # docstring section above.
            if h_new > 0.25 * h_old:
                c_a *= 10.0
            else:
                break

        h_old = h_new
        lambda_a += c_a * h_new

        if (
            h_new <= h_tol
            or total_blocks >= max_total_blocks
            or c_a >= c_max
        ):
            break

    print(
        f"  [DAG-GNN] schedule stopped after {total_blocks} block(s) "
        f"({total_blocks * inner_epochs} epochs), c_a={c_a:.2e}, "
        f"final h(A)={h_new:.2e}"
    )

    adjacency_np = adjacency.detach().cpu().numpy()

    # ── Threshold and optionally enforce a DAG ──────────────────────────────
    # (internal weight is removed before returning, to preserve app schema)
    internal_edges = []
    for cause_index in range(d):
        for effect_index in range(d):
            if cause_index == effect_index:
                continue

            weight = float(adjacency_np[cause_index, effect_index])
            magnitude = abs(weight)
            # graph_threshold=0.3 matches the reference implementation's own
            # operating point -- see run_dag_gnn's "Hyperparameter provenance"
            # docstring section above.
            if magnitude > graph_threshold:
                internal_edges.append(
                    {
                        "cause": columns[cause_index],
                        "effect": columns[effect_index],
                        "label": "DAGGNN_CAUSES",
                        "confidence": round(min(1.0, magnitude), 3),
                        "_magnitude": magnitude,
                    }
                )

    removed_cycles = 0
    if enforce_dag:
        while True:
            graph = nx.DiGraph()
            graph.add_nodes_from(columns)
            for edge in internal_edges:
                graph.add_edge(
                    edge["cause"],
                    edge["effect"],
                    magnitude=edge["_magnitude"],
                )

            try:
                cycle = nx.find_cycle(graph, orientation="original")
            except nx.NetworkXNoCycle:
                break

            cycle_pairs = [(item[0], item[1]) for item in cycle]
            weakest = min(
                cycle_pairs,
                key=lambda pair: graph[pair[0]][pair[1]]["magnitude"],
            )
            internal_edges = [
                edge
                for edge in internal_edges
                if not (
                    edge["cause"] == weakest[0]
                    and edge["effect"] == weakest[1]
                )
            ]
            removed_cycles += 1

    edges = [
        {
            "cause": edge["cause"],
            "effect": edge["effect"],
            "label": edge["label"],
            "confidence": edge["confidence"],
        }
        for edge in internal_edges
    ]

    final_graph = nx.DiGraph()
    final_graph.add_nodes_from(columns)
    final_graph.add_edges_from(
        (edge["cause"], edge["effect"])
        for edge in edges
    )
    is_dag = nx.is_directed_acyclic_graph(final_graph)

    print(
        f"  [DAG-GNN] {len(edges)} directed edge(s) kept at "
        f"threshold={graph_threshold}; removed_cycles={removed_cycles}; "
        f"is_dag={is_dag}"
    )

    return edges


def run_lpcmci(df: pd.DataFrame, columns: list, tau_min: int = 0, tau_max: int = 2, alpha: float = 0.05) -> list:
    """
    LPCMCI (Gerhardus & Runge, NeurIPS 2020) — latent-confounder-aware causal
    discovery for time series, via tigramite.lpcmci.LPCMCI. Requires a
    time-ordered dataset (row order = time order), same requirement as
    PCMCI/PCMCI+/TCDF. Unlike them, additionally distinguishes "A causes B"
    from "A and B share an unmeasured common cause" via a PAG-style output
    (bidirected '<->' marks) — the lagged-time-series counterpart of
    run_fci's i.i.d. bidirected-mark handling. tigramite itself documents
    this method as still EXPERIMENTAL (default hyperparameters still being
    tuned upstream).

    tau_min=0 (LPCMCI's own actual
    default) — tau_min=1 silently never tests contemporaneous (lag-0)
    relationships at all, not just hides them, which can miss real
    confounded pairs that only manifest at lag 0.
    '<--' marks are fully-resolved directed edges too, with cause/effect
    reversed from '-->'. Per tigramite's own mark conventions,
    '<--'/'<?-'/'<-o'/'<?o'/'o-o' only ever appear at lag 0, so only
    lag-0 entries need this reverse-role handling.

    Known caveat: runtime scales badly with tau_max (~5s at tau_max=2 vs.
    ~104s at tau_max=5 on a representative 6-variable/1826-row dataset).
    Default kept lower than run_pcmci/run_pcmci_plus's tau_max=5 for this
    reason, not because LPCMCI needs less lag depth in principle.

    Bidirected pairs are reported to the terminal as a latent-confounder
    diagnostic — same "report, don't fabricate" policy run_fci already
    uses for its own bidirected marks — but NOT turned into edges, since
    doing so would misrepresent what LPCMCI actually found (no direct
    causal claim either way).

    Returns list of {cause, effect, label, confidence} with cause/effect as
    DATASET COLUMN names.
    """
    edges, _graph, _val_matrix, _var_names = _run_lpcmci_impl(
        df, columns, tau_min=tau_min, tau_max=tau_max, alpha=alpha,
    )
    return edges


def run_lpcmci_full(
    df: pd.DataFrame,
    columns: list,
    tau_min: int = 0,
    tau_max: int = 2,
    alpha: float = 0.05,
) -> dict:
    """
    Same computation as run_lpcmci(), run exactly once, but also returns the
    raw Tigramite artifacts needed to draw the full causal graph — every
    link type LPCMCI found, including the bidirected '<->' latent-confounder
    marks and undetermined marks that run_lpcmci() reports but doesn't turn
    into edges.

    Returns:
        {
            "edges": [...],       # identical to run_lpcmci()'s return value
            "graph": np.ndarray,      # shape [N, N, tau_max+1], all link marks
            "val_matrix": np.ndarray, # cross-MCI / auto-MCI strengths
            "var_names": [...],       # == columns, axis order for the arrays above
        }
    """
    edges, graph, val_matrix, var_names = _run_lpcmci_impl(
        df, columns, tau_min=tau_min, tau_max=tau_max, alpha=alpha,
    )
    return {
        "edges": edges,
        "graph": graph,
        "val_matrix": val_matrix,
        "var_names": var_names,
    }


def _run_lpcmci_impl(
    df: pd.DataFrame,
    columns: list,
    tau_min: int,
    tau_max: int,
    alpha: float,
) -> tuple:
    """Shared implementation behind run_lpcmci() / run_lpcmci_full() — see
    run_lpcmci()'s docstring for the method's rationale and caveats. Returns
    (edges, graph, val_matrix, columns)."""
    import tigramite.data_processing as pp
    from tigramite.lpcmci import LPCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from sklearn.preprocessing import StandardScaler

    raw_frame = df.loc[:, columns].copy()

    nonnumeric_columns = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(raw_frame[column])
    ]
    if nonnumeric_columns:
        raise TypeError(f"LPCMCI requires numeric columns: {nonnumeric_columns}")

    # Interpolate rather than drop incomplete rows: dropna() silently splices
    # non-adjacent timepoints together across any gap, which breaks the fixed
    # time-lag assumption LPCMCI relies on for every row (same fix applied to
    # PCMCI/PCMCI+ in _run_pcmci_family).
    data = raw_frame.interpolate(method="linear", limit_direction="both")

    all_nan_columns = [
        column
        for column in columns
        if data[column].isna().all()
    ]
    if all_nan_columns:
        raise ValueError(
            f"Column(s) with no valid values to interpolate from: {all_nan_columns}"
        )

    if data.shape[0] < 30:
        raise ValueError(f"Not enough complete rows ({data.shape[0]}) to run LPCMCI reliably (need >= 30).")

    data_scaled = StandardScaler().fit_transform(data.values)

    print(f"  [LPCMCI] {data.shape[0]} rows x {len(columns)} variables, "
          f"tau_min={tau_min}, tau_max={tau_max}, alpha={alpha}")
    print("  [LPCMCI] NOTE: tigramite documents this method as still EXPERIMENTAL "
          "(default hyperparameters still being tuned upstream).")

    tigramite_df = pp.DataFrame(data_scaled, var_names=columns)
    lpcmci = LPCMCI(dataframe=tigramite_df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    results = lpcmci.run_lpcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha)

    graph = results["graph"]
    val_matrix = results["val_matrix"]

    n = len(columns)
    directed = {}
    bidirected_pairs = set()
    undetermined_count = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for tau in range(0, tau_max + 1):
                mark = graph[i, j, tau]
                if not mark:
                    continue
                val = float(val_matrix[i, j, tau])
                if mark == "-->":
                    key = (columns[i], columns[j])
                    if key not in directed or abs(val) > abs(directed[key][1]):
                        directed[key] = (tau, val)
                elif mark == "<--":
                    key = (columns[j], columns[i])
                    if key not in directed or abs(val) > abs(directed[key][1]):
                        directed[key] = (tau, val)
                elif mark == "<->":
                    bidirected_pairs.add(frozenset((columns[i], columns[j])))
                else:
                    undetermined_count += 1

    edges = []
    for (src, dst), (tau, val) in directed.items():
        edges.append({
            'cause': src,
            'effect': dst,
            'label': f'LPCMCI_LAG{tau}',
            'confidence': round(min(1.0, abs(val)), 3),
        })

    print(f"  [LPCMCI] {len(edges)} directed edge(s) kept"
          + (f", {len(bidirected_pairs)} latent-confounded pair(s) (possible "
             "unmeasured common cause) — not turned into edges" if bidirected_pairs else "")
          + (f", {undetermined_count} undetermined mark(s) — dropped" if undetermined_count else ""))
    if bidirected_pairs:
        print("  [LPCMCI] Latent-confounded pairs (reported, not scored as edges):")
        for pair in bidirected_pairs:
            a, b = tuple(pair)
            print(f"    {a} <-> {b}")

    return edges, graph, val_matrix, columns


def run_ges(df: pd.DataFrame, columns: list, score_func: str = "local_score_BIC", lambda_value: float = 0.5) -> list:
    """
    Greedy Equivalence Search (Chickering, 2002) — score-based causal
    discovery, via causal-learn's ScoreBased.GES.ges. Greedily adds then
    removes edges to maximize a global score (BIC by default) over the
    space of Markov equivalence classes, unlike PC/CD-NOD's sequential
    conditional-independence testing — a genuinely different search
    paradigm from every other i.i.d. method in this module. No time/lag
    awareness (like PC) — treats rows as i.i.d. samples.

    GES's own GeneralGraph output uses the identical endpoint-mark
    convention as run_pc's — graph[j,i]==1 and graph[i,j]==-1 means
    i --> j — so this reuses run_pc's exact classification/drop-undirected
    policy.

    Known limitation (not a bug in this port): benchmarked against a
    synthetic ground-truth DAG and the Sachs protein-signaling dataset
    (a standard causal-discovery benchmark), GES found
    fewer correct directed edges than plain run_pc on both, even after
    sweeping lambda_value (BIC sparsity penalty) from 0.5-8 — denser/
    sparser settings shrank correct and incorrect edges roughly
    proportionally rather than improving precision. A separate comparison
    against causal-learn's exact A* search (ScoreBased.ExactSearch, not
    wired into this codebase) found 7/8 correct on the same synthetic data
    vs. GES's 1/8, indicating GES's greedy hill-climbing is very likely
    getting stuck in a local optimum on that benchmark specifically, rather
    than BIC itself being a poor score there. GES is included over
    ExactSearch/BOSS/GRaSP as the most established, widely-recognized
    score-based method in the literature — a deliberate choice for
    method-suite breadth and citability, not a claim that it is the
    strongest empirical performer of the four on these benchmarks.

    Returns list of {cause, effect, label, confidence} with cause/effect as
    DATASET COLUMN names.
    """
    from causallearn.search.ScoreBased.GES import ges

    data = df[columns].dropna().to_numpy()
    if data.shape[0] < 20:
        raise ValueError(f"Not enough complete rows ({data.shape[0]}) to run GES reliably (need >= 20).")

    print(f"  [GES] {data.shape[0]} rows x {len(columns)} variables, "
          f"score_func={score_func}, lambda_value={lambda_value}")
    record = ges(data, score_func=score_func, node_names=columns, lambda_value=lambda_value)
    adj = record["G"].graph

    edges = []
    undirected_count = 0
    n = len(columns)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[j, i] == 1 and adj[i, j] == -1:  # tail at i, arrow at j => i -> j
                edges.append({
                    'cause': columns[i],
                    'effect': columns[j],
                    'label': 'GES_CAUSES',
                    'confidence': 1.0,
                })
            elif adj[i, j] == -1 and adj[j, i] == -1 and i < j:  # undirected, count once per pair
                undirected_count += 1

    print(f"  [GES] {len(edges)} directed edge(s) kept"
          + (f", {undirected_count} undirected pair(s) found but not orientable — dropped" if undirected_count else "")
          + f", BIC={record['score']:.1f}")
    return edges


# ── Map statistical edges back to KG variable names ─────────────────────────

def edges_to_kg_space(edges: list, mapping: dict) -> list:
    """
    Statistical edges come back with cause/effect as DATASET COLUMN names
    (e.g. 'SIC', 'T2M'). Reverse-maps them to the KG node names a user
    actually recognizes (e.g. 'sea ice concentration', 'surface temperature'),
    using the same {kg_node: dataset_column_or_None} mapping confirmed by the
    user before running discovery. Edges whose column has no KG node mapped
    to it (shouldn't happen, since columns come from the mapping itself) are
    dropped defensively.
    """
    reverse = {col: node for node, col in mapping.items() if col}
    return [
        {**e, 'cause': reverse[e['cause']], 'effect': reverse[e['effect']]}
        for e in edges
        if e['cause'] in reverse and e['effect'] in reverse
    ]
