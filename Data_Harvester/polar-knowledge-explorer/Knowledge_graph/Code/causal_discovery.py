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
    Constraint-based PC algorithm (causal-learn). Treats rows as i.i.d. samples —
    no time/lag awareness. Many true causal edges come back undirected (PC can only
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
    FCI (Fast Causal Inference, causal-learn). Like PC, treats rows as i.i.d.
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
    CD-NOD (causal-learn) — constraint-based discovery like PC, but augments
    the dataset with a context index (c_indx) representing possible
    heterogeneity/nonstationarity across rows, and can additionally flag
    which variables' causal mechanisms appear to depend on that index.
    Treats rows as i.i.d. samples otherwise — no time/lag awareness beyond
    the context index itself.

    c_indx is row order (np.arange), matching the convention PCMCI/TCDF
    already use elsewhere in this module when a real timestamp isn't
    threaded through separately (see cd_algorithm/CD-NOD.ipynb's synthetic
    and Sachs benchmark cells, which used the same row-order proxy — its
    real-timeseries cell used actual elapsed days instead, but the directed
    edges returned here don't depend on which proxy is used, only the
    nonstationarity diagnostic below would).

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
    DirectLiNGAM (Linear Non-Gaussian Acyclic Model) — full DAG orientation from
    i.i.d. data, complementary to PC: PC only orients v-structure/collider edges
    and often returns nothing for simple chains (e.g. X->Y->Z); LiNGAM uses the
    assumption that noise is non-Gaussian to fully orient every edge instead.
    No time/lag awareness (like PC) — treats rows as i.i.d. samples.

    LiNGAM assigns a coefficient to nearly every variable pair in its fitted
    causal order rather than doing sparse structure learning, so keeping every
    nonzero coefficient over-connects badly (36 edges kept vs. 18 true edges
    on the Sachs benchmark in cd_algorithm/LiNGAM.ipynb at threshold 1e-3).
    min_coefficient=0.1 was tuned against that same benchmark (20 edges vs.
    18 true) as a much closer default.

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


def _run_pcmci_family(df: pd.DataFrame, columns: list, tau_min: int, tau_max: int, alpha: float, plus: bool) -> list:
    from sklearn.preprocessing import StandardScaler
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    method_name = "PCMCI+" if plus else "PCMCI"

    data = df[columns].dropna().to_numpy()
    if data.shape[0] < 30:
        raise ValueError(f"Not enough complete rows ({data.shape[0]}) to run PCMCI reliably (need >= 30).")

    tau_max = max(1, min(tau_max, data.shape[0] // 10))
    print(f"  [{method_name}] {data.shape[0]} rows x {len(columns)} variables, tau_max={tau_max}, alpha={alpha}, fdr_method=fdr_bh")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tigramite_df = pp.DataFrame(data_scaled, var_names=columns)

    pcmci = PCMCI(dataframe=tigramite_df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)

    # fdr_method='fdr_bh' applies Benjamini-Hochberg correction across all
    # tested (variable, lag) combinations. Without it, testing many pairs at
    # a fixed alpha produces weak spurious "significant" reverse-direction
    # edges from multiple-testing noise alone — a synthetic X->Y, Y->Z-only
    # dataset produced a fake Y->X edge at raw alpha=0.05 (val=0.10), which
    # fdr_bh correctly drops.
    if plus:
        results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha, fdr_method='fdr_bh')
    else:
        results = pcmci.run_pcmci(tau_min=max(tau_min, 1), tau_max=tau_max, pc_alpha=alpha,
                                   alpha_level=alpha, fdr_method='fdr_bh')

    graph = results['graph']
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']

    # Keep the strongest lag per (source, target) pair so the comparison step
    # sees one edge per variable pair, not one per lag. Filtering on graph
    # marks (not just p_matrix < alpha) matters most at lag 0: a contemporaneous
    # link is one symmetric test, so p_matrix[i,j,0] == p_matrix[j,i,0] always —
    # raw threshold-checking both directions independently double-counts it as
    # two edges. tigramite's own orientation step (collider/v-structure rules)
    # already resolves this to a single direction, marked '-->' one way and
    # '<--' the other; keeping only '-->' marks uses that resolved direction
    # instead of re-deriving (and duplicating) it from the symmetric p-value.
    best = {}
    n = len(columns)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for tau in range(max(tau_min, 0 if plus else 1), tau_max + 1):
                if graph[i, j, tau] != '-->':
                    continue
                val = val_matrix[i, j, tau]
                key = (columns[i], columns[j])
                if key not in best or abs(val) > abs(best[key][1]):
                    best[key] = (round(float(p_matrix[i, j, tau]), 4), float(val), tau)

    edges = []
    for (source, target), (pval, val, tau) in best.items():
        edges.append({
            'cause': source,
            'effect': target,
            'label': f"PCMCI{'_PLUS' if plus else ''}_LAG{tau}",
            'confidence': round(min(1.0, abs(val)), 3),
        })
    print(f"  [{method_name}] {len(edges)} directed edge(s) kept after FDR correction")
    return edges


def run_pcmci(df: pd.DataFrame, columns: list, tau_max: int = 5, alpha: float = 0.05) -> list:
    """Lagged-only causal discovery (tigramite PCMCI). Requires a time-ordered dataset."""
    return _run_pcmci_family(df, columns, tau_min=1, tau_max=tau_max, alpha=alpha, plus=False)


def run_pcmci_plus(df: pd.DataFrame, columns: list, tau_max: int = 5, alpha: float = 0.05) -> list:
    """Lagged + contemporaneous causal discovery (tigramite PCMCI+). Requires a time-ordered dataset."""
    return _run_pcmci_family(df, columns, tau_min=0, tau_max=tau_max, alpha=alpha, plus=True)


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
    Temporal Causal Discovery Framework (TCDF) — ported from cd_algorithm/tcdf.ipynb,
    faithful to the original M-Nauta/TCDF depthwise dilated-causal-CNN + attention +
    PIVM (permutation importance) significance test. Requires a time-ordered dataset
    (row order = time order), same requirement as PCMCI/PCMCI+.

    Differs from the notebook in one way: no STL deseasonalization. The notebook's
    STL step assumed daily data with an annual cycle (period=365) specific to its
    Dallas air-quality dataset, which doesn't generalize to arbitrary uploaded
    datasets here, and needs statsmodels (not installed). Trains directly on
    standardized data instead — same approach PC/PCMCI/PCMCI+ already use.

    Trains one small depthwise causal CNN per target variable to predict it from
    all other selected variables; attention weights (gap-based selection) propose
    candidate causes, PIVM (shuffle each candidate's series, check if prediction
    degrades) validates them, and convolution kernel weights are read to estimate
    the lag. Self-causation edges are skipped (consistent with the rest of the app
    never emitting self-loops).

    Returns list of {cause, effect, label, confidence}, cause/effect as DATASET
    COLUMN names (same convention as run_pc/run_pcmci/run_pcmci_plus).
    """
    import heapq
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    data = df[columns].dropna().to_numpy()
    n_rows, n_vars = data.shape
    if n_rows < 30:
        raise ValueError(f"Not enough complete rows ({n_rows}) to run TCDF reliably (need >= 30).")
    if n_vars < 2:
        raise ValueError("Need at least 2 columns to run TCDF.")

    tau_max = max(1, min(tau_max, n_rows // 10))
    cuda = torch.cuda.is_available()

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

    from sklearn.preprocessing import StandardScaler
    data_scaled = StandardScaler().fit_transform(data).astype(np.float32)

    class _Chomp1d(nn.Module):
        """Removes future padding to enforce causality."""
        def __init__(self, chomp_size):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            return x[:, :, :-self.chomp_size].contiguous()

    class _FirstBlock(nn.Module):
        def __init__(self, n_inputs, kernel_size, dilation, padding):
            super().__init__()
            self.conv1 = nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=1,
                                    padding=padding, dilation=dilation, groups=n_inputs)
            self.net = nn.Sequential(self.conv1, _Chomp1d(padding))
            self.relu = nn.PReLU(n_inputs)
            self.conv1.weight.data.normal_(0, 0.1)

        def forward(self, x):
            return self.relu(self.net(x))

    class _TemporalBlock(nn.Module):
        def __init__(self, n_inputs, kernel_size, dilation, padding):
            super().__init__()
            self.conv1 = nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=1,
                                    padding=padding, dilation=dilation, groups=n_inputs)
            self.net = nn.Sequential(self.conv1, _Chomp1d(padding))
            self.relu = nn.PReLU(n_inputs)
            self.conv1.weight.data.normal_(0, 0.1)

        def forward(self, x):
            return self.relu(self.net(x) + x)

    class _LastBlock(nn.Module):
        def __init__(self, n_inputs, kernel_size, dilation, padding):
            super().__init__()
            self.conv1 = nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=1,
                                    padding=padding, dilation=dilation, groups=n_inputs)
            self.net = nn.Sequential(self.conv1, _Chomp1d(padding))
            self.linear = nn.Linear(n_inputs, n_inputs)
            self.linear.weight.data.normal_(0, 0.01)

        def forward(self, x):
            out = self.net(x)
            return self.linear((out + x).transpose(1, 2)).transpose(1, 2)

    class _DepthwiseCausalNet(nn.Module):
        def __init__(self, num_inputs, num_levels, kernel_size, dilation_c):
            super().__init__()
            layers = []
            for level in range(num_levels):
                dilation_size = dilation_c ** level
                padding = (kernel_size - 1) * dilation_size
                if level == 0:
                    layers.append(_FirstBlock(num_inputs, kernel_size, dilation_size, padding))
                elif level == num_levels - 1:
                    layers.append(_LastBlock(num_inputs, kernel_size, dilation_size, padding))
                else:
                    layers.append(_TemporalBlock(num_inputs, kernel_size, dilation_size, padding))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class _ADDSTCN(nn.Module):
        """fs_attention selects source variables; depthwise causal net + pointwise head predicts the target."""
        def __init__(self, input_size, num_levels, kernel_size, dilation_c, cuda):
            super().__init__()
            self.dwn = _DepthwiseCausalNet(input_size, num_levels, kernel_size, dilation_c)
            self.pointwise = nn.Conv1d(input_size, 1, 1)
            self.fs_attention = nn.Parameter(torch.ones(input_size, 1))
            if cuda:
                self.dwn = self.dwn.cuda()
                self.pointwise = self.pointwise.cuda()

        def forward(self, x):
            y1 = self.dwn(x * F.softmax(self.fs_attention, dim=0))
            return self.pointwise(y1).transpose(1, 2)

    def _prepare_data(arr, target_idx):
        T, V = arr.shape
        x = arr.copy()
        y = arr[:, target_idx].copy()
        lagged = np.zeros(T, dtype=np.float32)
        lagged[1:] = arr[:-1, target_idx]
        x[:, target_idx] = lagged  # avoid self-leak: target column in X is its own lag-1
        X = torch.from_numpy(x.T[np.newaxis, :, :])
        Y = torch.from_numpy(y[np.newaxis, :, np.newaxis])
        return X, Y

    def _select_potentials_by_gap(scores_np):
        s = sorted(scores_np.tolist(), reverse=True)
        indices = np.argsort(-scores_np).tolist()
        if len(s) <= 5:
            return [i for i in indices if scores_np[i] > 1.0]
        gaps = []
        for i in range(len(s) - 1):
            if s[i] < 1.0:
                break
            gaps.append(s[i] - s[i + 1])
        if not gaps:
            return []
        sort_gaps = sorted(gaps, reverse=True)
        ind = -1
        for g in sort_gaps:
            idx_gap = gaps.index(g)
            if 0 < idx_gap < (len(s) - 1) / 2:
                ind = idx_gap
                break
        if ind < 0:
            ind = 0
        return indices[:ind + 1]

    edges = []
    n = len(columns)
    log_interval = max(1, epochs // 5)

    print(f"  [TCDF] {n_rows} rows x {n} variables, {epochs} epochs/target, "
          f"device={'cuda' if cuda else 'cpu'}, tau_max={tau_max}")

    for target_idx in range(n):
        t_start = time.time()
        print(f"  [TCDF] Target {target_idx + 1}/{n}: '{columns[target_idx]}'")

        X_train, Y_train = _prepare_data(data_scaled, target_idx)
        if cuda:
            X_train, Y_train = X_train.cuda(), Y_train.cuda()

        model = _ADDSTCN(n, num_levels, kernel_size, dilation_c, cuda)
        if cuda:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        first_loss = last_loss = None
        for ep in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = F.mse_loss(output, Y_train)
            loss.backward()
            optimizer.step()
            if ep == 1:
                first_loss = loss.item()
            last_loss = loss.item()
            if ep == 1 or ep % log_interval == 0 or ep == epochs:
                print(f"    epoch [{ep}/{epochs}] loss={last_loss:.6f}")

        print(f"    training done in {time.time() - t_start:.1f}s "
              f"(loss {first_loss:.4f} -> {last_loss:.4f})")

        scores_np = model.fs_attention.data.view(-1).cpu().numpy()
        potentials = _select_potentials_by_gap(scores_np)
        print(f"    potential causes (pre-PIVM): {[columns[i] for i in potentials] or 'none'}")
        if not potentials:
            continue

        # PIVM: shuffle each candidate source's series; if prediction barely
        # degrades, it wasn't really being used as a cause — drop it.
        validated = list(potentials)
        diff = first_loss - last_loss
        for idx in potentials:
            random.seed(seed)
            X_test = X_train.clone().cpu().numpy()
            random.shuffle(X_test[:, idx, :][0])
            shuffled = torch.from_numpy(X_test)
            if cuda:
                shuffled = shuffled.cuda()
            model.eval()
            with torch.no_grad():
                testloss = F.mse_loss(model(shuffled), Y_train).cpu().item()
            testdiff = first_loss - testloss
            if testdiff > diff * significance:
                validated.remove(idx)

        print(f"    validated causes (post-PIVM): {[columns[i] for i in validated] or 'none'}")
        if not validated:
            continue

        weights = []
        for level in range(num_levels):
            w = model.dwn.network[level].net[0].weight.abs()
            weights.append(w.view(w.size(0), w.size(2)))

        attn_norm = F.softmax(torch.tensor(scores_np), dim=0).numpy()

        for src_idx in validated:
            if src_idx == target_idx:
                continue  # skip self-causation — app convention is no self-loops

            total_delay = 0
            for level, w in enumerate(weights):
                row = w[src_idx].tolist()
                m, m2 = heapq.nlargest(2, row)
                if m > m2:
                    index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
                else:
                    index_max = 0
                total_delay += index_max * (dilation_c ** level)

            tau = max(1, min(total_delay, tau_max))
            edges.append({
                'cause': columns[src_idx],
                'effect': columns[target_idx],
                'label': f'TCDF_LAG{tau}',
                'confidence': round(min(1.0, float(attn_norm[src_idx])), 3),
            })

    print(f"  [TCDF] {len(edges)} directed edge(s) found across all {n} target(s)")
    return edges


def run_dag_gnn(
    df: pd.DataFrame,
    columns: list,
    hidden_dims: int = 64,
    z_dims: int = 1,
    lr: float = 3e-3,
    tau_a: float = 0.0,
    trace_penalty: float = 100.0,
    graph_threshold: float = 0.3,
    inner_epochs: int = 300,
    k_max_iter: int = 100,
    c_init: float = 1.0,
    c_max: float = 1e20,
    h_tol: float = 1e-8,
    max_total_blocks: int = 40,
    seed: int = 42,
) -> list:
    """
    DAG-GNN (Yu et al., ICML 2019) — a GNN-based structure learner: a single
    learned d x d adjacency A routes every variable's data through a shared
    encoder/decoder via (I - A^T) / (I - A^T)^-1, trained with an augmented-
    Lagrangian schedule against a polynomial acyclicity constraint
    h(A) = tr[(I + A∘A/d)^d] - d. No time/lag awareness (like PC/LiNGAM) —
    treats rows as i.i.d. samples. GPU-accelerated (same
    torch.cuda.is_available() pattern as run_tcdf).

    Ported from cd_algorithm/DAG-GNN.ipynb, itself a faithful port of the
    reference implementation (github.com/fishmoon1234/DAG-GNN) — a first
    draft of that notebook only loosely resembled DAG-GNN; this port matches
    the reference's actual architecture:
      - sinh(3*raw_A) reparameterization of the adjacency (amplifies small
        values, accelerates convergence — the reference's own rationale)
      - the encoder has NO logvar/sampling step — its output ('logits') is
        used directly as z, deterministically. This is NOT a reparameterized
        Gaussian VAE despite the reference's own "VAE" framing; it's a
        deterministic SEM autoencoder. The "KL" term (kl_gaussian_sem) is an
        L2 pull toward zero, not a real posterior KL divergence.
      - reconstruction loss uses a hardcoded variance=0.0, collapsing it to
        plain scaled MSE
      - a shared per-node offset Wa, added before the graph operation and
        subtracted after, in both encoder and decoder
      - a 100*trace(A∘A) term — NOT a general sparsity regularizer (trace of
        an elementwise square only sums the diagonal), it penalizes self-
        loop magnitude specifically. This module additionally masks the
        adjacency's diagonal to exactly zero every forward pass (a stricter,
        harder guarantee than the reference's soft penalty, which is a
        disclosed deviation from it — the reference relies on the soft
        penalty alone)
      - tau_a defaults to 0.0, matching the reference's own default (its L1
        soft-threshold stau() is present but inactive at this value)
      - the augmented-Lagrangian schedule's "has h(A) shrunk enough" anchor
        (h_old) updates once per outer step, after the inner c_A-escalation
        loop exits — not on every escalation (an earlier version of this
        port had that bug, made the schedule meaningfully weaker)

    Known, load-bearing caveat: h(A) == 0 is a property of the CONTINUOUS
    weighted matrix, not a guarantee about the thresholded binary graph —
    thresholding to binarize is a heuristic afterthought with no formal
    acyclicity guarantee (a documented weakness of the whole NOTEARS/DAG-GNN
    continuous-relaxation family). On the synthetic benchmark in
    cd_algorithm/DAG-GNN.ipynb, the thresholded graph contains a directed
    cycle at every swept threshold despite h(A)=0, reproduced across 8
    seeds. This function reports (does not silently hide) whether its own
    thresholded graph is acyclic, and prints the cycle if not — it does NOT
    break cycles automatically (that is the planned, not-yet-built
    causal_assumptions.py DAG-enforcement layer's job, not this method's).

    Returns list of {cause, effect, label, confidence} with cause/effect as
    DATASET COLUMN names (same convention as run_pc/run_pcmci/run_lingam).
    """
    import math
    import networkx as nx
    import torch
    import torch.nn as nn

    data = df[columns].dropna()
    if data.shape[0] < 20:
        raise ValueError(f"Not enough complete rows ({data.shape[0]}) to run DAG-GNN reliably (need >= 20).")

    means = data.mean(axis=0)
    stds = data.std(axis=0, ddof=0)
    if (stds == 0).any():
        constant_columns = stds[stds == 0].index.tolist()
        raise ValueError(f"Constant column(s) found (cannot standardize): {constant_columns}")
    data_scaled = (data - means) / stds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = len(columns)
    print(f"  [DAG-GNN] {data.shape[0]} rows x {d} variables, device={device}")

    torch.manual_seed(seed)

    def _preprocess_adj_new(adj, eye):
        return eye - adj.t()

    def _preprocess_adj_new1(adj, eye):
        return torch.inverse(eye - adj.t())

    def _matrix_poly(matrix, dd, eye):
        x = eye + torch.div(matrix, dd)
        return torch.matrix_power(x, dd)

    def _h_acyclic(A, dd, eye):
        return torch.trace(_matrix_poly(A * A, dd, eye)) - dd

    def _nll_gaussian(preds, target, variance=0.0):
        neg_log_p = variance + torch.div((preds - target) ** 2, 2. * math.exp(2. * variance))
        return neg_log_p.sum() / target.shape[0]

    def _kl_gaussian_sem(preds):
        return (preds * preds).sum() / preds.shape[0] * 0.5

    _prox_plus = torch.nn.Threshold(0., 0.)

    def _stau(w, tau):
        w1 = _prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def _update_optimizer(optimizer, original_lr, c_a):
        max_lr, min_lr = 1e-2, 1e-4
        estimated_lr = original_lr / (math.log10(c_a) + 1e-10)
        new_lr = min(max(estimated_lr, min_lr), max_lr)
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        return optimizer, new_lr

    class _GraphEncoder(nn.Module):
        def __init__(self, x_dims, hidden, z_dim):
            super().__init__()
            self.fc1 = nn.Linear(x_dims, hidden)
            self.fc2 = nn.Linear(hidden, z_dim)
            self.Wa = nn.Parameter(torch.zeros(z_dim))

        def forward(self, x, adj_aforz):
            h1 = torch.relu(self.fc1(x))
            z_feat = self.fc2(h1)
            return torch.matmul(adj_aforz, z_feat + self.Wa) - self.Wa

    class _GraphDecoder(nn.Module):
        def __init__(self, z_dim, hidden, x_dims):
            super().__init__()
            self.fc1 = nn.Linear(z_dim, hidden)
            self.fc2 = nn.Linear(hidden, x_dims)

        def forward(self, z, adj_ainv, wa):
            mat_z = torch.matmul(adj_ainv, z + wa) - wa
            h3 = torch.relu(self.fc1(mat_z))
            return self.fc2(h3)

    X = torch.tensor(data_scaled.values.astype(np.float32), device=device).unsqueeze(-1)
    eye = torch.eye(d, device=device)
    offdiag_mask = 1.0 - torch.eye(d, device=device)

    raw_adj = nn.Parameter(torch.zeros(d, d, device=device))
    encoder = _GraphEncoder(1, hidden_dims, z_dims).to(device)
    decoder = _GraphDecoder(z_dims, hidden_dims, 1).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [raw_adj], lr=lr
    )

    def _forward_pass():
        origin_A = torch.sinh(3.0 * raw_adj) * offdiag_mask
        logits = encoder(X, _preprocess_adj_new(origin_A, eye))
        return logits, origin_A

    def _train_step(lambda_a, c_a):
        optimizer.zero_grad()
        logits, origin_A = _forward_pass()
        preds = decoder(logits, _preprocess_adj_new1(origin_A, eye), encoder.Wa)

        loss_nll = _nll_gaussian(preds, X, variance=0.0)
        loss_kl = _kl_gaussian_sem(logits)
        h = _h_acyclic(origin_A, d, eye)
        sparse_loss = tau_a * torch.sum(torch.abs(origin_A))

        loss = (loss_kl + loss_nll + lambda_a * h + 0.5 * c_a * h * h
                + trace_penalty * torch.trace(origin_A * origin_A) + sparse_loss)
        loss.backward()
        optimizer.step()

    c_a, lambda_a, h_old, h_new = c_init, 0.0, np.inf, 1.0
    total_blocks = 0
    origin_A = None

    for _step_k in range(k_max_iter):
        while c_a < c_max:
            optimizer, cur_lr = _update_optimizer(optimizer, lr, c_a)
            for _ in range(inner_epochs):
                _train_step(lambda_a, c_a)
                raw_adj.data = _stau(raw_adj.data, tau_a * cur_lr)
            total_blocks += 1

            with torch.no_grad():
                origin_A = torch.sinh(3.0 * raw_adj) * offdiag_mask
                h_new = _h_acyclic(origin_A, d, eye).item()

            if total_blocks >= max_total_blocks:
                break
            if h_new > 0.25 * h_old:
                c_a *= 10
            else:
                break

        h_old = h_new
        lambda_a += c_a * h_new
        if h_new <= h_tol or total_blocks >= max_total_blocks:
            break

    print(f"  [DAG-GNN] Lagrangian schedule stopped after {total_blocks} training "
          f"block(s) ({total_blocks * inner_epochs} epochs), c_a={c_a:.2e}, "
          f"final h(A)={h_new:.2e} "
          f"({'acyclic (continuous)' if h_new <= h_tol else 'NOT fully acyclic'})")

    A = origin_A.detach().cpu().numpy()

    edges = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if abs(A[i, j]) > graph_threshold:
                edges.append({
                    'cause': columns[i],
                    'effect': columns[j],
                    'label': 'DAGGNN_CAUSES',
                    'confidence': round(min(1.0, abs(float(A[i, j]))), 3),
                })

    pred_graph = nx.DiGraph()
    pred_graph.add_nodes_from(columns)
    for e in edges:
        pred_graph.add_edge(e['cause'], e['effect'])
    is_dag = nx.is_directed_acyclic_graph(pred_graph)
    cycle_note = ""
    if edges and not is_dag:
        try:
            cycle_note = f" — e.g. cycle: {nx.find_cycle(pred_graph)}"
        except nx.NetworkXNoCycle:
            pass

    print(f"  [DAG-GNN] {len(edges)} directed edge(s) kept at threshold={graph_threshold} "
          f"— thresholded graph is_dag={is_dag}{cycle_note}")
    if not is_dag:
        print("  [DAG-GNN] NOTE: h(A)=0 describes the continuous weighted matrix, "
              "not this thresholded graph — cycles here are not auto-broken "
              "(see causal_assumptions.py in the project roadmap for planned "
              "DAG enforcement).")

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

    Ported from cd_algorithm/LPCMCI.ipynb. tau_min=0 (LPCMCI's own actual
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
    import tigramite.data_processing as pp
    from tigramite.lpcmci import LPCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from sklearn.preprocessing import StandardScaler

    data = df[columns].dropna()
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

    return edges


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

    Known limitation (not a bug in this port): benchmarked against the
    synthetic and Sachs benchmarks in cd_algorithm/GES.ipynb, GES found
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
