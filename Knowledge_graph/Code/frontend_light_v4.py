import streamlit as st
from keywords_extraction import process
from neo4j_storage import Neo4jConnector
from qa_module import qa_system
from frontend_dataset_display import (
    display_gpt4_toggle,
    display_dataset_filter,
    display_datasets_section,
    display_cost_summary,
    export_datasets_to_csv
)
from causal_graph import extract_causal_relations, generate_causal_graph
import os
import sys
import re
import base64
from io import BytesIO
import io
import json
import pandas as pd
import tempfile
import requests
import zipfile

# ── Dataset Harvester imports ──────────────────────────────────────────────
_harvester_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset_harvester"))
if _harvester_path not in sys.path:
    sys.path.insert(0, _harvester_path)
try:
    from extractor import extract_from_pdfs
    from deduplicator import deduplicate
    from resolver import resolve_one, ResolvedDataset
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    _HARVESTER_AVAILABLE = True
except Exception:
    _HARVESTER_AVAILABLE = False

try:
    from downloaders.pangaea import download as _pangaea_dl
    from downloaders.zenodo import download as _zenodo_dl
    _DOWNLOADERS_AVAILABLE = True
except Exception:
    _pangaea_dl = None
    _zenodo_dl = None
    _DOWNLOADERS_AVAILABLE = False


_DOWNLOADS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "downloads"))

# ── Downloadability checker ────────────────────────────────────────────────
_AUTH_REQUIRED = {
    "nsidc", "nasa_earthdata", "copernicus_cds", "copernicus_marine",
    "ecmwf", "esa", "jma", "ncar_rda",
}
_LANDING_PAGE_ONLY = {
    "uw_apl", "met_norway", "noaa", "noaa_ncei", "noaa_psl",
    "usgs", "argo", "gebco", "other",
}
_CSV_EXTS = {".csv", ".txt", ".tsv", ".tab"}

def _pangaea_id(r):
    for s in [r.resolved_url or "", r.doi or "", r.accession or ""]:
        m = re.search(r'PANGAEA\.(\d+)', s, re.IGNORECASE)
        if m:
            return m.group(1)
    return None

def _zenodo_id(r):
    for s in [r.resolved_url or "", r.doi or ""]:
        m = re.search(r'zenodo\.org/(?:records?/)?(\d+)|zenodo\.(\d+)', s, re.IGNORECASE)
        if m:
            return m.group(1) or m.group(2)
    return None

def check_downloadable(r):
    """Returns (status_key, label, detail)."""
    if not r.resolved_url:
        return "none", "No URL", "Dataset not resolved"
    repo = (r.repository or "").lower()
    if repo in _AUTH_REQUIRED:
        return "auth", "Login required", f"Needs account on {repo}"
    if repo in _LANDING_PAGE_ONLY:
        return "page", "Homepage only", "No direct file — manual download"
    if repo == "pangaea" or "pangaea" in (r.resolved_url or "").lower():
        pid = _pangaea_id(r)
        if pid:
            try:
                resp = requests.head(
                    f"https://doi.pangaea.de/10.1594/PANGAEA.{pid}?format=textfile",
                    timeout=5, allow_redirects=True
                )
                if resp.status_code == 200:
                    return "yes", "Downloadable", "PANGAEA tab-separated file"
                if resp.status_code == 503:
                    return "yes", "Downloadable", "PANGAEA file (server busy — download will retry)"
                return "collection", "Collection", "Parent record — contains child datasets"
            except Exception:
                return "unknown", "Unknown", "Could not reach PANGAEA"
    if repo == "zenodo" or "zenodo" in (r.resolved_url or "").lower():
        zid = _zenodo_id(r)
        if zid:
            try:
                resp = requests.get(f"https://zenodo.org/api/records/{zid}", timeout=5)
                if resp.status_code == 200:
                    files = resp.json().get("files", [])
                    csv_files = [f for f in files if os.path.splitext(f.get("key","").lower())[1] in _CSV_EXTS]
                    if csv_files:
                        return "yes", "Downloadable", f"Zenodo: {len(csv_files)} CSV/text file(s)"
                    return "no_csv", "No CSV files", "Zenodo record has no CSV/text files"
            except Exception:
                return "unknown", "Unknown", "Could not reach Zenodo"
    return "page", "Homepage only", "Landing page — no direct file"

def _do_download(row: dict) -> list[str]:
    """Download a primary/active dataset to _DOWNLOADS_ROOT. Returns list of local file paths."""
    url = row.get("URL", "")
    doi = row.get("_doi") or ""
    accession = row.get("_accession") or ""
    repo = (row.get("Repository") or "").lower()

    os.makedirs(_DOWNLOADS_ROOT, exist_ok=True)

    if (repo == "pangaea" or "pangaea" in url.lower() or "pangaea" in doi.lower()
            or "pangaea" in accession.lower()) and _pangaea_dl:
        return _pangaea_dl(url=url or None, doi=doi or None,
                           accession=accession or None, dest_root=_DOWNLOADS_ROOT)
    if (repo == "zenodo" or "zenodo" in url.lower() or "zenodo" in doi.lower()) and _zenodo_dl:
        return _zenodo_dl(url=url or None, doi=doi or None, dest_root=_DOWNLOADS_ROOT)
    raise ValueError(f"No downloader available for repository '{repo}' — URL: {url}")


@st.cache_data(show_spinner=False)
def _fetch_collection_children(pid: str) -> list[dict]:
    """List child datasets inside a PANGAEA collection.

    Scrapes the collection HTML page for child PANGAEA IDs, then tries
    the ZIP central directory as a fallback for filenames.
    """
    children = []

    # Attempt 1 — scrape the PANGAEA collection page
    try:
        resp = requests.get(
            f"https://doi.pangaea.de/10.1594/PANGAEA.{pid}",
            timeout=20,
            headers={"Accept": "text/html,*/*"},
            allow_redirects=True,
        )
        if resp.status_code == 200:
            child_ids = sorted(set(re.findall(r'PANGAEA\.(\d+)', resp.text)) - {pid})
            if child_ids:
                children = [
                    {
                        "name": f"PANGAEA.{cid}",
                        "id": cid,
                        "url": f"https://doi.pangaea.de/10.1594/PANGAEA.{cid}",
                        "downloadable": False,
                    }
                    for cid in child_ids
                ]
    except Exception:
        pass

    # Attempt 2 — download collection ZIP and read filenames from central directory
    if not children:
        try:
            zip_url = f"https://doi.pangaea.de/10.1594/PANGAEA.{pid}?format=zip"
            resp = requests.get(zip_url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                return [{"name": n, "id": "", "url": "", "downloadable": False} for n in zf.namelist()]
        except Exception:
            return []

    # Check downloadability for each child in parallel
    def _head_check(cid: str) -> bool:
        try:
            r = requests.head(
                f"https://doi.pangaea.de/10.1594/PANGAEA.{cid}?format=textfile",
                timeout=6, allow_redirects=True,
            )
            return r.status_code in (200, 503)
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=10) as _pool:
        _futs = {_pool.submit(_head_check, c["id"]): i for i, c in enumerate(children)}
        for _fut in _as_completed(_futs):
            children[_futs[_fut]]["downloadable"] = _fut.result()

    return children


def _read_pangaea_lines(fpath: str) -> list[str]:
    """Read a PANGAEA textfile, handling gzip or zip compression transparently."""
    import gzip as _gzip
    with open(fpath, "rb") as _fb:
        magic = _fb.read(4)
    if magic[:2] == b"PK":
        # ZIP — read the first text file inside
        with zipfile.ZipFile(fpath) as _zf:
            for _name in _zf.namelist():
                if os.path.splitext(_name)[1].lower() in (".txt", ".tsv", ".tab", ".csv"):
                    with _zf.open(_name) as _inner:
                        return _inner.read().decode("utf-8", errors="replace").splitlines(keepends=True)
        return []
    if magic[:2] == b"\x1f\x8b":
        with _gzip.open(fpath, "rt", encoding="utf-8", errors="replace") as _f:
            return _f.readlines()
    with open(fpath, encoding="utf-8", errors="replace") as _f:
        return _f.readlines()


def _preview_file(fpath: str, n: int = 10) -> "pd.DataFrame | None":
    """Return the first n rows of a downloaded dataset file as a DataFrame."""
    ext = os.path.splitext(fpath)[1].lower()
    try:
        if ext in (".txt", ".tsv", ".tab"):
            lines = _read_pangaea_lines(fpath)
            # Skip the /* ... */ metadata header block
            start = 0
            in_header = False
            for idx, line in enumerate(lines):
                s = line.strip()
                if s.startswith("/*"):
                    in_header = True
                if in_header and s.endswith("*/"):
                    start = idx + 1
                    break
            data_text = "".join(lines[start:])
            return pd.read_csv(
                io.StringIO(data_text), sep="\t", nrows=n,
                on_bad_lines="skip",
            )
        if ext == ".csv":
            return pd.read_csv(
                fpath, nrows=n,
                on_bad_lines="skip", encoding="utf-8", encoding_errors="replace",
            )
    except Exception:
        return None
    return None


# Page config
st.set_page_config(
    page_title="PolarKD — Polar Knowledge Discovery Toolkit",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── EDITORIAL CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Reset & Base ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    html, body, .stApp {
        background-color: #F5F8FC !important;
        color: #0D2347 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .main .block-container {
        background-color: transparent !important;
        padding: 0 2rem 4rem 2rem !important;
        max-width: 1280px !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Typography overrides ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #0D2347 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    p, span, div, label, li {
        color: #1E3A6E !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Ensure button inner elements are never hijacked by the rule above */
    .stButton > button, .stButton > button *,
    .stDownloadButton > button, .stDownloadButton > button *,
    .stFormSubmitButton > button, .stFormSubmitButton > button *,
    [data-testid="baseButton-primary"], [data-testid="baseButton-primary"] *,
    [data-testid="baseButton-secondary"], [data-testid="baseButton-secondary"] * {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Hero Banner ── */
    .polar-hero {
        background: linear-gradient(135deg, #081A3A 0%, #0D2347 45%, #1B3A7A 100%);
        padding: 4rem 4rem 3rem 4rem;
        margin: -1rem -2rem 0 -2rem;
        position: relative;
        overflow: hidden;
    }

    .polar-hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 320px; height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(74,159,212,0.15) 0%, transparent 70%);
        pointer-events: none;
    }

    .polar-hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 10%;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(74,159,212,0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #4A9FD4 !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .hero-eyebrow::before, .hero-eyebrow::after {
        content: '';
        display: inline-block;
        width: 36px; height: 1px;
        background: #4A9FD4;
        opacity: 0.6;
    }

    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: clamp(2.6rem, 5vw, 4rem) !important;
        font-weight: 700 !important;
        color: #F5F8FC !important;
        line-height: 1.08 !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.5rem;
    }

    .hero-title em {
        font-style: italic;
        color: #4A9FD4 !important;
    }

    .hero-subtitle {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.25rem !important;
        color: rgba(245,248,252,0.65) !important;
        font-weight: 300 !important;
        margin-top: 1.25rem !important;
        max-width: 580px;
        line-height: 1.6;
    }

    .hero-meta {
        margin-top: 2.5rem;
        display: flex;
        gap: 2.5rem;
        align-items: center;
        flex-wrap: wrap;
    }

    .hero-stat {
        text-align: left;
    }

    .hero-stat-num {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #4A9FD4 !important;
        font-weight: 600;
        line-height: 1;
    }

    .hero-stat-label {
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: rgba(245,248,252,0.45) !important;
        margin-top: 0.2rem;
    }

    .hero-divider {
        width: 1px; height: 40px;
        background: rgba(74,159,212,0.25);
    }

    /* ── Navigation Pills ── */
    .polar-nav {
        display: flex;
        gap: 0.5rem;
        padding: 1.25rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(13,35,71,0.1);
        margin-bottom: 2rem;
    }

    .polar-nav-pill {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.55rem 1.25rem;
        border-radius: 2px;
        color: #1E3A6E !important;
        background: transparent;
        border: 1px solid rgba(13,35,71,0.15);
        cursor: pointer;
        transition: all 0.25s ease;
    }

    .polar-nav-pill:hover, .polar-nav-pill.active {
        background: #0D2347;
        color: #F5F8FC !important;
        border-color: #0D2347;
    }

    /* ── Section Labels ── */
    .section-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #4A9FD4 !important;
        margin-bottom: 0.6rem;
        display: block;
    }

    .section-heading {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.9rem !important;
        font-weight: 600 !important;
        color: #0D2347 !important;
        letter-spacing: -0.025em !important;
        margin-bottom: 1.75rem !important;
        line-height: 1.15 !important;
    }

    .section-heading em {
        font-style: italic;
        color: #2E5FA0 !important;
    }

    /* ── Upload Zone ── */
    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;
        border: 1.5px dashed rgba(46,95,160,0.3) !important;
        border-radius: 4px !important;
        padding: 3rem 2rem !important;
        text-align: center !important;
        min-height: 180px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4A9FD4 !important;
        background: #EEF4FB !important;
        box-shadow: 0 0 0 4px rgba(74,159,212,0.08) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #2E5FA0 !important;
        font-size: 0.9rem !important;
        font-weight: 400 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "↑";
        display: block;
        font-size: 2rem;
        margin-bottom: 0.75rem;
        color: #4A9FD4 !important;
        font-weight: 300;
    }

    /* ── Buttons ── */
    .stButton > button,
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"] {
        background: #0D2347 !important;
        color: #F5F8FC !important;
        border: 1px solid #0D2347 !important;
        padding: 0.7rem 1.75rem !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        transition: all 0.25s ease !important;
        box-shadow: none !important;
        white-space: nowrap !important;
    }

    /* Force ALL child text inside buttons to stay cream — overrides global p/span/div rules */
    .stButton > button *,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div,
    [data-testid="baseButton-primary"] *,
    [data-testid="baseButton-secondary"] * {
        color: #F5F8FC !important;
    }

    .stButton > button:hover,
    .stButton > button:hover *,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="baseButton-primary"]:hover *,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover * {
        background: #4A9FD4 !important;
        border-color: #4A9FD4 !important;
        color: #0D2347 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(74,159,212,0.3) !important;
    }

    .stDownloadButton > button {
        background: transparent !important;
        color: #0D2347 !important;
        border: 1px solid rgba(13,35,71,0.4) !important;
        padding: 0.7rem 1.75rem !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        transition: all 0.25s ease !important;
    }

    .stDownloadButton > button *,
    .stDownloadButton > button p,
    .stDownloadButton > button span {
        color: #0D2347 !important;
    }

    .stDownloadButton > button:hover,
    .stDownloadButton > button:hover * {
        background: #0D2347 !important;
        color: #F5F8FC !important;
        border-color: #0D2347 !important;
    }

    .stFormSubmitButton > button {
        background: #4A9FD4 !important;
        color: #0D2347 !important;
        border: 1px solid #4A9FD4 !important;
        padding: 0.7rem 2rem !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        transition: all 0.25s ease !important;
        white-space: nowrap !important;
        min-width: 90px !important;
    }

    .stFormSubmitButton > button *,
    .stFormSubmitButton > button p,
    .stFormSubmitButton > button span {
        color: #0D2347 !important;
    }

    .stFormSubmitButton > button:hover,
    .stFormSubmitButton > button:hover * {
        background: #0D2347 !important;
        color: #F5F8FC !important;
        border-color: #0D2347 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Cards ── */
    .polar-card {
        background: #FFFFFF;
        border: 1px solid rgba(13,35,71,0.08);
        border-radius: 4px;
        padding: 1.75rem;
        margin-bottom: 1rem;
    }

    .polar-card-dark {
        background: #0D2347;
        border: none;
        border-radius: 4px;
        padding: 1.75rem;
        margin-bottom: 1rem;
    }

    .polar-card-dark p, .polar-card-dark span, .polar-card-dark div, .polar-card-dark label {
        color: rgba(245,248,252,0.75) !important;
    }

    .polar-card-dark h3 {
        color: #F5F8FC !important;
    }

    .polar-info-row {
        background: #FFFFFF;
        border-left: 3px solid #4A9FD4;
        padding: 1rem 1.25rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        color: #1E3A6E !important;
    }

    /* ── Alert / Info Boxes ── */
    .stAlert, [data-testid="stAlert"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(74,159,212,0.4) !important;
        border-left: 3px solid #4A9FD4 !important;
        border-radius: 4px !important;
        color: #1E3A6E !important;
    }

    .stSuccess {
        background: #F2FBF4 !important;
        border-left-color: #4A9B6F !important;
    }

    .stWarning {
        background: #FFFBF0 !important;
        border-left-color: #D4A017 !important;
    }

    .stError {
        background: #FFF5F5 !important;
        border-left-color: #C0392B !important;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(13,35,71,0.1) !important;
        margin: 2.5rem 0 !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(13,35,71,0.07) !important;
        border-radius: 4px !important;
        padding: 1.25rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: #5F7A9D !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
        color: #0D2347 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #4A9B6F !important;
    }

    /* ── Keyword Tags ── */
    .kw-tag {
        display: inline-block;
        background: #0D2347;
        color: #F5F8FC !important;
        padding: 0.3rem 0.85rem;
        border-radius: 2px;
        margin: 0.2rem;
        font-size: 0.72rem;
        font-weight: 400;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }

    .kw-tag-light {
        display: inline-block;
        background: transparent;
        color: #1E3A6E !important;
        border: 1px solid rgba(13,35,71,0.25);
        padding: 0.3rem 0.85rem;
        border-radius: 2px;
        margin: 0.2rem;
        font-size: 0.72rem;
        font-weight: 400;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Causal Tags ── */
    .causal-cause-tag {
        display: inline-block;
        background: #C0392B;
        color: #F5F8FC !important;
        padding: 0.3rem 0.85rem;
        border-radius: 2px;
        margin: 0.2rem;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }

    .causal-effect-tag {
        display: inline-block;
        background: #E67E22;
        color: #F5F8FC !important;
        padding: 0.3rem 0.85rem;
        border-radius: 2px;
        margin: 0.2rem;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Chat ── */
    .chat-bubble-user {
        background: #0D2347;
        color: #F5F8FC !important;
        padding: 1rem 1.25rem;
        border-radius: 4px 4px 4px 0;
        margin: 0.75rem 0;
        font-size: 0.875rem;
        line-height: 1.6;
    }

    .chat-bubble-user strong, .chat-bubble-user span, .chat-bubble-user div {
        color: #F5F8FC !important;
    }

    .chat-bubble-assistant {
        background: #FFFFFF;
        border: 1px solid rgba(13,35,71,0.09);
        padding: 1rem 1.25rem;
        border-radius: 4px 4px 0 4px;
        margin: 0.75rem 0;
        font-size: 0.875rem;
        line-height: 1.6;
        border-left: 3px solid #4A9FD4;
    }

    .chat-label {
        font-size: 0.62rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-weight: 500;
        margin-bottom: 0.3rem;
        display: block;
    }

    .chat-label-user { color: #4A9FD4 !important; }
    .chat-label-ai { color: #5F7A9D !important; }

    /* ── Text Input ── */
    .stTextInput > div > div > input {
        background: #FFFFFF !important;
        color: #0D2347 !important;
        border: 1px solid rgba(13,35,71,0.2) !important;
        border-radius: 2px !important;
        padding: 0.65rem 1rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
        transition: border-color 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4A9FD4 !important;
        box-shadow: 0 0 0 3px rgba(74,159,212,0.12) !important;
        outline: none !important;
    }

    /* ── Selectbox — trigger box ── */
    .stSelectbox > div > div,
    [data-testid="stSelectbox"] > div > div {
        background: #FFFFFF !important;
        border: 1px solid rgba(13,35,71,0.2) !important;
        border-radius: 2px !important;
    }

    /* Text shown inside the selectbox trigger */
    .stSelectbox [data-baseweb="select"] > div,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] input {
        background: #FFFFFF !important;
        color: #0D2347 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
    }

    /* Dropdown list container (the dark popover) */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"],
    [role="listbox"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(13,35,71,0.12) !important;
        border-radius: 4px !important;
        box-shadow: 0 8px 24px rgba(13,35,71,0.12) !important;
    }

    /* Each dropdown option */
    [role="option"],
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] ul li {
        background: #FFFFFF !important;
        color: #0D2347 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
    }

    /* Hovered / focused option */
    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    [data-baseweb="menu"] li:hover {
        background: #E3EDF8 !important;
        color: #0D2347 !important;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] {
        padding-top: 0.25rem;
    }

    .stSlider [role="slider"] {
        background: #0D2347 !important;
    }

    .stSlider [data-testid="stThumbValue"] {
        color: #0D2347 !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div {
        background: #4A9FD4 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #FFFFFF !important;
        border: 1px solid rgba(13,35,71,0.1) !important;
        border-radius: 4px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        color: #0D2347 !important;
    }

    /* ── Database Document Item ── */
    .doc-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: #FFFFFF;
        border: 1px solid rgba(13,35,71,0.08);
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.83rem;
        color: #0D2347 !important;
    }

    .doc-item::before {
        content: '↗';
        color: #4A9FD4;
        font-size: 1rem;
        flex-shrink: 0;
    }

    /* ── Graph Legend ── */
    .graph-legend {
        display: flex;
        gap: 2rem;
        padding: 1rem 1.5rem;
        background: #FFFFFF;
        border: 1px solid rgba(13,35,71,0.08);
        border-radius: 4px;
        margin-bottom: 1.25rem;
        align-items: center;
    }

    .graph-legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #1E3A6E !important;
        font-weight: 500;
    }

    .graph-legend-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Empty State ── */
    .empty-state {
        background: #FFFFFF;
        border: 1px dashed rgba(13,35,71,0.15);
        border-radius: 4px;
        padding: 4rem 2rem;
        text-align: center;
    }

    .empty-state-glyph {
        font-size: 2rem;
        color: #4A9FD4 !important;
        margin-bottom: 1rem;
        display: block;
        font-weight: 300;
    }

    .empty-state-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #1E3A6E !important;
        margin-bottom: 0.5rem;
    }

    .empty-state-text {
        font-size: 0.82rem;
        color: #5F7A9D !important;
        max-width: 320px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Footer ── */
    .polar-footer {
        margin-top: 4rem;
        padding: 2.5rem 0;
        border-top: 1px solid rgba(13,35,71,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .polar-footer-brand {
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        color: #0D2347 !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .polar-footer-links {
        display: flex;
        gap: 1.5rem;
    }

    .polar-footer-links a {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #5F7A9D !important;
        text-decoration: none;
        transition: color 0.2s;
    }

    .polar-footer-links a:hover { color: #0D2347 !important; }

    .polar-footer-copy {
        font-size: 0.72rem;
        color: #5F7A9D !important;
        letter-spacing: 0.05em;
    }

    /* ── Column padding ── */
    [data-testid="column"] { padding: 0 0.75rem !important; }

    /* ── Checkbox ── */
    .stCheckbox span {
        font-size: 0.83rem !important;
        color: #0D2347 !important;
    }

    /* ── Tooltip ── */
    .stTooltipIcon { color: #4A9FD4 !important; }

    /* ── Subtle horizontal rule inside sections ── */
    .inner-rule {
        border: none;
        border-top: 1px solid rgba(13,35,71,0.08);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOGO LOADING (GitHub raw URLs) ────────────────────────────────────────
IHARP_LOGO_URL = "https://raw.githubusercontent.com/d3lab-unt/polarKD/main/Knowledge_graph/images/iharp%20logo.png"
UNT_LOGO_URL   = "https://raw.githubusercontent.com/d3lab-unt/polarKD/main/Knowledge_graph/images/university-of-north-texas-seeklogo.png"


# ─── SESSION STATE ─────────────────────────────────────────────────────────
for key, default in [
    ('uploaded_files', []),
    ('databases', []),
    ('chat_history', []),
    ('processed_pdfs', {}),
    ('current_graph', None),
    ('show_qa_dialog', False),
    ('show_kg_dialog', False),
    ('show_cg_dialog', False),
    ('cg_relations', []),
    ('harvester_cache_key', None),
    ('harvester_results', None),
    ('harvester_elapsed', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── HERO ──────────────────────────────────────────────────────────────────
iharp_logo_html = f'<img src="{IHARP_LOGO_URL}" style="height:48px;width:auto;margin-bottom:2rem;opacity:0.9;" alt="iHARP Logo">'

unt_logo_html = (
    '<div style="position:absolute;top:1.5rem;right:2rem;'
    'background:white;border-radius:50%;padding:6px;'
    'box-shadow:0 2px 12px rgba(0,0,0,0.2);'
    'display:flex;align-items:center;justify-content:center;">'
    f'<img src="{UNT_LOGO_URL}" '
    'style="height:80px;width:80px;object-fit:contain;border-radius:50%;" alt="UNT Logo">'
    '</div>'
)

hero_html = (
    '<div class="polar-hero">'
    + unt_logo_html
    + iharp_logo_html
    + '<div class="hero-eyebrow">iHARP Research Initiative</div>'
    + '<div class="hero-title">Polar <em>Knowledge</em><br>Discovery Toolkit</div>'
    + '<div class="hero-subtitle">Extract climate variables, build semantic knowledge graphs, discover causal relationships, and interrogate polar science literature — all within a single intelligent workspace.</div>'
    + '<div class="hero-meta">'
    + '<a href="#section-upload" style="text-decoration:none;">'
    +   '<div class="hero-stat" style="cursor:pointer;" onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
    +     '<div class="hero-stat-num">PDF</div>'
    +     '<div class="hero-stat-label">Ingestion</div>'
    +   '</div>'
    + '</a>'
    + '<div class="hero-divider"></div>'
    + '<a href="#section-qa" style="text-decoration:none;">'
    +   '<div class="hero-stat" style="cursor:pointer;" onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
    +     '<div class="hero-stat-num">Q&amp;A</div>'
    +     '<div class="hero-stat-label">Document QA</div>'
    +   '</div>'
    + '</a>'
    + '<div class="hero-divider"></div>'
    + '<a href="#section-kg" style="text-decoration:none;">'
    +   '<div class="hero-stat" style="cursor:pointer;" onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
    +     '<div class="hero-stat-num">KG</div>'
    +     '<div class="hero-stat-label">Knowledge Graph</div>'
    +   '</div>'
    + '</a>'
    + '<div class="hero-divider"></div>'
    + '<a href="#section-cg" style="text-decoration:none;">'
    +   '<div class="hero-stat" style="cursor:pointer;" onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
    +     '<div class="hero-stat-num">CG</div>'
    +     '<div class="hero-stat-label">Causal Graph</div>'
    +   '</div>'
    + '</a>'
    + '</div>'
    + '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div id="section-upload"></div>', unsafe_allow_html=True)
st.markdown('<span class="section-label">Step 01</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Upload <em>Documents</em></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Drag & Drop PDFs here or Click to Upload",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="visible",
        help="Select multiple PDF files"
    )

    if uploaded_files is not None and len(uploaded_files) > 0:
        st.success(f"✓ {len(uploaded_files)} file(s) ready")
        for i, file in enumerate(uploaded_files, 1):
            st.markdown(f'<div class="doc-item">{i}. {file.name} &nbsp;·&nbsp; {file.size // 1024} KB</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin-top:0.5rem;padding:0.75rem 1rem;background:#FFFFFF;border-radius:4px;border:1px solid rgba(13,35,71,0.08);font-size:0.8rem;color:#5F7A9D;">No files selected yet — drag PDFs above or click to browse.</div>', unsafe_allow_html=True)
with col1:
    st.markdown('<span class="section-label">Configuration</span>', unsafe_allow_html=True)
    k = st.slider("Keywords to Extract (Knowledge Graph)", min_value=5, max_value=50, value=15, step=5)
    use_gpt4_datasets = display_gpt4_toggle()
    filter_variables = st.checkbox(
        "Filter to Climate Variables Only",
        value=True,
        help="Retain only measurable variables (temperature, salinity, pressure…). Removes organisations, locations, methods."
    )

with col2:
    # ── Dialog states
    if 'show_qa_dialog' not in st.session_state:
        st.session_state.show_qa_dialog = False
    if 'show_kg_dialog' not in st.session_state:
        st.session_state.show_kg_dialog = False
    if 'show_cg_dialog' not in st.session_state:
        st.session_state.show_cg_dialog = False

    # ── Q&A Button
    if st.button("📚 Send to Q&A", use_container_width=True, key="send_qa"):
        if uploaded_files and len(uploaded_files) > 0:
            st.session_state.show_qa_dialog = True
            st.session_state.show_kg_dialog = False
            st.session_state.show_cg_dialog = False
        else:
            st.warning("Please upload files first.")

    # ── Q&A inline card
    if st.session_state.show_qa_dialog:
        st.markdown("""
        <div style="
            background:#F0F6FF;
            border:1.5px solid #4A9FD4;
            border-left:4px solid #4A9FD4;
            border-radius:6px;
            padding:1.25rem 1.5rem 0.75rem 1.5rem;
            margin-top:0.5rem;
            margin-bottom:0.25rem;
        ">
            <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D2347;margin-bottom:0.2rem;">
                📚 Send to Q&amp;A
            </div>
            <div style="font-size:0.76rem;color:#5F7A9D;letter-spacing:0.04em;">
                Select the LLM model to use for answering questions
            </div>
        </div>
        """, unsafe_allow_html=True)

        qa_model = st.selectbox(
            "LLM Model",
            options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
            index=0,
            key="qa_model_dialog",
            help="Ollama model for answering questions."
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ Confirm", use_container_width=True, key="qa_confirm"):
                st.session_state.show_qa_dialog = False
                qa_system.set_model(qa_model)
                st.info(f"Model: **{qa_model}**")
                with st.spinner("Indexing documents…"):
                    added_count = 0
                    for file in uploaded_files:
                        if file.name not in st.session_state.databases:
                            file.seek(0)
                            temp_path = f"temp_qa_{file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(file.read())
                            if qa_system.add_document(file.name, pdf_path=temp_path):
                                st.session_state.databases.append(file.name)
                                added_count += 1
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    if added_count > 0:
                        st.success(f"✓ {added_count} file(s) indexed.")
                    else:
                        st.info("Files already indexed.")
        with col_cancel:
            if st.button("✕ Cancel", use_container_width=True, key="qa_cancel"):
                st.session_state.show_qa_dialog = False
                st.rerun()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── KG Button
    if st.button("◎ Generate Knowledge Graph", use_container_width=True, key="gen_kg"):
        if uploaded_files and len(uploaded_files) > 0:
            st.session_state.show_kg_dialog = True
            st.session_state.show_qa_dialog = False
            st.session_state.show_cg_dialog = False
        else:
            st.warning("Please upload files first.")

    # ── KG inline card
    if st.session_state.show_kg_dialog:
        st.markdown("""
        <div style="
            background:#F0F6FF;
            border:1.5px solid #4A9FD4;
            border-left:4px solid #4A9FD4;
            border-radius:6px;
            padding:1.25rem 1.5rem 0.75rem 1.5rem;
            margin-top:0.5rem;
            margin-bottom:0.25rem;
        ">
            <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D2347;margin-bottom:0.2rem;">
                ◎ Knowledge Graph Configuration
            </div>
            <div style="font-size:0.76rem;color:#5F7A9D;letter-spacing:0.04em;">
                Choose model and graph type before generating
            </div>
        </div>
        """, unsafe_allow_html=True)

        kg_model = st.selectbox(
            "LLM Model for Relation Extraction",
            options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
            index=0,
            key="kg_model_dialog",
        )
        kg_graph_type = st.selectbox(
            "Graph Visualization Type",
            options=["Full Graph (with Datasets)", "Knowledge Graph Only (without Datasets)"],
            index=0,
            key="kg_graph_type_dialog",
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ Confirm", use_container_width=True, key="kg_confirm"):
                st.session_state.show_kg_dialog = False
                st.session_state.kg_model_selected = kg_model
                st.session_state.kg_graph_type_selected = kg_graph_type
                st.rerun()
        with col_cancel:
            if st.button("✕ Cancel", use_container_width=True, key="kg_cancel"):
                st.session_state.show_kg_dialog = False
                st.rerun()

#     k = st.slider("Keywords to Extract (Knowledge Graph)", min_value=5, max_value=50, value=15, step=5)

#     use_gpt4_datasets = display_gpt4_toggle()

#     filter_variables = st.checkbox(
#         "Filter to Climate Variables Only",
#         value=True,
#         help="Retain only measurable variables (temperature, salinity, pressure…). Removes organisations, locations, methods."
#     )

#     # ── Dialog states
#     if 'show_qa_dialog' not in st.session_state:
#         st.session_state.show_qa_dialog = False
#     if 'show_kg_dialog' not in st.session_state:
#         st.session_state.show_kg_dialog = False

#     # ── Q&A Button
#     if st.button("📚 Send to Q&A", use_container_width=True, key="send_qa"):
#         if uploaded_files and len(uploaded_files) > 0:
#             st.session_state.show_qa_dialog = True
#             st.session_state.show_kg_dialog = False
#         else:
#             st.warning("Please upload files first.")

#     # ── Q&A inline card
#     if st.session_state.show_qa_dialog:
#         st.markdown("""
#         <div style="
#             background:#F0F6FF;
#             border:1.5px solid #4A9FD4;
#             border-left:4px solid #4A9FD4;
#             border-radius:6px;
#             padding:1.25rem 1.5rem 0.75rem 1.5rem;
#             margin-top:0.5rem;
#             margin-bottom:0.25rem;
#         ">
#             <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D2347;margin-bottom:0.2rem;">
#                 📚 Send to Q&amp;A
#             </div>
#             <div style="font-size:0.76rem;color:#5F7A9D;letter-spacing:0.04em;">
#                 Select the LLM model to use for answering questions
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#         qa_model = st.selectbox(
#             "LLM Model",
#             options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
#             index=0,
#             key="qa_model_dialog",
#             help="Ollama model for answering questions."
#         )
#         col_confirm, col_cancel = st.columns(2)
#         with col_confirm:
#             if st.button("✓ Confirm", use_container_width=True, key="qa_confirm"):
#                 st.session_state.show_qa_dialog = False
#                 qa_system.set_model(qa_model)
#                 st.info(f"Model: **{qa_model}**")
#                 with st.spinner("Indexing documents…"):
#                     added_count = 0
#                     for file in uploaded_files:
#                         if file.name not in st.session_state.databases:
#                             file.seek(0)
#                             temp_path = f"temp_qa_{file.name}"
#                             with open(temp_path, "wb") as f:
#                                 f.write(file.read())
#                             if qa_system.add_document(file.name, pdf_path=temp_path):
#                                 st.session_state.databases.append(file.name)
#                                 added_count += 1
#                             if os.path.exists(temp_path):
#                                 os.remove(temp_path)
#                     if added_count > 0:
#                         st.success(f"✓ {added_count} file(s) indexed.")
#                     else:
#                         st.info("Files already indexed.")
#         with col_cancel:
#             if st.button("✕ Cancel", use_container_width=True, key="qa_cancel"):
#                 st.session_state.show_qa_dialog = False
    #             st.rerun()
    
    # # ── KG Button
    # if st.button("◎ Generate Knowledge Graph", use_container_width=True, key="gen_kg"):
    #     if uploaded_files and len(uploaded_files) > 0:
    #         st.session_state.show_kg_dialog = True
    #         st.session_state.show_qa_dialog = False
    #     else:
    #         st.warning("Please upload files first.")

    # # ── KG Dialog
    # # ── KG inline card
    # if st.session_state.show_kg_dialog:
    #     st.markdown("""
    #     <div style="
    #         background:#F0F6FF;
    #         border:1.5px solid #4A9FD4;
    #         border-left:4px solid #4A9FD4;
    #         border-radius:6px;
    #         padding:1.25rem 1.5rem 0.75rem 1.5rem;
    #         margin-top:0.5rem;
    #         margin-bottom:0.25rem;
    #     ">
    #         <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D2347;margin-bottom:0.2rem;">
    #             ◎ Knowledge Graph Configuration
    #         </div>
    #         <div style="font-size:0.76rem;color:#5F7A9D;letter-spacing:0.04em;">
    #             Choose model and graph type before generating
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    #     kg_model = st.selectbox(
    #         "LLM Model for Relation Extraction",
    #         options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
    #         index=0,
    #         key="kg_model_dialog",
    #     )
    #     kg_graph_type = st.selectbox(
    #         "Graph Visualization Type",
    #         options=["Full Graph (with Datasets)", "Knowledge Graph Only (without Datasets)"],
    #         index=0,
    #         key="kg_graph_type_dialog",
    #     )
    #     col_confirm, col_cancel = st.columns(2)
    #     with col_confirm:
    #         if st.button("✓ Confirm", use_container_width=True, key="kg_confirm"):
    #             st.session_state.show_kg_dialog = False
    #             st.session_state.kg_model_selected = kg_model
    #             st.session_state.kg_graph_type_selected = kg_graph_type
    #             st.rerun()
    #     with col_cancel:
    #         if st.button("✕ Cancel", use_container_width=True, key="kg_cancel"):
    #             st.session_state.show_kg_dialog = False
    #             st.rerun()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── CG Button (only active after KG is generated)
    kg_ready = bool(st.session_state.processed_pdfs)
    cg_help = (
        "Extract causal relationships from the KG edges using a second LLM pass."
        if kg_ready else
        "Generate a Knowledge Graph first, then click here to extract causal relationships."
    )
    if st.button(
        "⟶ Generate Causal Graph",
        use_container_width=True,
        key="gen_cg",
        disabled=not kg_ready,
        help=cg_help
    ):
        st.session_state.show_cg_dialog = True
        st.session_state.show_kg_dialog = False
        st.session_state.show_qa_dialog = False

    # ── CG inline card
    if st.session_state.show_cg_dialog:
        st.markdown("""
        <div style="
            background:#FFF5F0;
            border:1.5px solid #E67E22;
            border-left:4px solid #C0392B;
            border-radius:6px;
            padding:1.25rem 1.5rem 0.75rem 1.5rem;
            margin-top:0.5rem;
            margin-bottom:0.25rem;
        ">
            <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D2347;margin-bottom:0.2rem;">
                ⟶ Causal Graph Configuration
            </div>
            <div style="font-size:0.76rem;color:#5F7A9D;letter-spacing:0.04em;">
                The LLM will re-analyse the KG edges to identify causal relationships
            </div>
        </div>
        """, unsafe_allow_html=True)

        cg_model = st.selectbox(
            "LLM Model for Causal Extraction",
            options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
            index=0,
            key="cg_model_dialog",
            help="Ollama model to reason over KG edges and identify causal links."
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ Confirm", use_container_width=True, key="cg_confirm"):
                st.session_state.show_cg_dialog = False
                st.session_state.cg_model_selected = cg_model
                st.rerun()
        with col_cancel:
            if st.button("✕ Cancel", use_container_width=True, key="cg_cancel"):
                st.session_state.show_cg_dialog = False
                st.rerun()

    # ── CG Processing
    if 'cg_model_selected' in st.session_state and st.session_state.get('cg_model_selected'):
        cg_model = st.session_state.cg_model_selected
        st.session_state.cg_model_selected = None

        all_kg_edges = []
        for data in st.session_state.processed_pdfs.values():
            all_kg_edges.extend(data.get('relations', []))

        if all_kg_edges:
            st.info(f"Sending {len(all_kg_edges)} KG edges to **{cg_model}** for causal analysis…")
            with st.spinner("Extracting causal relationships (Pass 2)…"):
                try:
                    causal_rels = extract_causal_relations(all_kg_edges, model=cg_model)
                    st.session_state.cg_relations = causal_rels
                except Exception as e:
                    st.error(f"Causal extraction error: {str(e)}")
            st.rerun()
        else:
            st.warning("No KG edges found. Please generate a Knowledge Graph first.")

    # ── KG Processing
    if 'kg_model_selected' in st.session_state and st.session_state.kg_model_selected and uploaded_files and len(uploaded_files) > 0:
        kg_model = st.session_state.kg_model_selected
        kg_graph_type_ui = st.session_state.get('kg_graph_type_selected', "Full Graph (with Datasets)")
        graph_type_map = {
            "Full Graph (with Datasets)": "with_datasets",
            "Knowledge Graph Only (without Datasets)": "without_datasets"
        }
        kg_graph_type = graph_type_map[kg_graph_type_ui]
        st.session_state.kg_model_selected = None
        st.session_state.kg_graph_type_selected = None

        st.info(f"Model: **{kg_model}** · Graph: **{kg_graph_type_ui}**")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        all_keywords = []
        all_datasets = []

        for idx, file in enumerate(uploaded_files):
            progress_text.text(f"Processing {file.name}… ({idx+1}/{total_files})")
            progress_bar.progress((idx + 1) / total_files)
            try:
                file.seek(0)
                file_content = file.read()
                temp_filename = f"temp_{idx}_{file.name.replace(' ', '_')}"
                with open(temp_filename, "wb") as f:
                    f.write(file_content)
                nodes, relations, datasets, keywords_metadata = process(
                    temp_filename, k=k, filter_variables=filter_variables,
                    llm_model=kg_model, use_gpt4_datasets=use_gpt4_datasets
                )
                if keywords_metadata and keywords_metadata.get('from_keywords_section'):
                    st.success(f"Keywords section found in {file.name} — {keywords_metadata['total_found']} extracted.")
                elif keywords_metadata:
                    st.info(f"No keywords section in {file.name}. Using {keywords_metadata.get('method', 'algorithmic extraction')}.")

                if keywords_metadata and keywords_metadata.get('filtering_applied'):
                    original_kw = keywords_metadata.get('original_keywords', [])
                    filtered_kw = keywords_metadata.get('filtered_keywords', [])
                    removed_kw = keywords_metadata.get('removed_keywords', [])
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Original Keywords", len(original_kw))
                    with col2: st.metric("Climate Variables", len(filtered_kw), delta=f"{len(filtered_kw)/len(original_kw)*100:.1f}%" if original_kw else "0%")
                    with col3: st.metric("Filtered Out", len(removed_kw), delta=f"-{len(removed_kw)/len(original_kw)*100:.1f}%" if original_kw else "0%")
                    with st.expander(f"Variable Filtering Details — {file.name}"):
                        col_kept, col_removed = st.columns(2)
                        with col_kept:
                            st.markdown("**Variables Kept:**")
                            st.write(", ".join(filtered_kw[:20]) + (f" … +{len(filtered_kw)-20} more" if len(filtered_kw) > 20 else "") if filtered_kw else "None")
                        with col_removed:
                            st.markdown("**Removed:**")
                            st.write(", ".join(removed_kw[:10]) + (f" … +{len(removed_kw)-10} more" if len(removed_kw) > 10 else "") if removed_kw else "None")

                extraction_stats = keywords_metadata.get('extraction_stats', {}) if keywords_metadata else {}
                if file.name not in st.session_state.processed_pdfs:
                    st.session_state.processed_pdfs[file.name] = {
                        'nodes': nodes, 'relations': relations, 'datasets': datasets,
                        'keywords_metadata': keywords_metadata, 'used_gpt4': use_gpt4_datasets,
                        'extraction_cost': extraction_stats.get('total_cost', 0),
                        'graph_type': kg_graph_type
                    }
                else:
                    st.session_state.processed_pdfs[file.name]['nodes'].extend(nodes)
                    st.session_state.processed_pdfs[file.name]['relations'].extend(relations)

                all_keywords.extend(nodes)
                if datasets:
                    for ds in datasets:
                        if ds.get('source') != 'Not specified':
                            all_datasets.append(ds.get('source'))
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

        progress_text.empty()
        progress_bar.empty()
        st.success(f"✓ Knowledge graphs generated for {total_files} file(s).")
        st.info("Tip — click 'Generate Causal Graph' below to discover causal relationships from the KG edges.")

        if filter_variables:
            st.markdown('<span class="section-label">Variable Filtering Summary</span>', unsafe_allow_html=True)
            total_original = total_kept = total_removed = 0
            for pdf_name, pdf_data in st.session_state.processed_pdfs.items():
                if pdf_data.get('keywords_metadata', {}).get('filtering_applied'):
                    meta = pdf_data['keywords_metadata']
                    total_original += len(meta.get('original_keywords', []))
                    total_kept += len(meta.get('filtered_keywords', []))
                    total_removed += len(meta.get('removed_keywords', []))
            if total_original > 0:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Total Keywords", total_original)
                with c2: st.metric("Climate Variables", total_kept, delta=f"{total_kept/total_original*100:.1f}%")
                with c3: st.metric("Filtered Out", total_removed, delta=f"-{total_removed/total_original*100:.1f}%")
                with c4: st.metric("Retention Rate", f"{total_kept/total_original*100:.1f}%")

        if all_keywords:
            st.markdown('<span class="section-label">Extracted Climate Variables</span>', unsafe_allow_html=True)
            unique_kw = list(set(all_keywords))
            kw_html = "".join(f'<span class="kw-tag">{kw}</span>' for kw in unique_kw[:30])
            if len(unique_kw) > 30:
                kw_html += f'<span class="kw-tag-light">+{len(unique_kw)-30} more</span>'
            st.markdown(kw_html, unsafe_allow_html=True)

        if all_datasets:
            st.markdown('<span class="section-label">Datasets Identified</span>', unsafe_allow_html=True)
            for ds in set(all_datasets):
                st.markdown(f'<div class="polar-info-row">◎ {ds}</div>', unsafe_allow_html=True)

        st.rerun()  # re-render so CG button reads populated processed_pdfs and becomes enabled


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 1b — DATASET HARVESTER (auto-runs on upload)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<span class="section-label">Dataset Intelligence</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Referenced <em>Datasets</em></div>', unsafe_allow_html=True)

if not _HARVESTER_AVAILABLE:
    st.warning("Dataset harvester module not found — check that `dataset_harvester/` is in the project root.")
elif uploaded_files and len(uploaded_files) > 0:
    cache_key = "_".join(f"{f.name}:{f.size}" for f in uploaded_files)

    if st.session_state.get("harvester_cache_key") != cache_key:
        st.session_state.harvester_cache_key = cache_key
        st.session_state.harvester_results = None

    if st.session_state.get("harvester_results") is None:
        import time as _time
        _t0 = _time.time()

        # Steps 1+2 need the PDF files on disk
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_paths = []
            for f in uploaded_files:
                f.seek(0)
                p = os.path.join(tmpdir, f.name)
                with open(p, "wb") as out:
                    out.write(f.read())
                pdf_paths.append(p)

            with st.status("Extracting dataset references...", expanded=True) as s1:
                _t1 = _time.time()
                refs_by_source = extract_from_pdfs(pdf_paths, use_llm=True)
                raw_count = sum(len(v) for v in refs_by_source.values())
                s1.update(label=f"{raw_count} raw references extracted ({_time.time()-_t1:.1f}s)", state="complete", expanded=False)

            with st.status("Deduplicating...", expanded=True) as s2:
                _t2 = _time.time()
                datasets = deduplicate(refs_by_source)
                s2.update(label=f"{raw_count} raw → {len(datasets)} unique datasets ({_time.time()-_t2:.1f}s)", state="complete", expanded=False)

        # Steps 3+4: resolve + check in parallel (no files needed)
        def _resolve_and_check(d):
            r = resolve_one(d)
            sk, lbl, det = check_downloadable(r)
            return r, sk, lbl, det

        rows = [None] * len(datasets)
        with st.status(f"Resolving & checking {len(datasets)} datasets...", expanded=True) as s34:
            _prog = st.progress(0, text="Starting…")
            _done = 0
            with ThreadPoolExecutor(max_workers=10) as _pool:
                _futures = {_pool.submit(_resolve_and_check, d): i for i, d in enumerate(datasets)}
                for _fut in _as_completed(_futures):
                    _idx = _futures[_fut]
                    try:
                        _r, _sk, _lbl, _det = _fut.result()
                    except Exception as _exc:
                        _d = datasets[_idx]
                        _r = ResolvedDataset(
                            canonical_name=_d.canonical_name, resolved_url=None,
                            repository=None, doi=_d.doi, accession=_d.accession,
                            notes=str(_exc), resolution_method="unresolved",
                            mention_count=_d.mention_count, sources=_d.sources,
                            is_primary=_d.is_primary,
                        )
                        _sk, _lbl, _det = "unknown", "Error", str(_exc)
                    rows[_idx] = {
                        "Dataset": _r.canonical_name,
                        "URL": _r.resolved_url or "",
                        "Repository": _r.repository or "unknown",
                        "Status": _lbl,
                        "Detail": _det,
                        "_status_key": _sk,
                        "_is_primary": _r.is_primary,
                        "_mention_count": _r.mention_count,
                        "_doi": _r.doi or "",
                        "_accession": _r.accession or "",
                    }
                    _done += 1
                    _prog.progress(_done / len(datasets), text=f"{_done}/{len(datasets)} processed")
            _resolved_count = sum(1 for row in rows if row["URL"])
            _dl_count = sum(1 for row in rows if row["_status_key"] == "yes")
            s34.update(label=f"{_resolved_count} resolved · {_dl_count} downloadable ({_time.time()-_t0:.1f}s total)", state="complete", expanded=False)

        st.session_state.harvester_results = rows
        st.session_state.harvester_elapsed = _time.time() - _t0

    rows = st.session_state.harvester_results
    if rows:
        # Heuristic fallback: if LLM didn't mark any primary, use highest mention count
        if not any(r["_is_primary"] for r in rows):
            best = max(rows, key=lambda r: r["_mention_count"])
            best["_is_primary"] = True
            best["_primary_how"] = "heuristic"
        for r in rows:
            if "_primary_how" not in r:
                r["_primary_how"] = "llm" if r["_is_primary"] else ""

        primary_rows = [r for r in rows if r["_is_primary"]]
        secondary_rows = [r for r in rows if not r["_is_primary"]]
        elapsed = st.session_state.get("harvester_elapsed")

        # Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total datasets found", len(rows))
        c2.metric("Active dataset(s)", len(primary_rows))
        c3.metric("Secondary references", len(secondary_rows))
        c4.metric("Directly downloadable", sum(1 for r in primary_rows if r["_status_key"] == "yes"))
        c5.metric("Time taken", f"{elapsed:.1f}s" if elapsed else "—")

        st.divider()

        # ── ACTIVE DATASETS ────────────────────────────────────────────────
        st.markdown("### Active Dataset(s)")
        st.caption("The dataset(s) this paper introduces, collects, or primarily analyses — these are the ones you want to download.")

        if "dl_results" not in st.session_state:
            st.session_state.dl_results = {}

        for i, p in enumerate(primary_rows):
            heuristic_note = (" *(identified by mention frequency — LLM did not mark a primary)*"
                              if p["_primary_how"] == "heuristic" else "")
            can_dl = p["_status_key"] == "yes" and _DOWNLOADERS_AVAILABLE

            with st.container():
                st.markdown(
                    f"""<div style="border:2px solid #2E5FA0;border-radius:8px;padding:1rem 1.4rem;
                                background:#EEF3FB;margin-bottom:0.5rem;">
  <strong style="font-size:1.05rem;color:#0D2347">{p['Dataset']}</strong>{heuristic_note}<br>
  <span style="color:#555;font-size:0.85rem">Repository:</span> <code>{p['Repository']}</code>
  &emsp;{p['Status']}<br>
  <span style="color:#555;font-size:0.85rem">Detail:</span> <span style="font-size:0.85rem">{p['Detail']}</span><br>
  {"<a href='" + p['URL'] + "' target='_blank' style='color:#2E5FA0;font-size:0.85rem'>" + p['URL'][:90] + ("…" if len(p['URL']) > 90 else "") + "</a>" if p['URL'] else "<em style='color:#888;font-size:0.85rem'>URL not resolved</em>"}
</div>""",
                    unsafe_allow_html=True,
                )

                result_key = f"dlresult_{i}_{p['Dataset'][:30]}"
                col_res_key = f"col_children_{i}"

                if can_dl:
                    if st.button("Download to disk", key=f"dl_btn_{i}"):
                        with st.status("Downloading…", expanded=True) as _dl_status:
                            try:
                                files = _do_download(p)
                                st.session_state.dl_results[result_key] = ("ok", files)
                                _dl_status.update(
                                    label=f"Downloaded {len(files)} file(s)",
                                    state="complete", expanded=False
                                )
                            except Exception as _dl_err:
                                st.session_state.dl_results[result_key] = ("err", str(_dl_err))
                                _dl_status.update(
                                    label=f"Download failed: {_dl_err}",
                                    state="error", expanded=False
                                )

                elif p["_status_key"] == "collection":
                    if st.button("📂 List Child Datasets", key=f"col_btn_{i}"):
                        _pid_m = re.search(
                            r'PANGAEA\.(\d+)\b',
                            " ".join([p.get("URL") or "", p.get("_doi") or "", p.get("_accession") or ""]),
                            re.IGNORECASE,
                        )
                        if _pid_m:
                            with st.spinner("Fetching child datasets from PANGAEA…"):
                                st.session_state[col_res_key] = _fetch_collection_children(_pid_m.group(1))
                        else:
                            st.session_state[col_res_key] = []

                elif p["_status_key"] == "auth":
                    st.info(f"Requires account on {p['Repository']} — {p['Detail']}. Visit the link above to access.")
                elif p["_status_key"] in ("page", "none"):
                    st.info("No direct file available — use the link above to access the data manually.")

                # Collection children list
                if col_res_key in st.session_state:
                    children = st.session_state[col_res_key]
                    if children:
                        st.markdown(f"**{len(children)} dataset(s) in this collection:**")
                        for j, c in enumerate(children):
                            _c_dl_key = f"child_dl_{i}_{j}"
                            _c_res_key = f"child_dl_result_{i}_{j}"
                            _ccol1, _ccol2 = st.columns([5, 1])
                            with _ccol1:
                                if c["url"]:
                                    st.markdown(f"- [{c['name']}]({c['url']})")
                                else:
                                    st.markdown(f"- {c['name']}")
                            with _ccol2:
                                if c.get("downloadable") and st.button("⬇ Download", key=_c_dl_key):
                                    _child_row = {
                                        "URL": c["url"],
                                        "_doi": f"10.1594/PANGAEA.{c['id']}",
                                        "_accession": "",
                                        "Repository": "pangaea",
                                    }
                                    with st.spinner(f"Downloading {c['name']}…"):
                                        try:
                                            _child_files = _do_download(_child_row)
                                            st.session_state[_c_res_key] = ("ok", _child_files)
                                        except Exception as _ce:
                                            st.session_state[_c_res_key] = ("err", str(_ce))
                            if _c_res_key in st.session_state:
                                _c_ok, _c_data = st.session_state[_c_res_key]
                                if _c_ok == "ok":
                                    st.success(f"{len(_c_data)} file(s) saved:")
                                    for _cfpath in _c_data:
                                        st.code(_cfpath)
                                        _cdf = _preview_file(_cfpath)
                                        if _cdf is not None and not _cdf.empty:
                                            st.caption(f"First {len(_cdf)} row(s) — {os.path.basename(_cfpath)}")
                                            st.dataframe(_cdf, use_container_width=True, hide_index=True)
                                else:
                                    st.error(f"Download failed: {_c_data}")
                    else:
                        st.warning("Could not retrieve child datasets — visit the URL above to browse the collection manually.")

                # Download result + first-10-rows preview
                if result_key in st.session_state.dl_results:
                    ok, data = st.session_state.dl_results[result_key]
                    if ok == "ok":
                        st.success(f"{len(data)} file(s) saved to disk:")
                        for fpath in data:
                            st.code(fpath)
                            df_preview = _preview_file(fpath)
                            if df_preview is not None and not df_preview.empty:
                                st.caption(
                                    f"First {len(df_preview)} row(s) — {os.path.basename(fpath)}"
                                )
                                st.dataframe(df_preview, use_container_width=True, hide_index=True)
                    else:
                        st.error(f"Download failed: {data}")

        st.divider()

        # ── SECONDARY REFERENCES ───────────────────────────────────────────
        st.markdown("### Secondary References")
        st.caption("Other datasets cited in this paper (forcing data, reanalysis products, validation sources). Links provided for manual access.")

        if secondary_rows:
            sec_display = [{
                "Dataset": r["Dataset"][:80] + ("…" if len(r["Dataset"]) > 80 else ""),
                "Repository": r["Repository"],
                "Status": r["Status"],
                "URL": r["URL"],
            } for r in secondary_rows]
            st.dataframe(sec_display, use_container_width=True, hide_index=True,
                column_config={
                    "URL": st.column_config.LinkColumn("URL", display_text="Open"),
                    "Status": st.column_config.TextColumn("Status", width="medium"),
                })
        else:
            st.info("No secondary references identified.")
else:
    st.markdown(
        '<div style="padding:1rem;background:#F0F4FA;border-radius:6px;color:#5F7A9D;font-size:0.85rem;">'
        'Upload PDFs above — dataset references will appear here automatically.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Q&A
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div id="section-qa"></div>', unsafe_allow_html=True)
st.markdown('<span class="section-label">Step 02</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Document <em>Q&A</em></div>', unsafe_allow_html=True)

if qa_system.list_documents():
    st.success(f"✓ Q&A System Ready — {len(qa_system.list_documents())} document(s) indexed")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("Clear Chat History", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    with col_r2:
        if st.button("Reset Q&A System", use_container_width=True, key="reset_qa"):
            qa_system.reset_and_reload()
            st.session_state.databases = []
            st.session_state.chat_history = []
            st.rerun()
else:
    st.warning("No documents indexed — upload PDFs and click 'Send to Q&A' above.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<span class="section-label">Indexed Documents</span>', unsafe_allow_html=True)
    if st.session_state.databases:
        for db in st.session_state.databases:
            st.markdown(f'<div class="doc-item">{db}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state" style="padding:2rem;"><span class="empty-state-glyph">&#8599;</span><div class="empty-state-text">No documents indexed yet.</div></div>', unsafe_allow_html=True)

with col2:
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-bubble-user"><span class="chat-label chat-label-user">You</span>{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-assistant"><span class="chat-label chat-label-ai">PolarKD</span>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state"><span class="empty-state-glyph">&#8596;</span><div class="empty-state-title">Start a conversation</div><div class="empty-state-text">Try: "What datasets were used?" &middot; "Summarise the findings." &middot; "What methods were employed?" &middot; "What is the study time period?"</div></div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question…",
            placeholder="Type your question here…",
            label_visibility="collapsed"
        )
        col_in, col_send = st.columns([4, 1])
        with col_send:
            submit = st.form_submit_button("Send →", use_container_width=True)
        if submit and user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            with st.spinner("Thinking…"):
                try:
                    response = qa_system.answer_question(user_input)
                except Exception as e:
                    response = f"Error: {str(e)}. Ensure Ollama is running and accessible."
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div id="section-kg"></div>', unsafe_allow_html=True)
st.markdown('<span class="section-label">Step 03</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Knowledge <em>Graph</em></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="graph-legend">'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#0D2347;"></div> Entity</div>'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#4A9FD4;"></div> Relationship</div>'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#2E5FA0;"></div> Concept</div>'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.processed_pdfs:
    st.markdown('<span class="section-label">Processing Summary</span>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Files Processed", len(st.session_state.processed_pdfs))
    with c2: st.metric("Total Keywords", sum(len(d.get('nodes', [])) for d in st.session_state.processed_pdfs.values()))
    with c3: st.metric("Total Relations", sum(len(d.get('relations', [])) for d in st.session_state.processed_pdfs.values()))
    with c4: st.metric("Files", len(st.session_state.processed_pdfs))

    dataset_filter = display_dataset_filter()
    display_datasets_section(st.session_state.processed_pdfs, dataset_filter)
    display_cost_summary(st.session_state.processed_pdfs)

    all_keywords = []
    keywords_by_file = {}
    for filename, data in st.session_state.processed_pdfs.items():
        fk = data.get('nodes', [])
        all_keywords.extend(fk)
        keywords_by_file[filename] = fk

    if all_keywords:
        st.markdown('<span class="section-label">Extracted Keywords</span>', unsafe_allow_html=True)
        for filename, keywords in keywords_by_file.items():
            st.markdown(f"**{filename}**")
            kw_html = "".join(f'<span class="kw-tag">{kw}</span>' for kw in keywords[:10])
            st.markdown(kw_html, unsafe_allow_html=True)
        st.markdown(f"<small style='color:#5F7A9D;font-size:0.75rem;'>Total unique keywords: {len(set(all_keywords))}</small>", unsafe_allow_html=True)

    try:
        neo = Neo4jConnector()
        all_nodes, all_relations, all_datasets = [], [], []

        st.markdown(f"<small style='color:#5F7A9D;'>Combining data from {len(st.session_state.processed_pdfs)} files…</small>", unsafe_allow_html=True)
        for filename, data in st.session_state.processed_pdfs.items():
            nodes = data.get('nodes', [])
            relations = data.get('relations', [])
            file_datasets = data.get('datasets', [])
            st.markdown(f"<small style='color:#5F7A9D;'>— {filename}: {len(nodes)} nodes · {len(relations)} relations · {len(file_datasets) if file_datasets else 0} dataset(s)</small>", unsafe_allow_html=True)
            all_nodes.extend(nodes)
            all_relations.extend(relations)
            if file_datasets:
                for ds in file_datasets:
                    if ds.get('source') != 'Not specified':
                        all_datasets.append(ds)

        if all_nodes and all_relations:
            st.markdown(f"<small style='color:#5F7A9D;'>Total: {len(all_nodes)} nodes · {len(all_relations)} relations · {len(all_datasets)} dataset(s)</small>", unsafe_allow_html=True)
            graph_type_to_use = 'with_datasets'
            if st.session_state.processed_pdfs:
                first_file = list(st.session_state.processed_pdfs.keys())[0]
                graph_type_to_use = st.session_state.processed_pdfs[first_file].get('graph_type', 'with_datasets')
            st.info(f"Graph mode: **{graph_type_to_use.replace('_', ' ').title()}**")
            neo.store_keywords_and_relations(all_nodes, all_relations, all_datasets)
            rels = neo.retrieve_relations()
            graph, expansion_js = neo.generate_graph(rels, graph_type=graph_type_to_use)
            graph.save_graph("graph.html")
            with open("graph.html", "r") as f:
                html_content = f.read()
            html_content = html_content.replace("</body>", expansion_js + "</body>")
            with open("graph.html", "w") as f:
                f.write(html_content)
            with open("graph.html", "r") as f:
                html_string = f.read()
            st.components.v1.html(html_string, height=500, scrolling=True)
            neo.close()

            st.markdown('<span class="section-label">Graph Statistics</span>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Unique Nodes", len(set(all_nodes)))
            with c2: st.metric("Total Relations", len(all_relations))
            with c3: st.metric("Datasets Found", len(all_datasets))
            with c4:
                avg = len(all_relations) // len(st.session_state.processed_pdfs) if st.session_state.processed_pdfs else 0
                st.metric("Avg Relations / File", avg)

    except Exception as e:
        st.error(f"Neo4j error: {str(e)}")
        st.info("Please check your Neo4j credentials.")

else:
    st.markdown(
        '<div class="empty-state">'
        '<svg width="120" height="120" viewBox="0 0 200 200" style="opacity:0.25;margin-bottom:1.5rem;">'
        '<circle cx="100" cy="50" r="14" fill="#0D2347"/>'
        '<circle cx="50" cy="110" r="14" fill="#0D2347"/>'
        '<circle cx="150" cy="110" r="14" fill="#0D2347"/>'
        '<circle cx="75" cy="160" r="14" fill="#0D2347"/>'
        '<circle cx="125" cy="160" r="14" fill="#0D2347"/>'
        '<circle cx="100" cy="100" r="18" fill="#4A9FD4"/>'
        '<line x1="100" y1="100" x2="100" y2="50" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="50" y2="110" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="150" y2="110" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="75" y2="160" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="125" y2="160" stroke="#B8CCE8" stroke-width="1.5"/>'
        '</svg>'
        '<div class="empty-state-title">No graph generated yet</div>'
        '<div class="empty-state-text">Upload PDFs and click "Generate Knowledge Graph" to visualise extracted climate variables and their relationships.</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — CAUSAL GRAPH
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div id="section-cg"></div>', unsafe_allow_html=True)
st.markdown('<span class="section-label">Step 04</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Causal <em>Graph</em></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="graph-legend">'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#C0392B;"></div> Root Cause</div>'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#8E44AD;"></div> Intermediate</div>'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#E67E22;"></div> Terminal Effect</div>'
    '<div class="graph-legend-item" style="font-size:0.68rem;color:#5F7A9D !important;">Arrows show causal direction →</div>'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.cg_relations:
    causal_rels = st.session_state.cg_relations

    st.markdown('<span class="section-label">Causal Summary</span>', unsafe_allow_html=True)
    all_causes = list({r['cause'] for r in causal_rels})
    all_effects = list({r['effect'] for r in causal_rels})
    root_causes = [c for c in all_causes if c not in {r['effect'] for r in causal_rels}]
    terminal_effects = [e for e in all_effects if e not in {r['cause'] for r in causal_rels}]
    avg_conf = sum(r['confidence'] for r in causal_rels) / len(causal_rels)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Causal Pairs", len(causal_rels))
    with c2: st.metric("Unique Variables", len(set(all_causes) | set(all_effects)))
    with c3: st.metric("Root Causes", len(root_causes))
    with c4: st.metric("Avg Confidence", f"{avg_conf:.2f}")

    if root_causes:
        st.markdown('<span class="section-label">Root Causes Identified</span>', unsafe_allow_html=True)
        rc_html = "".join(f'<span class="causal-cause-tag">{rc}</span>' for rc in root_causes)
        st.markdown(rc_html, unsafe_allow_html=True)

    if terminal_effects:
        st.markdown('<span class="section-label">Terminal Effects</span>', unsafe_allow_html=True)
        te_html = "".join(f'<span class="causal-effect-tag">{te}</span>' for te in terminal_effects)
        st.markdown(te_html, unsafe_allow_html=True)

    try:
        _, cg_html = generate_causal_graph(causal_rels)
        st.components.v1.html(cg_html, height=500, scrolling=True)
    except Exception as e:
        st.error(f"Graph rendering error: {str(e)}")

    with st.expander("View All Causal Relationships"):
        for rel in sorted(causal_rels, key=lambda x: -x['confidence']):
            conf_color = "#C0392B" if rel['confidence'] >= 0.7 else "#E67E22" if rel['confidence'] >= 0.4 else "#F39C12"
            st.markdown(
                f'<div class="polar-info-row" style="border-left-color:{conf_color};">'
                f'<b>{rel["cause"]}</b> &nbsp;⟶&nbsp; <span style="color:{conf_color};font-weight:600;">{rel["label"]}</span> &nbsp;⟶&nbsp; <b>{rel["effect"]}</b>'
                f'&nbsp;&nbsp;<span style="font-size:0.72rem;color:#5F7A9D;">confidence: {rel["confidence"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
else:
    st.markdown(
        '<div class="empty-state">'
        '<svg width="120" height="120" viewBox="0 0 200 200" style="opacity:0.25;margin-bottom:1.5rem;">'
        '<circle cx="40" cy="100" r="14" fill="#C0392B"/>'
        '<circle cx="100" cy="60" r="14" fill="#8E44AD"/>'
        '<circle cx="100" cy="140" r="14" fill="#8E44AD"/>'
        '<circle cx="160" cy="100" r="14" fill="#E67E22"/>'
        '<line x1="54" y1="100" x2="86" y2="68" stroke="#C0392B" stroke-width="2"/>'
        '<line x1="54" y1="100" x2="86" y2="132" stroke="#C0392B" stroke-width="2"/>'
        '<line x1="114" y1="60" x2="146" y2="92" stroke="#8E44AD" stroke-width="2"/>'
        '<line x1="114" y1="140" x2="146" y2="108" stroke="#8E44AD" stroke-width="2"/>'
        '</svg>'
        '<div class="empty-state-title">No causal graph yet</div>'
        '<div class="empty-state-text">First generate a Knowledge Graph, then click "Generate Causal Graph" to discover causal relationships using a second LLM pass.</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ─── EXPORT ────────────────────────────────────────────────────────────────
if st.session_state.processed_pdfs:
    st.markdown("---")
    st.markdown('<span class="section-label">Export</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.4rem!important;">Download <em>Results</em></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    all_relations = []
    for data in st.session_state.processed_pdfs.values():
        all_relations.extend(data.get('relations', []))

    with col1:
        if all_relations:
            json_data = json.dumps(all_relations, indent=2)
            st.download_button(
                label="Export KG — JSON",
                data=json_data,
                file_name="knowledge_graph.json",
                mime="application/json",
                use_container_width=True
            )
    with col2:
        if all_relations:
            df = pd.DataFrame(all_relations)
            st.download_button(
                label="Export KG — CSV",
                data=df.to_csv(index=False),
                file_name="knowledge_graph.csv",
                mime="text/csv",
                use_container_width=True
            )
    with col3:
        datasets_csv = export_datasets_to_csv(st.session_state.processed_pdfs)
        if datasets_csv:
            st.download_button(
                label="Export Datasets — CSV",
                data=datasets_csv,
                file_name="extracted_datasets.csv",
                mime="text/csv",
                use_container_width=True
            )
    with col4:
        if st.session_state.cg_relations:
            cg_df = pd.DataFrame(st.session_state.cg_relations)
            st.download_button(
                label="Export Causal — CSV",
                data=cg_df.to_csv(index=False),
                file_name="causal_graph.csv",
                mime="text/csv",
                use_container_width=True
            )


# ─── FOOTER ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="polar-footer">'
    '<div class="polar-footer-brand">PolarKD</div>'
    '<div class="polar-footer-links"><a href="#">About</a><a href="#">Documentation</a><a href="#">Contact</a><a href="#">Privacy</a></div>'
    '<div class="polar-footer-copy">AI-powered polar science document intelligence &middot; iHARP &copy; 2024</div>'
    '</div>',
    unsafe_allow_html=True
)
