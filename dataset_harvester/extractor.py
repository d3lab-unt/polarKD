"""
Iteration 1 — PDF Dataset Reference Extractor

Extracts dataset references from research PDFs using two passes:
  1. Regex: explicit URLs, DOIs, accession numbers
  2. LLM (GPT-4o-mini): implicit dataset mentions without a direct link

Output is a list of DatasetRef objects ready for deduplication (Iteration 2).
"""

import re
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import pdfplumber
import requests

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DatasetRef:
    name: str                          # Human-readable dataset name
    url: Optional[str] = None          # Direct URL if found
    doi: Optional[str] = None          # Data DOI (not paper DOI)
    accession: Optional[str] = None    # Repository accession number
    repository_hint: Optional[str] = None  # e.g. "pangaea", "zenodo", "nsidc"
    raw_citation: str = ""             # Original text snippet that mentioned it
    source: str = ""                   # "regex" | "llm"
    is_primary: bool = False           # True for the paper's one central/introduced dataset
    used_in_study: bool = False        # True if the text indicates this dataset was actually used in the paper

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Regex patterns — arctic/env research common patterns
# ---------------------------------------------------------------------------

# Standard HTTP/HTTPS URLs (not purely paper/journal URLs)
_URL_RE = re.compile(
    r'https?://[^\s\)\(\],\'"<>]{10,}',
    re.IGNORECASE
)

# Data DOIs — 10.XXXX/... format, skipping common journal publishers
_DOI_RE = re.compile(
    r'\b(10\.\d{4,9}/[^\s\)\(\],\'"<>]+)',
    re.IGNORECASE
)

# PANGAEA accession  e.g. PANGAEA.123456 or doi:10.1594/PANGAEA.123456
_PANGAEA_RE = re.compile(r'PANGAEA[.\s]*(\d{5,7})', re.IGNORECASE)

# Zenodo record  e.g. zenodo.org/record/1234567 or zenodo.1234567
_ZENODO_RE = re.compile(r'zenodo\.org/(?:record|records)/(\d+)', re.IGNORECASE)

# NSIDC accession  e.g. NSIDC-0001  or  nsidc.org/data/NSIDC-0001
_NSIDC_RE = re.compile(r'NSIDC[-\s]*(\d{4})', re.IGNORECASE)

# Arctic Data Center  https://arcticdata.io/catalog/view/doi:...
_ADC_RE = re.compile(r'arcticdata\.io[^\s\)\],\'"<>]*', re.IGNORECASE)

# Copernicus / CDS
_CDS_RE = re.compile(r'cds\.climate\.copernicus\.eu[^\s\)\],\'"<>]*', re.IGNORECASE)

# Known dataset names that appear without URLs in arctic literature
_KNOWN_DATASETS = [
    "ERA5", "ERA-Interim", "ERA-40",
    "MODIS", "VIIRS", "Landsat",
    "Sentinel-1", "Sentinel-2", "Sentinel-3", "Sentinel-5P",
    "AMSR-E", "AMSR2", "SSMIS", "SSM/I",
    "ICESat", "ICESat-2", "CryoSat-2",
    "GRACE", "GRACE-FO",
    "PIOMAS", "TOPAZ", "TOPAZ4",
    "NCEP", "NCEP/NCAR", "CFSR", "CFSv2",
    "JRA-55", "MERRA-2",
    "AVHRR", "GOES",
    "NSIDC Sea Ice Index",
    "HadGHCND", "HadSST", "HadCRUT",
    "GISTEMP", "Berkeley Earth",
    "ETOPO", "GEBCO",
    "WOA", "World Ocean Atlas",
    "ARGO", "Argo",
    "MOSAiC",
    "SHEBA",
    "IABP",
    "IMB",
    # Climate model intercomparison
    "CMIP3", "CMIP5", "CMIP6",
    # Ocean/atmosphere models frequently cited as data sources
    "HYCOM", "FESOM", "NEMO", "ROMS",
    "WRF",
    # Ocean reanalysis / products
    "GLORYS", "GLORYS12", "OSTIA",
    # Sea-ice satellite products
    "OSI-SAF", "SMOS",
    # Precipitation datasets
    "GPM", "IMERG", "GPCP", "TRMM", "CMORPH", "PERSIANN",
    # Atmospheric / climate
    "CALIPSO", "CloudSat", "CAMSRA",
    # Climate model families
    "AWI-CM", "MPI-ESM", "CESM", "CESM2",
]

_KNOWN_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(d) for d in _KNOWN_DATASETS) + r')\b'
)

# DOIs that belong to journal publishers — skip these (not datasets)
_JOURNAL_DOI_PREFIXES = {
    "10.1016", "10.1038", "10.1126", "10.1175", "10.1029",
    "10.1007", "10.3390", "10.1002", "10.1080", "10.1017",
    "10.5194", "10.1073", "10.3189", "10.1098", "10.1007",
    "10.1177", "10.1093", "10.1111", "10.2307", "10.1146",
    "10.1214", "10.1137", "10.1257", "10.1353", "10.2307",
    "10.1515", "10.1017",
    "10.1525",  # UC Press / Elementa journal
    "10.13039", # CrossRef Funder Registry
    "10.17815", # JLSRF journal
    "10.2312",  # AWI Berichte (reports/publications)
    "10.5670",  # Oceanography (The Oceanography Society)
    "10.1109",  # IEEE (Transactions, TGRS, JSTARS, etc.)
    "10.3389",  # Frontiers journals
    "10.1088",  # IOP Publishing
}

# DOI prefixes that are definitely data repositories — always keep these
_DATA_REPO_DOI_PREFIXES = {
    "10.1594",  # PANGAEA
    "10.5281",  # Zenodo
    "10.7289",  # NOAA NCEI
    "10.5067",  # NASA EOSDIS / NSIDC
    "10.18739", # Arctic Data Center
    "10.6073",  # BCO-DMO
    "10.6084",  # Figshare
    "10.5061",  # Dryad
    "10.26050", # SEANOE
    "10.17882", # SEANOE (alt prefix)
    "10.3334",  # ORNL DAAC
    "10.5285",  # BODC (British Oceanographic Data Centre)
    "10.5441",  # PANGAEA (alt prefix)
    "10.5884",  # Polar Data Catalogue (Canada)
    "10.26071", # OGSL / St. Lawrence Global Observatory
    "10.17616", # re3data (data repository registry)
    "10.48670", # CMEMS (Copernicus Marine)
    "10.24381", # Copernicus Climate Data Store
    "10.5439",  # ARM (Atmospheric Radiation Measurement) facility
    "10.25923", # NOAA NCEI (alternate prefix)
    "10.22008", # GEUS / Arctic Greenland data portals
    "10.7265",  # NSIDC (alternate dataset DOI prefix)
}

# URLs from sites that are never data repositories
_JOURNAL_URL_DOMAINS = {
    "journals.ametsoc.org", "pnas.org", "nature.com", "science.org",
    "sciencedirect.com", "springer.com", "wiley.com", "tandfonline.com",
    "cambridge.org", "agu.org", "essoar.org", "arxiv.org",
    "researchgate.net", "semanticscholar.org", "carbonbrief.org",
    "newengineer.com", "gsas.harvard.edu", "un.org", "usembassy.gov",
    "wwf.org", "wwf.org.uk", "environment.nsw.gov.au",
    "greatwhitecon.info",
}


def _is_data_doi(doi: str) -> bool:
    prefix = doi.split("/")[0].lower()
    if prefix in _DATA_REPO_DOI_PREFIXES:
        return True   # explicitly known data repo — always keep
    return prefix not in _JOURNAL_DOI_PREFIXES


def _is_journal_url(url: str) -> bool:
    try:
        domain = url.split("//", 1)[1].split("/")[0].lower()
        domain = domain.lstrip("www.")
        if domain in _JOURNAL_URL_DOMAINS:
            return True
        # doi.org resolver pointing at a journal article — treat as journal URL
        if domain == "doi.org":
            m = re.search(r'doi\.org/(10\.\d{4,9})', url, re.IGNORECASE)
            if m and m.group(1).lower() in _JOURNAL_DOI_PREFIXES:
                return True
        return False
    except Exception:
        return False


def _is_truncated(s: str) -> bool:
    """True if a URL/DOI was cut off mid-word by a line break."""
    return s.endswith(("-", "/", ".", "_", "="))


def _join_broken_urls(text: str) -> str:
    """Join URLs split across lines by PDF word-wrap: 'https://example.\ncom/path'."""
    return re.sub(
        r'(https?://[^\s\n]+[./\-_=])\n[ \t]*([a-zA-Z0-9][^\s\n]*)',
        r'\1\2',
        text,
    )


def _repo_hint_from_url(url: str) -> Optional[str]:
    u = url.lower()
    if "pangaea" in u:               return "pangaea"
    if "zenodo" in u:                return "zenodo"
    if "arcticdata.io" in u:         return "arctic_data_center"
    if "cds.climate" in u or ("copernicus" in u and "marine" not in u):
                                     return "copernicus_cds"
    if "marine.copernicus" in u:     return "cmems"
    if "ncei.noaa" in u:             return "noaa_ncei"
    if "earthdata.nasa.gov" in u:    return "nasa_earthdata"
    if "nsidc" in u:                 return "nsidc"
    if "noaa.gov" in u:              return "noaa"
    if "figshare.com" in u:          return "figshare"
    if "datadryad.org" in u:         return "dryad"
    if "dataverse" in u:             return "dataverse"
    if "osf.io" in u:                return "osf"
    if "bco-dmo.org" in u:           return "bco_dmo"
    if "seanoe.org" in u:            return "seanoe"
    if "data.mendeley.com" in u:     return "mendeley"
    if "bodc.ac.uk" in u:            return "bodc"
    if "ornl.gov" in u:              return "ornl_daac"
    if "usgs.gov" in u and "data" in u: return "usgs"
    if "ncdc.noaa" in u:             return "noaa_ncei"
    if "gfz-potsdam.de" in u:        return "gfz"
    if "polardata.ca" in u:          return "polar_data_catalogue"
    if "ogsl.ca" in u or "slgo.ca" in u: return "ogsl"
    # doi.org resolver: check if the embedded DOI belongs to a known data prefix
    if "doi.org" in u:
        m = re.search(r'doi\.org/(10\.\d{4,9})', u)
        if m:
            prefix = m.group(1).lower()
            if prefix in ("10.1594", "10.5441"): return "pangaea"
            if prefix == "10.5281":              return "zenodo"
            if prefix == "10.18739":             return "arctic_data_center"
            if prefix in ("10.5067",):           return "nasa_earthdata"
            if prefix in _DATA_REPO_DOI_PREFIXES: return "data_repository"
    return None


# Matches phrases that signal a data access URL follows immediately after
_DATA_CONTEXT_RE = re.compile(
    r'(?:available\s+(?:at|from|via|through|online\s+at)|'
    r'downloaded?\s+from|obtained?\s+from|accessed?\s+(?:at|from|via|through)|'
    r'retrieved?\s+from|deposit(?:ed)?\s+(?:at|in|to)|archived?\s+(?:at|in)|'
    r'data\s+(?:are\s+)?(?:available|accessible|hosted)\s+(?:at|from|via)|'
    # Merged forms produced by two-column PDF text extraction (no spaces between words)
    r'availableat|availableon|availablefrom|depositedat|areavailableat|areavailableon)',
    re.IGNORECASE
)

_USE_CONTEXT_RE = re.compile(
    r'\b(?:we\s+(?:use|used|utilize|utilised)|this\s+(?:study|paper)\s+(?:uses|used)|'
    r'data\s+(?:were|was)\s+(?:obtained|downloaded|retrieved|accessed)|available\s+(?:at|from|via)|'
    r'downloaded?\s+from|retrieved?\s+from|accessed?\s+(?:at|from|via)|'
    r'collected\s+by|deposited\s+at|provided\s+by|released\s+by|generated\s+by)\b|'
    # Merged forms from two-column PDF extraction
    r'(?:alldataareavailab|dataareavailab|availableat|availableon|availablefrom|depositedat)',
    re.IGNORECASE
)

_SECTION_HEADING_RE = re.compile(
    r'(?m)^[ \t]*'
    r'(?:'
    # 1. Numbered heading: "4 Data availability", "3.1 Methods", "2 Data and methods"
    #    Section number + horizontal space (not newline) + short content with data/method.
    r'\d[\d\.]*\.?[ \t]+(?=[^\n]{1,60}[ \t]*$)[^\n]*\b(?:data|method)\b[^\n]*'
    r'|'
    # 2. Star/bullet heading: "* Data availability", "• Data Access"
    r'[*•\-][ \t]+(?=[^\n]{1,60}[ \t]*$)[^\n]*\b(?:data|method)\b[^\n]*'
    r'|'
    # 3. Unnumbered heading starting WITH "Data"/"Method": ≤ 40 chars, 2+ words.
    #    Requires at least one more word after "data"/"method" (so bare "data" or
    #    "dataset" alone on a line does not match — those are sentence fragments).
    #    Verb exclusion (are/were/is/was) blocks body sentences like "data are...".
    #    Covers: "Data availability", "DATA AVAILABILITY", "Methods", "Data Access"
    r'(?=[^\n]{6,40}[ \t]*$)(?:data|method)\b(?!\s+(?:are|were|is|was|have|has|had)\b)\s+\S[^\n]*'
    r'|'
    # 4. Merged-word artefact from two-column PDF encoding (no spaces between words).
    #    pdfplumber returns "Dataavailability", "Datacollected", "Dataoverview" as one token.
    #    Minimum 4 letters after "data" to exclude the common word "dataset" (3 letters).
    #    Optional number/bullet prefix: "4 Dataavailability", "* Datacollected"
    r'(?:\d[\d\.]*\.?[ \t]+|[*•\-][ \t]+)?data[A-Za-z]{5,}'
    r')[ \t]*$',
    re.IGNORECASE
)
_SECTION_END_RE = re.compile(
    r'(?m)^(?:\d+\.?\s*)?(References|Bibliography|Acknowledgements|Acknowledgments|Supplementary|Appendix|Funding)\b',
    re.IGNORECASE
)

def _section_ranges(text: str) -> list[tuple[int, int]]:
    starts = sorted(m.start() for m in _SECTION_HEADING_RE.finditer(text))
    hard_ends = sorted(m.start() for m in _SECTION_END_RE.finditer(text))
    # Drop any section heading that starts after the first hard end marker
    # (catches citation years like "1958" or dataset names in the reference list).
    if hard_ends:
        starts = [s for s in starts if s < hard_ends[0]]
    # Each section ends at the next section heading OR a hard end (References/etc.),
    # whichever comes first — prevents one section from swallowing the entire paper.
    all_ends = sorted(starts[1:] + hard_ends)
    ranges: list[tuple[int, int]] = []
    for start in starts:
        end = next((e for e in all_ends if e > start), len(text))
        ranges.append((start, end))
    return ranges


def _is_in_section(idx: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= idx < end for start, end in ranges)


def _has_use_context(text: str, start: int, end: int,
                     data_ranges: list[tuple[int, int]]) -> bool:
    if _is_in_section(start, data_ranges):
        return True
    window = text[max(0, start - 180): min(len(text), end + 180)]
    return bool(_USE_CONTEXT_RE.search(window))


# ---------------------------------------------------------------------------
# Pass 1: Regex extraction
# ---------------------------------------------------------------------------

def extract_regex(text: str) -> list[DatasetRef]:
    text = _join_broken_urls(text)
    refs: list[DatasetRef] = []
    seen_identifiers: dict[str, int] = {}  # key → index in refs
    data_ranges = _section_ranges(text)

    # Per-pattern counters for the terminal report
    _c = {
        "url_raw": 0, "url_truncated": 0, "url_journal": 0,
        "url_no_hint": 0, "url_context": 0, "url_kept": 0,
        "doi_raw": 0, "doi_truncated": 0, "doi_journal": 0, "doi_kept": 0,
        "pangaea": 0, "zenodo": 0, "nsidc": 0, "known": 0,
    }

    def _add(ref: DatasetRef, key: str):
        if key not in seen_identifiers:
            seen_identifiers[key] = len(refs)
            refs.append(ref)
        elif ref.used_in_study and not refs[seen_identifiers[key]].used_in_study:
            refs[seen_identifiers[key]].used_in_study = True

    # URLs — only data repository URLs
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:")
        _c["url_raw"] += 1
        if _is_truncated(url):
            _c["url_truncated"] += 1
            continue
        if _is_journal_url(url):
            _c["url_journal"] += 1
            continue
        hint = _repo_hint_from_url(url)
        if hint is None:
            pre = text[max(0, m.start() - 150):m.start()].lower()
            if _DATA_CONTEXT_RE.search(pre):
                hint = "other"
                _c["url_context"] += 1
            else:
                _c["url_no_hint"] += 1
                continue
        _c["url_kept"] += 1
        ctx = text[max(0, m.start()-80):m.end()+80].replace("\n", " ")
        use_in_study = _has_use_context(text, m.start(), m.end(), data_ranges)
        _add(DatasetRef(
            name=url, url=url, repository_hint=hint,
            raw_citation=ctx, source="regex", used_in_study=use_in_study
        ), url)

    # DOIs — only data DOIs
    for m in _DOI_RE.finditer(text):
        raw_doi = m.group(1).rstrip(".,;:")
        doi = re.sub(r'^https?://doi\.org/', '', raw_doi, flags=re.IGNORECASE)
        _c["doi_raw"] += 1
        if _is_truncated(doi):
            _c["doi_truncated"] += 1
            continue
        if not _is_data_doi(doi):
            _c["doi_journal"] += 1
            continue
        _c["doi_kept"] += 1
        ctx = text[max(0, m.start()-80):m.end()+80].replace("\n", " ")
        hint = None
        if "1594/PANGAEA" in doi.upper():
            hint = "pangaea"
        elif "5281/zenodo" in doi.lower():
            hint = "zenodo"
        use_in_study = _has_use_context(text, m.start(), m.end(), data_ranges)
        _add(DatasetRef(
            name=doi, doi=doi, repository_hint=hint,
            raw_citation=ctx, source="regex", used_in_study=use_in_study
        ), doi.lower())

    # PANGAEA accessions
    for m in _PANGAEA_RE.finditer(text):
        _c["pangaea"] += 1
        acc = f"PANGAEA.{m.group(1)}"
        ctx = text[max(0, m.start()-80):m.end()+80].replace("\n", " ")
        use_in_study = _has_use_context(text, m.start(), m.end(), data_ranges)
        _add(DatasetRef(
            name=acc, accession=acc, repository_hint="pangaea",
            raw_citation=ctx, source="regex", used_in_study=use_in_study
        ), acc)

    # Zenodo records
    for m in _ZENODO_RE.finditer(text):
        _c["zenodo"] += 1
        acc = f"zenodo.{m.group(1)}"
        ctx = text[max(0, m.start()-80):m.end()+80].replace("\n", " ")
        use_in_study = _has_use_context(text, m.start(), m.end(), data_ranges)
        _add(DatasetRef(
            name=acc, accession=acc,
            url=f"https://zenodo.org/record/{m.group(1)}",
            repository_hint="zenodo",
            raw_citation=ctx, source="regex", used_in_study=use_in_study
        ), acc)

    # NSIDC accessions
    for m in _NSIDC_RE.finditer(text):
        _c["nsidc"] += 1
        acc = f"NSIDC-{m.group(1).zfill(4)}"
        ctx = text[max(0, m.start()-80):m.end()+80].replace("\n", " ")
        use_in_study = _has_use_context(text, m.start(), m.end(), data_ranges)
        _add(DatasetRef(
            name=acc, accession=acc, repository_hint="nsidc",
            raw_citation=ctx, source="regex", used_in_study=use_in_study
        ), acc)

    # Known named datasets
    for m in _KNOWN_PATTERN.finditer(text):
        _c["known"] += 1
        name = m.group(0)
        ctx = text[max(0, m.start()-100):m.end()+100].replace("\n", " ")
        use_in_study = _has_use_context(text, m.start(), m.end(), data_ranges)
        _add(DatasetRef(
            name=name, repository_hint=_hint_for_known(name),
            raw_citation=ctx, source="regex", used_in_study=use_in_study
        ), name.lower())

    # ── Terminal report ──────────────────────────────────────────────────────
    W = 60
    print(f"\n{'─' * W}")
    print(f"  PASS 1 — REGEX")
    print(f"{'─' * W}")
    print(f"  URLs      scanned {_c['url_raw']:>4}  │  "
          f"truncated {_c['url_truncated']:>3}  journal {_c['url_journal']:>3}  "
          f"no-hint {_c['url_no_hint']:>3}  ctx-saved {_c['url_context']:>3}  │  kept {_c['url_kept']:>3}")
    print(f"  DOIs      scanned {_c['doi_raw']:>4}  │  "
          f"truncated {_c['doi_truncated']:>3}  journal {_c['doi_journal']:>3}"
          f"              │  kept {_c['doi_kept']:>3}")
    print(f"  PANGAEA accessions : {_c['pangaea']:>3}")
    print(f"  Zenodo   records   : {_c['zenodo']:>3}")
    print(f"  NSIDC    accs      : {_c['nsidc']:>3}")
    print(f"  Known dataset names: {_c['known']:>3}")
    print(f"  {'─' * (W-2)}")
    print(f"  Regex refs (unique): {len(refs):>3}")
    print(f"{'─' * W}")

    return refs


def _hint_for_known(name: str) -> Optional[str]:
    n = name.upper()
    if n.startswith("ERA") or n in {"CAMSRA"}:
        return "copernicus_cds"
    if n in {"MODIS", "VIIRS", "LANDSAT", "ICESAT", "ICESAT-2", "GRACE", "GRACE-FO",
             "MERRA-2", "CALIPSO", "CLOUDSAT", "GPM", "IMERG", "TRMM", "SMOS"}:
        return "nasa_earthdata"
    if n in {"AMSR-E", "AMSR2", "SSMIS", "SSM/I", "NSIDC SEA ICE INDEX", "IABP"}:
        return "nsidc"
    if n in {"NCEP", "NCEP/NCAR", "CFSR", "CFSV2", "GPCP", "CMORPH", "PERSIANN"}:
        return "noaa"
    if n in {"GLORYS", "GLORYS12", "OSTIA", "OSI-SAF"}:
        return "cmems"
    if n in {"CMIP3", "CMIP5", "CMIP6"}:
        return "other"
    return None


# ---------------------------------------------------------------------------
# Pass 2: LLM extraction — tries Ollama first, falls back to OpenAI
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a scientific data archivist specialising in arctic and environmental research.

Your task: given a chunk of text from a research paper, extract EVERY dataset that is mentioned —
including datasets that have no explicit URL or DOI but are referred to by name or description.

For EACH dataset return a JSON object with these fields (omit fields you cannot determine):
  - name        : string  — canonical dataset name
  - url         : string  — direct download or landing page URL, if present
  - doi         : string  — data DOI (10.XXXX/...), if present
  - accession   : string  — repository accession number, if present
  - repository_hint : string — one of: pangaea, zenodo, nsidc, copernicus_cds, nasa_earthdata,
                               noaa, arctic_data_center, dryad, figshare, other
  - raw_citation : string — the sentence(s) in the text that mention this dataset (verbatim, max 200 chars)
  - is_primary  : boolean — true ONLY for the single dataset that this paper is centrally about:
                  the dataset being introduced, collected, or the main observation record being
                  analyzed. Set false for all forcing inputs, reanalysis products, comparison
                  sources, validation references, and background data. If no single dataset is
                  clearly central, set false for all.

Rules:
- Include reanalysis products (ERA5, MERRA-2, JRA-55, NCEP), remote sensing products (MODIS, VIIRS,
  Sentinel, Landsat, ICESat-2, CryoSat-2, AMSR2), in-situ datasets, model outputs, and campaign data.
- Do NOT include journal papers, books, or software packages — only datasets.
- Do NOT duplicate: if the same dataset is mentioned twice, return it once.
- Return ONLY a JSON array. No prose, no markdown fences.
- If there are no datasets, return [].
"""

_CHUNK_SIZE = 3000
_OVERLAP = 400
_OLLAMA_URL = "http://localhost:11434/api/chat"
_OLLAMA_MODELS = ["mistral:latest", "mistral:7b", "llama3:latest", "qwen2.5:7b", "gemma3:12b"]

# Signals that a chunk is worth sending to the LLM
_DATASET_SIGNAL_RE = re.compile(
    r'\b(?:'
    r'dataset|data\s+availability|data\s+available|code\s+and\s+data|'
    r'downloaded?\s+from|obtained?\s+from|accessed?\s+(?:at|from)|'
    r'accession|repository|zenodo|pangaea|figshare|dryad|dataverse|osf\.io|'
    r'doi\.org|10\.\d{4}/'
    r'|ERA5|ERA-Interim|MODIS|CryoSat|Sentinel|ICESat|GRACE|MERRA|CMIP|'
    r'TOPAZ|PIOMAS|AMSR|HYCOM|FESOM|GLORYS|OSTIA|GPM|IMERG|CALIPSO|'
    r'reanalysis|remote\s+sensing|satellite\s+data|in\s+situ|buoy|CTD|'
    r'mooring|observat(?:ion|ory)|measurement|field\s+campaign'
    r')\b',
    re.IGNORECASE
)


def _has_dataset_signals(chunk: str) -> bool:
    return bool(_DATASET_SIGNAL_RE.search(chunk))


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += _CHUNK_SIZE - _OVERLAP
    return chunks


def _detect_ollama_model() -> Optional[str]:
    """Return the first available Ollama model, or None if Ollama is not running."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            return None
        available = {m["name"] for m in resp.json().get("models", [])}
        for m in _OLLAMA_MODELS:
            if m in available:
                return m
        # If any model is present use the first one
        if available:
            return next(iter(available))
    except Exception:
        pass
    return None


def _call_ollama(model: str, chunk: str) -> tuple[list[dict], dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(_OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    content = body["message"]["content"].strip()
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    usage = {
        "input_tokens":  body.get("prompt_eval_count", 0),
        "output_tokens": body.get("eval_count", 0),
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return json.loads(content), usage


def _call_azure_openai(chunk: str) -> tuple[list[dict], dict]:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        temperature=0,
        max_tokens=2048,
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    u = response.usage
    usage = {
        "input_tokens":  u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens":  u.total_tokens if u else 0,
    }
    return json.loads(content), usage


def _call_openai(chunk: str, api_key: str) -> tuple[list[dict], dict]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        temperature=0,
        max_tokens=2048,
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    u = response.usage
    usage = {
        "input_tokens":  u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens":  u.total_tokens if u else 0,
    }
    return json.loads(content), usage


def extract_llm(text: str, api_key: Optional[str] = None) -> list[DatasetRef]:
    # Priority: Azure OpenAI → regular OpenAI → Ollama (local fallback)
    azure_ready = bool(os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    if azure_ready:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
        backend_label = f"Azure OpenAI  ({deployment})"
        # gpt-4.1-mini pricing (per 1M tokens, as of 2025)
        cost_per_1m_input, cost_per_1m_output = 0.40, 1.60
        call_fn = _call_azure_openai
    elif openai_key:
        backend_label = "OpenAI  gpt-4o-mini"
        cost_per_1m_input, cost_per_1m_output = 0.15, 0.60
        call_fn = lambda chunk: _call_openai(chunk, openai_key)
    else:
        ollama_model = _detect_ollama_model()
        if ollama_model:
            backend_label = f"Ollama  ({ollama_model})  [local — no cost]"
            cost_per_1m_input, cost_per_1m_output = 0.0, 0.0
            call_fn = lambda chunk: _call_ollama(ollama_model, chunk)
        else:
            print("[extractor] No LLM backend available — skipping LLM pass")
            return []

    # Don't feed the reference list to the LLM — it causes mass over-extraction.
    hard_end = _SECTION_END_RE.search(text)
    llm_text = text[:hard_end.start()] if hard_end else text

    all_chunks = _chunk_text(llm_text)
    chunks_with_idx = [(i, c) for i, c in enumerate(all_chunks) if _has_dataset_signals(c)]
    skipped = len(all_chunks) - len(chunks_with_idx)
    n = len(chunks_with_idx)
    W = 60

    print(f"\n{'─' * W}")
    print(f"  PASS 2 — LLM")
    print(f"{'─' * W}")
    print(f"  Backend     : {backend_label}")
    print(f"  Chunk size  : {_CHUNK_SIZE} chars  overlap: {_OVERLAP} chars")
    print(f"  Chunks      : {len(all_chunks)} total  │  {skipped} skipped (no signals)  │  {n} sent to LLM")
    print(f"  {'─' * (W-2)}")
    print(f"  {'Chunk':<13} {'Time':>6}  {'Datasets':>9}  {'In tok':>7}  {'Out tok':>8}  {'Status'}")
    print(f"  {'─' * (W-2)}")

    refs: list[DatasetRef] = []
    seen_names: set[str] = set()

    total_input_tok = 0
    total_output_tok = 0
    succeeded = 0
    errors = 0
    llm_start = time.time()

    for seq, (i, chunk) in enumerate(chunks_with_idx):
        t0 = time.time()
        status = "ok"
        chunk_datasets = 0
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            items, usage = call_fn(chunk)
            for item in items:
                name = item.get("name", "").strip()
                if not name or name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())
                raw_primary = item.get("is_primary", False)
                is_primary = (
                    raw_primary is True
                    or (isinstance(raw_primary, str) and raw_primary.lower() == "true")
                )
                refs.append(DatasetRef(
                    name=name,
                    url=item.get("url"),
                    doi=item.get("doi"),
                    accession=item.get("accession"),
                    repository_hint=item.get("repository_hint"),
                    raw_citation=item.get("raw_citation", "")[:300],
                    source="llm",
                    is_primary=is_primary,
                    used_in_study=is_primary,
                ))
                chunk_datasets += 1
            succeeded += 1
        except json.JSONDecodeError:
            status = "JSON-err"
            errors += 1
        except Exception as e:
            status = f"ERR:{str(e)[:18]}"
            errors += 1

        elapsed = time.time() - t0
        total_input_tok  += usage["input_tokens"]
        total_output_tok += usage["output_tokens"]
        tok_in  = usage["input_tokens"]  or "—"
        tok_out = usage["output_tokens"] or "—"
        print(f"  {seq+1:>3}/{n:<4} [c{i+1:>2}]  {elapsed:>5.1f}s  {chunk_datasets:>9}  "
              f"{str(tok_in):>7}  {str(tok_out):>8}  {status}")

    total_time = time.time() - llm_start
    total_tok  = total_input_tok + total_output_tok
    cost_input  = (total_input_tok  / 1_000_000) * cost_per_1m_input
    cost_output = (total_output_tok / 1_000_000) * cost_per_1m_output
    cost_total  = cost_input + cost_output

    print(f"  {'─' * (W-2)}")
    print(f"  Calls       : {n} attempted  │  {succeeded} ok  │  {errors} errors  │  {skipped} skipped")
    avg = (total_time / n) if n else 0
    print(f"  Total time  : {total_time:.1f}s  ({avg:.1f}s avg per call)")
    if total_tok:
        print(f"  Tokens      : {total_tok:,} total  "
              f"(in {total_input_tok:,}  out {total_output_tok:,})")
        print(f"  Est. cost   : ${cost_total:.5f}  "
              f"(in ${cost_input:.5f}  out ${cost_output:.5f})")
    else:
        print(f"  Tokens      : not reported by backend")
        print(f"  Est. cost   : $0.00  (local model)")
    print(f"  LLM refs    : {len(refs)} (unique new names)")
    print(f"{'─' * W}")

    return refs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_from_pdf(pdf_path: str, use_llm: bool = True,
                     api_key: Optional[str] = None) -> list[DatasetRef]:
    """
    Extract all dataset references from a PDF file.
    Returns raw (pre-dedup) list of DatasetRef objects.
    """
    fname = os.path.basename(pdf_path)
    W = 60
    print(f"\n{'═' * W}")
    print(f"  EXTRACTOR  —  {fname}")
    print(f"{'═' * W}")

    text = _read_pdf(pdf_path)
    if not text:
        print(f"  ERROR: could not read text from {pdf_path}")
        return []
    print(f"  PDF size  : {len(text):,} chars")

    t0 = time.time()
    regex_refs = extract_regex(text)

    llm_refs = []
    if use_llm:
        llm_refs = extract_llm(text, api_key=api_key)

    total = regex_refs + llm_refs
    print(f"\n{'═' * W}")
    print(f"  TOTALS  —  {fname}")
    print(f"  Regex refs : {len(regex_refs):>4}")
    print(f"  LLM   refs : {len(llm_refs):>4}")
    print(f"  Combined   : {len(total):>4}  raw refs  (before dedup)")
    print(f"  Wall time  : {time.time()-t0:.1f}s")
    print(f"{'═' * W}\n")

    return total


def extract_from_pdfs(pdf_paths: list[str], use_llm: bool = True,
                      api_key: Optional[str] = None) -> dict[str, list[DatasetRef]]:
    """Extract from multiple PDFs. Returns {filename: [DatasetRef]}."""
    results = {}
    for path in pdf_paths:
        fname = os.path.basename(path)
        print(f"\n[extractor] Processing: {fname}")
        results[fname] = extract_from_pdf(path, use_llm=use_llm, api_key=api_key)
    return results


def _words_to_text(words: list) -> str:
    """Reconstruct text from pdfplumber word dicts, one visual line per output line."""
    if not words:
        return ""
    sorted_w = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[str] = []
    cur_top: Optional[float] = None
    cur_line: list[dict] = []
    for w in sorted_w:
        if cur_top is None or (w["top"] - cur_top) > 5:
            if cur_line:
                lines.append(" ".join(x["text"] for x in cur_line))
            cur_line = [w]
            cur_top = w["top"]
        else:
            cur_line.append(w)
    if cur_line:
        lines.append(" ".join(x["text"] for x in cur_line))
    return "\n".join(lines)


def _extract_page_text(page) -> str:
    """
    Column-aware page extraction.
    Splits the page at its horizontal midpoint: full-width elements (page
    headers/numbers) come first, then the left column top-to-bottom, then
    the right column top-to-bottom.  This keeps section headings like
    "Data availability" on their own line regardless of what the right
    column contains on the same visual row.
    """
    words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    if not words:
        return page.extract_text() or ""

    mid = page.width / 2
    left  = [w for w in words if w["x1"] < mid - 10]
    right = [w for w in words if w["x0"] > mid + 10]
    span  = [w for w in words if not (w["x1"] < mid - 10 or w["x0"] > mid + 10)]

    # Two-column: both sides have content and together account for ≥60% of words
    two_col = (
        len(left) >= 5
        and len(right) >= 5
        and (len(left) + len(right)) >= len(words) * 0.6
    )

    if two_col:
        parts = [_words_to_text(span), _words_to_text(left), _words_to_text(right)]
        return "\n".join(p for p in parts if p)
    return _words_to_text(words)


def _read_pdf(path: str) -> str:
    """
    Column-aware PDF text extraction.
    Two-column pages are extracted with each column processed independently
    so section headings are never interleaved with unrelated text.
    """
    try:
        with pdfplumber.open(path) as pdf:
            pages = [_extract_page_text(p) for p in pdf.pages]
            return "\n".join(p for p in pages if p)
    except Exception as e:
        print(f"[extractor] PDF read error: {e}")
        return ""
