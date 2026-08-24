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
    "Sentinel-1", "Sentinel-2", "Sentinel-3", "Sentinel-5P", "Sentinel-6",
    "AMSR-E", "AMSR2", "SSMIS", "SSM/I",
    "ICESat", "ICESat-2", "CryoSat-2",
    "GRACE", "GRACE-FO",
    "PIOMAS", "TOPAZ", "TOPAZ4",
    "NCEP", "NCEP/NCAR", "CFSR", "CFSv2",
    "JRA-55", "MERRA-2", "MERRA",
    "AVHRR", "GOES",
    "NSIDC Sea Ice Index",
    "HadGHCND", "HadSST", "HadCRUT", "HadISST",
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
    "GLORYS", "GLORYS12", "OSTIA", "ORAS5", "ECCO", "EN4",
    # Sea-ice satellite products
    "OSI-SAF", "SMOS", "GlobSnow",
    # Precipitation datasets
    "GPM", "IMERG", "GPCP", "TRMM", "CMORPH", "PERSIANN", "PERSIANN-CDR",
    "MSWEP", "GPCC",
    # Atmospheric / climate
    "CALIPSO", "CloudSat", "CAMSRA", "TROPOMI", "AERONET",
    # Climate model families
    "AWI-CM", "MPI-ESM", "CESM", "CESM2", "E3SM",
    # Terrain / elevation datasets
    "ArcticDEM", "REMA", "SRTM", "TanDEM-X", "ASTER GDEM",
    # Snow / permafrost observation networks
    "SNOTEL", "GTN-P",
    # Flux tower networks
    "FLUXNET", "FLUXNET2015", "AmeriFlux", "ICOS",
    # Ocean biogeochemistry observation systems
    "SOCAT", "GLODAP", "ICOADS", "PSMSL",
    # Climate observation datasets
    "GHCND", "GHCNM", "GHCN-D", "GHCN-M",
    "CRU TS", "CHELSA", "DAYMET", "WorldClim",
    # Modern satellite missions
    "SMAP", "SWOT", "GEDI",
    # Radiosonde / upper-air
    "IGRA",
    # Sea surface temperature reconstructions
    "ERSST", "ICOADS",
    # Ocean altimetry satellites
    "Jason-1", "Jason-2", "Jason-3", "TOPEX/Poseidon",
    # World Ocean Database
    "WOD", "World Ocean Database",
    # NCAR Research Data Archive
    "NCAR-RDA",
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
    "10.1371",  # PLOS ONE / PLOS journals
    "10.7554",  # eLife
    "10.1128",  # ASM journals (mSystems, etc.)
    "10.1890",  # Ecological Society of America
    "10.1002",  # Wiley (already present but adding alias)
    "10.1139",  # NRC Research Press (Arctic journal)
    "10.14430", # SCAR / Antarctic Science
    "10.1657",  # Arctic, Antarctic, and Alpine Research
    "10.3354",  # Inter-Research (Mar Ecol Prog Ser, etc.)
    "10.1007",  # Springer (already present)
    "10.3402",  # Polar Research (Norwegian journal)
    "10.1080",  # Taylor & Francis
    "10.1163",  # Brill journals
    "10.4319",  # Limnology and Oceanography
    "10.1899",  # Journal of the North American Benthological Society
    # Additional journal prefixes found to slip through
    "10.2136",  # ASA/CSSA/SSSA journals (Vadose Zone Journal, etc.)
    "10.21105", # Journal of Open Source Software (JOSS)
    "10.1641",  # BioScience (AIBS)
    "10.1023",  # Springer/Kluwer legacy prefix
    "10.1046",  # Blackwell Publishing legacy
    "10.1101",  # bioRxiv / medRxiv preprints
    "10.2478",  # Sciendo / De Gruyter journals
    "10.4141",  # Canadian Journal of Soil/Plant Science (NRC)
    "10.2166",  # IWA Publishing (water/hydrology journals)
    "10.1002",  # Wiley — broad Wiley journals
    "10.2307",  # JSTOR
    "10.7717",  # PeerJ
    "10.7150",  # Ivyspring (Theranostics, etc.)
    "10.3390",  # MDPI journals
    "10.9753",  # ICCE (International Conference on Coastal Engineering)
    "10.1061",  # ASCE (American Society of Civil Engineers)
    "10.2151",  # Journal of the Meteorological Society of Japan
    "10.12952", # Elementa: Science of the Anthropocene (pre-AGU prefix)
    "10.48550", # arXiv preprints
    "10.1103",  # American Physical Society (Phys. Rev. Lett., etc.)
    "10.31223", # EarthArXiv preprints
    "10.2314",  # K10plus/GVK catalog (tech reports, not datasets)
}

# DOI prefixes that are definitely data repositories — always keep these
_DATA_REPO_DOI_PREFIXES = {
    "10.1594",  # PANGAEA
    "10.5281",  # Zenodo
    "10.7289",  # NOAA NCEI
    "10.5067",  # NASA EOSDIS / NSIDC
    "10.18739", # Arctic Data Center
    "10.6073",  # BCO-DMO (Biological and Chemical Oceanography Data Management)
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
    # Expanded data repository coverage
    "10.5065",  # NCAR/UCAR Research Data Archive (rda.ucar.edu)
    "10.17190", # AmeriFlux network flux tower data
    "10.17632", # Mendeley Data
    "10.7910",  # Harvard Dataverse
    "10.17605", # OSF (Open Science Framework)
    "10.22033", # EarthChem / IEDA (Interdisciplinary Earth Data Alliance)
    "10.15468", # GBIF (Global Biodiversity Information Facility)
    "10.17026", # DANS/EASY (Netherlands)
    "10.5255",  # UK Data Service (UKDS)
    "10.5256",  # CEDA / BADC (Centre for Environmental Data Analysis, UK)
    "10.11578", # DOE Office of Scientific & Technical Information (OSTI)
    "10.18160", # ICOS (Integrated Carbon Observation System)
    "10.34770", # Princeton DataSpace
    "10.25914", # TERN (Terrestrial Ecosystem Research Network, Australia)
    "10.26022", # IMOS (Integrated Marine Observing System, Australia)
    "10.21420", # SND (Swedish National Data Service)
    "10.7488",  # Edinburgh DataShare (University of Edinburgh)
    "10.6075",  # KNB Data Repository (Knowledge Network for Biocomplexity)
    "10.15485", # EUDAT B2SHARE
    "10.21233", # LifeWatch ERIC
    "10.20350", # IEO (Instituto Español de Oceanografía)
    "10.12770", # SeaDataNet (pan-European marine data)
    "10.25504", # IGSN (International Geo Sample Number registry)
    "10.26181", # Monash University Research Repository
    "10.48350", # University of Edinburgh Research Data Repository
}

# LLM false-positive filter: terms that indicate the name is NOT a dataset
_NOT_DATASET_TERMS = re.compile(
    r'\b(?:'
    # Lab instruments / equipment
    r'analyzer|analyser|instrument|sensor|detector|spectrometer|chromatograph|'
    r'centrifuge|incubator|sampler|probe|logger|thermometer|'
    # Lab consumables / brand names
    r'Whatman|Millipore|Sigma.Aldrich|Thermo\s*Fisher|Merck|GF/[A-Z]|filter\s+paper|'
    r'Shimadzu|Agilent|Waters|PerkinElmer|Varian|Bruker|'
    # Journal/publisher names
    r'biogeosciences|geophysical research letters|nature climate|science advances|'
    r'global biogeochemical|journal of geophysical|limnology and oceanography|'
    r'environmental science|water resources research|geochemistry geophysics|'
    r'copernicus publications|european geosciences union'
    r')\b',
    re.IGNORECASE
)

# Reject figure/table/supplementary cross-references
_FIGURE_TABLE_RE = re.compile(
    r'^\s*(?:Fig\.?|Figure|Table|Tbl\.?|Suppl?\.?|Supplementary|Appendix|Supp\s*Table|Supp\s*Fig)\s*[\dSA-Z]',
    re.IGNORECASE
)

# Reject author citation patterns: "Mann et al., 2012" / "Bradlow et al. (2002)"
_CITATION_RE = re.compile(r'\bet\s+al\.?\b', re.IGNORECASE)

# Reject bibliographic citation format: "Journal Name, 18, 1215–1239, 2024"
_JOURNAL_CITATION_RE = re.compile(
    r'\b\d{1,4}\s*,\s*\d{1,6}\s*[–\-]\s*\d{1,6}\s*,\s*(?:19|20)\d{2}\b'
)

# Reject funding acknowledgement strings
_FUNDING_RE = re.compile(
    r'\b(?:NWO|NSF|NSERC|NIH|ERC|DFG|NERC|ARC\b|venigrant|grant\s+#|award\s+#|'
    r'dutch\s+nwo|permafrost\s+carbon\s+network|great\s+rivers\s+observatory)\b',
    re.IGNORECASE
)

_NOT_DATASET_EXACT = {
    # Bare measurement abbreviations with no context
    "doc", "dic", "toc", "poc", "tdn", "rdoc", "dom",
    "measurements", "samples", "data", "results", "analysis",
    "soil literature", "aquatic literature",
    # Data-table row/column labels (statistics-table artifact, seen in tc-18-1259-2024:
    # a table comparing lead-fraction (LF) products by mean/stddev/coeff-of-variation)
    "lfdiv", "lfmodis", "lfheli_tir", "lfpmw", "lfcs2", "lflkf",
    "lflkffraction", "lflkfpixel", "lfclassified_sar",
    "classified_sar", "heli_tir", "pmw", "c2absolute", "sar image",
    "div lf", "div lkf", "lf with div lkf",
}

# Reject LLM names that look like data-table cells rather than natural-language
# dataset names -- e.g. "classified_SAR" (underscore), "accu.div" (dotted lowercase
# compound), "LF5×accu.div" (multiplication sign). These show up when an LLM
# chunk lands on a validation/statistics table and reads row labels as dataset names.
_TABLE_CELL_RE = re.compile(r'_|[×✕]|^[a-z]+\.[a-z]+$')


def _is_likely_dataset(name: str) -> bool:
    """Return False if the name looks like lab equipment, a citation, figure ref, or bare measurement."""
    n = name.strip()
    if not n or len(n) < 3:
        return False
    if n.lower() in _NOT_DATASET_EXACT:
        return False
    if _NOT_DATASET_TERMS.search(n):
        return False
    if _FIGURE_TABLE_RE.match(n):
        return False
    if _CITATION_RE.search(n):
        return False
    if _JOURNAL_CITATION_RE.search(n):
        return False
    if _FUNDING_RE.search(n):
        return False
    if _TABLE_CELL_RE.search(n):
        return False
    # Reject suspiciously long names (> 250 chars) — usually multi-grant strings
    if len(n) > 250:
        return False
    return True


# URLs from sites that are never data repositories
_JOURNAL_URL_DOMAINS = {
    "journals.ametsoc.org", "pnas.org", "nature.com", "science.org",
    "sciencedirect.com", "springer.com", "wiley.com", "tandfonline.com",
    "cambridge.org", "agu.org", "essoar.org", "arxiv.org",
    "researchgate.net", "semanticscholar.org", "carbonbrief.org",
    "newengineer.com", "gsas.harvard.edu", "un.org", "usembassy.gov",
    "wwf.org", "wwf.org.uk", "environment.nsw.gov.au",
    "greatwhitecon.info",
    # Additional journal/publisher domains
    "agupubs.onlinelibrary.wiley.com",  # AGU journals on Wiley platform
    "journals.agu.org",                  # AGU legacy journal site
    "royalsocietypublishing.org",        # Royal Society journals
    "rmets.onlinelibrary.wiley.com",     # Royal Meteorological Society
    "link.springer.com",                 # Springer journal articles
    "academic.oup.com",                  # Oxford University Press journals
    "iopscience.iop.org",                # IOP Publishing
    "mdpi.com",                          # MDPI open-access journals
    "frontiersin.org",                   # Frontiers journals
    "pubs.acs.org",                      # American Chemical Society
    "pubs.rsc.org",                      # Royal Society of Chemistry
    "ascelibrary.org",                   # ASCE journals
    "esajournals.onlinelibrary.wiley.com", # ESA journals (Ecology, etc.)
    "brill.com",                         # Brill academic journals
    "degruyter.com",                     # De Gruyter / Sciendo journals
    "joss.theoj.org",                    # Journal of Open Source Software
    "journals.plos.org",                 # PLOS journals
    "elifesciences.org",                 # eLife journal
    "bioone.org",                        # BioOne journals
    "ingentaconnect.com",                # Ingenta aggregator
    "jstor.org",                         # JSTOR journal archive
    "annals.org",                        # Annals of Internal Medicine
    "bmj.com",                           # British Medical Journal
}


# ── Online DOI classification (authoritative; replaces prefix guessing) ───────
# A DOI's *registration agency* is a strong signal: DataCite registers
# data/software, Crossref registers articles/books/proceedings. We only consult
# the network for DOIs whose prefix is in NEITHER hardcoded list (the ones that
# would otherwise be blindly kept). Known prefixes stay instant and offline.
# Results are cached to disk so each DOI is looked up at most once, ever.
# Set HARVESTER_DOI_ONLINE=0 to force fully-offline behaviour.
_DOI_ONLINE_CHECK = os.getenv("HARVESTER_DOI_ONLINE", "1") != "0"
_DOI_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".doi_class_cache.json")
_doi_class_cache: Optional[dict] = None

_DATACITE_DATA_TYPES = {"dataset", "collection", "software"}
_DATACITE_TEXT_TYPES = {"text", "journalarticle", "conferencepaper", "preprint",
                        "book", "bookchapter", "dissertation", "report", "peerreview"}


def _load_doi_cache() -> dict:
    global _doi_class_cache
    if _doi_class_cache is None:
        try:
            with open(_DOI_CACHE_PATH, encoding="utf-8") as f:
                _doi_class_cache = json.load(f)
        except Exception:
            _doi_class_cache = {}
    return _doi_class_cache


def _classify_doi_online(doi: str) -> str:
    """Ask the DOI registrar what this DOI actually is.

    Returns 'data', 'journal', or 'unknown' (network failure / inconclusive).
    Cached to disk. Never raises.
    """
    doi = doi.lower().strip()
    cache = _load_doi_cache()
    if doi in cache:
        return cache[doi]

    result = "unknown"
    try:
        ra = requests.get(f"https://doi.org/doiRA/{doi}", timeout=6).json()
        agency = ""
        if isinstance(ra, list) and ra and isinstance(ra[0], dict):
            agency = str(ra[0].get("RA", "")).lower()
        if agency == "datacite":
            d = requests.get(f"https://api.datacite.org/dois/{doi}", timeout=6).json()
            rt = str(((d.get("data", {}) or {}).get("attributes", {}) or {})
                     .get("types", {}).get("resourceTypeGeneral", "")).lower()
            if rt in _DATACITE_DATA_TYPES:
                result = "data"
            elif rt in _DATACITE_TEXT_TYPES:
                result = "journal"
            else:
                result = "data"          # unlabelled DataCite record → lean data
        elif agency == "crossref":
            d = requests.get(f"https://api.crossref.org/works/{doi}", timeout=6).json()
            ctype = str((d.get("message", {}) or {}).get("type", "")).lower()
            result = "data" if ctype in ("dataset", "database") else "journal"
    except Exception:
        result = "unknown"

    cache[doi] = result
    try:
        with open(_DOI_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass
    return result


def _is_data_doi(doi: str) -> bool:
    prefix = doi.split("/")[0].lower()
    if prefix in _DATA_REPO_DOI_PREFIXES:
        return True            # explicitly known data repo — always keep, no network
    if prefix in _JOURNAL_DOI_PREFIXES:
        return False           # explicitly known journal — always reject, no network
    # Unknown prefix: the old code blindly kept it (source of journal false
    # positives like 10.1051). Ask the registrar instead.
    if _DOI_ONLINE_CHECK:
        cls = _classify_doi_online(doi)
        if cls == "data":
            return True
        if cls == "journal":
            return False
    return True                # offline / inconclusive → historical default (keep)


def _is_journal_url(url: str) -> bool:
    try:
        # Malformed URL: two protocols merged (e.g. "https://doi.org/10.5281/https://doi.org/...")
        if url.count("://") > 1:
            return True
        domain = url.split("//", 1)[1].split("/")[0].lower()
        domain = domain.lstrip("www.")
        if domain in _JOURNAL_URL_DOMAINS:
            return True
        # doi.org resolver pointing at a journal article — treat as journal URL
        if domain == "doi.org":
            mfull = re.search(r'doi\.org/(10\.\d{4,9}/[^\s\)\(\],\'"<>]+)', url, re.IGNORECASE)
            m = re.search(r'doi\.org/(10\.\d{4,9})', url, re.IGNORECASE)
            prefix = m.group(1).lower() if m else None
            if prefix in _JOURNAL_DOI_PREFIXES:
                return True
            if prefix in _DATA_REPO_DOI_PREFIXES:
                return False
            # Unknown prefix — ask the registrar (same logic as _is_data_doi)
            if prefix and _DOI_ONLINE_CHECK and mfull:
                if _classify_doi_online(mfull.group(1).rstrip('.,;:')) == "journal":
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
    if "marine.copernicus" in u or "copernicusmarine" in u: return "cmems"
    if "ncei.noaa" in u or "ncdc.noaa" in u: return "noaa_ncei"
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
    if "ornl.gov" in u or "daac.ornl" in u: return "ornl_daac"
    if "usgs.gov" in u and "data" in u: return "usgs"
    if "gfz-potsdam.de" in u:        return "gfz"
    if "polardata.ca" in u:          return "polar_data_catalogue"
    if "ogsl.ca" in u or "slgo.ca" in u: return "ogsl"
    # Flux tower networks
    if "ameriflux.lbl.gov" in u:     return "ameriflux"
    if "fluxnet" in u or "fluxdata.org" in u: return "fluxnet"
    if "icos-cp.eu" in u or "icos-ri.eu" in u: return "icos"
    # Atmospheric / met data
    if "arm.gov" in u:               return "arm"
    if "rda.ucar.edu" in u or "ncar.ucar.edu" in u: return "ncar_rda"
    if "toolik.alaska.edu" in u:     return "toolik_field_station"
    if "aeronet.gsfc.nasa.gov" in u: return "nasa_earthdata"
    # Ocean biogeochemistry
    if "socat.info" in u:            return "socat"
    if "glodap.info" in u:           return "glodap"
    if "bco-dmo.org" in u:           return "bco_dmo"
    if "seadatanet.org" in u:        return "seadatanet"
    if "emodnet.eu" in u:            return "emodnet"
    # Sea level / ocean monitoring
    if "psmsl.org" in u:             return "psmsl"
    if "aviso.altimetry.fr" in u:    return "aviso"
    if "marine.copernicus.eu" in u:  return "cmems"
    # Biodiversity
    if "gbif.org" in u:              return "gbif"
    if "obis.org" in u:              return "obis"
    # UK environmental data
    if "ceda.ac.uk" in u:            return "ceda"
    if "ceh.ac.uk" in u:             return "eidc"
    if "bas.ac.uk" in u:             return "bas"
    if "data.nerc.ac.uk" in u:       return "nerc"
    # Polar data centres
    if "data.aad.gov.au" in u:       return "aadc"
    if "tern.org.au" in u:           return "tern"
    if "imos.org.au" in u:           return "imos"
    # NASA portals
    if "podaac" in u:                return "nasa_earthdata"
    if "disc.gsfc.nasa.gov" in u or "gesdisc" in u: return "nasa_earthdata"
    if "lpdaac.usgs.gov" in u:       return "nasa_earthdata"
    if "search.earthdata.nasa.gov" in u: return "nasa_earthdata"
    if "oceancolor.gsfc.nasa.gov" in u: return "nasa_earthdata"
    # ESGF (CMIP / model output)
    if "esgf" in u:                  return "esgf"
    # Hydrology
    if "hydroshare.org" in u:        return "hydroshare"
    # Nordic / European met offices
    if "thredds.met.no" in u or "api.met.no" in u: return "met_norway"
    if "opendata.dwd.de" in u:       return "dwd"
    if "smhi.se" in u:               return "smhi"
    # Data registries
    if "registry.opendata.aws" in u: return "aws_open_data"
    if "portal.edirepository.org" in u or "lternet.edu" in u: return "edi_lter"
    # doi.org resolver: check if the embedded DOI belongs to a known data prefix
    if "doi.org" in u:
        m = re.search(r'doi\.org/(10\.\d{4,9})', u)
        if m:
            prefix = m.group(1).lower()
            if prefix in ("10.1594", "10.5441"): return "pangaea"
            if prefix == "10.5281":              return "zenodo"
            if prefix == "10.18739":             return "arctic_data_center"
            if prefix in ("10.5067", "10.7265"): return "nasa_earthdata"
            if prefix == "10.5065":              return "ncar_rda"
            if prefix == "10.17190":             return "ameriflux"
            if prefix == "10.17632":             return "mendeley"
            if prefix == "10.7910":              return "harvard_dataverse"
            if prefix == "10.22033":             return "earthchem"
            if prefix == "10.18160":             return "icos"
            if prefix == "10.6073":              return "bco_dmo"
            if prefix == "10.3334":              return "ornl_daac"
            if prefix == "10.7289":              return "noaa_ncei"
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
        # Normalize doi.org URLs to bare DOI so they dedup with the DOI pass below
        _doi_in_url = re.search(r'doi\.org/(10\.\d{4,9}/[^\s\)\(\],\'"<>]+)', url, re.IGNORECASE)
        dedup_key = _doi_in_url.group(1).lower() if _doi_in_url else url
        _add(DatasetRef(
            name=url, url=url, repository_hint=hint,
            raw_citation=ctx, source="regex", used_in_study=use_in_study
        ), dedup_key)

    # DOIs — only data DOIs
    for m in _DOI_RE.finditer(text):
        raw_doi = m.group(1).rstrip(".,;:")
        doi = re.sub(r'^https?://doi\.org/', '', raw_doi, flags=re.IGNORECASE)
        _c["doi_raw"] += 1
        # Reject malformed DOIs that swallowed an adjacent URL (e.g. 10.5281/https://doi.org/...)
        if "http" in doi or "://" in doi:
            _c["doi_journal"] += 1
            continue
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
        num = m.group(1)
        acc = f"PANGAEA.{num}"
        ctx = text[max(0, m.start()-80):m.end()+80].replace("\n", " ")
        use_in_study = _has_use_context(text, m.start(), m.end(), data_ranges)
        # Use canonical DOI key so this deduplicates with the URL/DOI pass
        _add(DatasetRef(
            name=acc, accession=acc, repository_hint="pangaea",
            raw_citation=ctx, source="regex", used_in_study=use_in_study
        ), f"10.1594/pangaea.{num}")

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
             "MERRA-2", "MERRA", "CALIPSO", "CLOUDSAT", "GPM", "IMERG", "TRMM", "SMOS",
             "SMAP", "SWOT", "GEDI", "AERONET", "ARCTICDEM", "REMA", "SRTM",
             "JASON-1", "JASON-2", "JASON-3", "TOPEX/POSEIDON",
             "ECCO", "DAYMET"}:
        return "nasa_earthdata"
    if n in {"AMSR-E", "AMSR2", "SSMIS", "SSM/I", "NSIDC SEA ICE INDEX", "IABP"}:
        return "nsidc"
    if n in {"NCEP", "NCEP/NCAR", "CFSR", "CFSV2", "GPCP", "CMORPH", "PERSIANN",
             "PERSIANN-CDR", "MSWEP", "GPCC", "GHCND", "GHCNM", "GHCN-D", "GHCN-M",
             "ERSST", "ICOADS", "PSMSL", "IGRA", "WOD", "WORLD OCEAN DATABASE"}:
        return "noaa"
    if n in {"GLORYS", "GLORYS12", "OSTIA", "OSI-SAF", "ORAS5"}:
        return "cmems"
    if n in {"CMIP3", "CMIP5", "CMIP6"}:
        return "esgf"
    if n in {"FLUXNET", "FLUXNET2015"}:
        return "fluxnet"
    if n in {"AMERIFLUX"}:
        return "ameriflux"
    if n in {"ICOS"}:
        return "icos"
    if n in {"SOCAT", "GLODAP", "EN4", "HADISST", "HADCRUT", "HADSST",
             "CHELSA", "WORLDCLIM", "CRU TS", "E3SM"}:
        return "other"
    if n in {"SNOTEL"}:
        return "noaa"
    if n in {"GTN-P"}:
        return "other"
    if n in {"TANDEM-X", "ASTER GDEM"}:
        return "other"
    if n in {"GLOBSNOW"}:
        return "other"
    if n in {"NCAR-RDA"}:
        return "ncar_rda"
    if n in {"TROPOMI", "SENTINEL-1", "SENTINEL-2", "SENTINEL-3", "SENTINEL-5P", "SENTINEL-6"}:
        return "copernicus_cds"
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


def _extract_json_array(text: str) -> list:
    """
    Robustly extract a JSON array from LLM output.
    Handles: markdown fences, leading prose, trailing notes, partial wrapping.
    """
    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    text = re.sub(r'```\s*$', '', text).strip()

    # 2. Try direct parse first
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass

    # 3. Extract the outermost [...] block (handles leading/trailing prose)
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            pass

    # 4. Try to find and parse individual {...} objects (handles array without brackets)
    objects = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
    parsed = []
    for obj in objects:
        try:
            parsed.append(json.loads(obj))
        except json.JSONDecodeError:
            continue
    if parsed:
        return parsed

    # 5. Give up — return empty list (logged as JSON-err by caller)
    raise json.JSONDecodeError("No JSON array found in LLM response", text, 0)

# Signals that a chunk is worth sending to the LLM
_DATASET_SIGNAL_RE = re.compile(
    r'\b(?:'
    # Explicit data access language
    r'dataset|data\s+availability|data\s+available|code\s+and\s+data|'
    r'downloaded?\s+from|obtained?\s+from|accessed?\s+(?:at|from)|'
    r'accession|repository|zenodo|pangaea|figshare|dryad|dataverse|osf\.io|'
    r'doi\.org|10\.\d{4}/'
    # Known remote sensing / reanalysis products
    r'|ERA5|ERA-Interim|ERA-40|MODIS|VIIRS|CryoSat|Sentinel|ICESat|GRACE|'
    r'MERRA|CMIP|TOPAZ|PIOMAS|AMSR|HYCOM|FESOM|GLORYS|OSTIA|GPM|IMERG|'
    r'CALIPSO|NCEP|JRA-55|CFSR|GISTEMP|HadCRUT|AVHRR|ARGO|MOSAiC|SHEBA|'
    r'SMAP|SWOT|GEDI|TROPOMI|AERONET|ArcticDEM|REMA|SRTM|TanDEM|'
    r'FLUXNET|AmeriFlux|ICOS|SOCAT|GLODAP|SNOTEL|GTN-P|'
    r'GHCN|GPCC|CRU\s+TS|CHELSA|DAYMET|WorldClim|'
    r'ORAS5|ECCO|HadISST|ERSST|ICOADS|PSMSL|IGRA|'
    r'Jason-[123]|GlobSnow|E3SM|NCAR-RDA|'
    # General data/observation terms
    r'reanalysis|remote\s+sensing|satellite\s+data|in\s+situ|buoy|CTD|'
    r'mooring|observat(?:ion|ory)|field\s+campaign|field\s+station|'
    r'flux\s+tower|eddy\s+covariance|radiocarbon|dendrochronolog|'
    r'GPS\s+(?:survey|data|measurement)|raised\s+beach|isolation\s+basin|'
    r'isostatic|cosmogenic|diatom\s+(?:analysis|assemblage|record)|'
    # Field measurement / sample collection
    r'samples?\s+(?:were\s+)?collect|water\s+samples?|soil\s+samples?|'
    r'collected\s+from|sampl(?:ed|ing)\s+(?:at|in|from)|'
    r'incubat(?:ed|ion)|dissolved\s+organic|stream(?:water)?|'
    r'permafrost|tundra|peatland|wetland|watershed|catchment|'
    r'flux\s+(?:tower|data|measurements?)|'
    r'ice\s+core|sediment\s+core|water\s+column|depth\s+profile|'
    r'monitoring\s+(?:station|network|program|data)|long.term\s+(?:data|monitoring)|'
    # Climate modeling / ocean science papers
    r'RCP\s*\d|SSP\s*\d|climate\s+(?:model|projection|scenario)|'
    r'GCM|AOGCM|earth\s+system\s+model|coupled\s+model|'
    r'model\s+(?:output|simulation|experiment|run)|'
    r'observational\s+(?:data|record|constraint)|'
    r'sea\s+surface\s+temperature|ocean\s+(?:heat|carbon|storage)|'
    r'carbon\s+(?:uptake|flux|storage|sink|cycle)|'
    r'dissolved\s+inorganic\s+carbon|alkalinity|pCO2|fCO2|'
    r'sea\s+ice\s+(?:extent|concentration|thickness)|'
    r'temperature\s+(?:record|anomaly|trend)|'
    r'instrumental\s+record|paleoclimate|proxy\s+(?:data|record)|'
    r'HadGEM|GISS|CESM|MPI-ESM|IPSL|MIROCe|ACCESS|CanESM|NorESM|GFDL'
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
    resp = requests.post(_OLLAMA_URL, json=payload, timeout=240)
    resp.raise_for_status()
    body = resp.json()
    content = body["message"]["content"].strip()
    usage = {
        "input_tokens":  body.get("prompt_eval_count", 0),
        "output_tokens": body.get("eval_count", 0),
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return _extract_json_array(content), usage


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
    u = response.usage
    usage = {
        "input_tokens":  u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens":  u.total_tokens if u else 0,
    }
    return _extract_json_array(content), usage


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
    u = response.usage
    usage = {
        "input_tokens":  u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "total_tokens":  u.total_tokens if u else 0,
    }
    return _extract_json_array(content), usage


def extract_llm(text: str, api_key: Optional[str] = None,
                existing_refs: Optional[list] = None) -> list[DatasetRef]:
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
    # Only apply the cut if the section marker is past 20% of the document to
    # avoid a "Funding" line in the header/metadata block cutting the text too early.
    hard_end = _SECTION_END_RE.search(text)
    _min_cut = max(20_000, len(text) // 5)
    llm_text = text[:hard_end.start()] if (hard_end and hard_end.start() > _min_cut) else text

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
    # Seed seen_names with regex-pass refs so LLM doesn't re-extract them
    seen_names: set[str] = set(r.name.lower() for r in (existing_refs or []))
    # Track lowercased accepted names for prefix-based dedup
    # Rule: if accepted "era5" exists, skip "era5 reanalysis product" (starts with "era5 ")
    #       Note: "era5 " (with space) prevents "era5" from matching "era5-land"
    seen_prefixes: list[str] = [r.name.lower() for r in (existing_refs or [])]
    # Track first tokens of multi-word names where the first token is an acronym/proper noun
    # e.g. "PAMARCMiP aircraft campaign data" → first_tok = "pamarcmip"
    # Any later "PAMARCMiP campaign data", "PAMARCMiP observations" → skip
    _ACRONYM_RE = re.compile(r'^[A-Z][A-Za-z0-9\-]{3,}$')
    seen_first_toks: set[str] = set()
    for r in (existing_refs or []):
        toks = r.name.split()
        if len(toks) > 1 and _ACRONYM_RE.match(toks[0]):
            seen_first_toks.add(toks[0].lower())

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
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "").strip()
                if not name or not _is_likely_dataset(name):
                    continue
                name_lower = name.lower()
                if name_lower in seen_names:
                    continue
                # Prefix dedup: "ERA5 reanalysis product" → skip if "era5" already seen;
                # "ERA5" → skip if "era5 reanalysis product" already seen.
                # Use " " boundary so "era5" doesn't match "era5-land".
                if any(
                    name_lower.startswith(p + " ") or p.startswith(name_lower + " ")
                    for p in seen_prefixes
                ):
                    continue
                # First-token dedup: "PAMARCMiP campaign data" → skip if "pamarcmip" seen
                # from a previous multi-word name with that first token
                name_toks = name.split()
                first_tok_lower = name_toks[0].lower() if name_toks else ""
                if (len(name_toks) > 1
                        and _ACRONYM_RE.match(name_toks[0])
                        and first_tok_lower in seen_first_toks):
                    continue
                seen_names.add(name_lower)
                seen_prefixes.append(name_lower)
                if len(name_toks) > 1 and _ACRONYM_RE.match(name_toks[0]):
                    seen_first_toks.add(first_tok_lower)
                raw_primary = item.get("is_primary", False)
                is_primary = (
                    raw_primary is True
                    or (isinstance(raw_primary, str) and raw_primary.lower() == "true")
                )
                # Strip journal URLs/DOIs that the LLM picked up from the paper itself
                url_raw = item.get("url") or ""
                url = url_raw if url_raw and not _is_journal_url(url_raw) else None
                doi_raw = item.get("doi") or ""
                doi = doi_raw if doi_raw and _is_data_doi(doi_raw) else None
                refs.append(DatasetRef(
                    name=name,
                    url=url,
                    doi=doi,
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
    if len(text) < 3000:
        print(f"  WARNING: Very little text extracted ({len(text)} chars). "
              f"This PDF may be scanned/image-based — dataset extraction will be limited.")

    t0 = time.time()
    regex_refs = extract_regex(text)

    llm_refs = []
    if use_llm:
        llm_refs = extract_llm(text, api_key=api_key, existing_refs=regex_refs)

    total = regex_refs + llm_refs
    print(f"\n{'═' * W}")
    print(f"  TOTALS  —  {fname}")
    print(f"  Regex refs : {len(regex_refs):>4}")
    print(f"  LLM   refs : {len(llm_refs):>4}")
    print(f"  Combined   : {len(total):>4}  raw refs  (before dedup)")
    print(f"  Wall time  : {time.time()-t0:.1f}s")
    print(f"{'═' * W}")

    # ── Active / Secondary summary (mirrors the UI display) ─────────────────
    if total:
        primary = [r for r in total if r.is_primary]
        secondary = [r for r in total if not r.is_primary]

        heuristic_note = ""
        if not primary and total:
            # LLM found no primary — apply preference logic: specific repos > portals > position
            # Preference: paper-output repos first, forcing/reanalysis portals last
            _PREF_ORDER = [
                ("pangaea", "zenodo", "figshare"),               # primary paper-output repos
                ("arctic_data_center", "dryad", "seanoe"),       # other specific data repos
                ("data_repository",),                            # generic doi (better than portals)
                ("nasa_earthdata", "nsidc", "cmems", "copernicus_cds",
                 "eumetsat", "noaa", "noaa_ncei", "ncar_rda"),   # forcing/reanalysis portals
            ]
            promoted = False
            for hint_group in _PREF_ORDER:
                for r in total:
                    if (r.url or r.doi or r.accession) and r.repository_hint in hint_group:
                        primary = [r]
                        secondary = [x for x in total if x is not r]
                        heuristic_note = f"  (promoted from refs — no LLM primary; best repo: {r.repository_hint})"
                        promoted = True
                        break
                if promoted:
                    break
            if not promoted:
                primary = [total[0]]
                secondary = total[1:]
                heuristic_note = "  (primary identified by position — LLM did not mark one)"
        elif len(primary) > 2:
            # Multiple chunks each marked a primary — prefer refs with real data URLs
            has_url = [r for r in primary if r.url or r.doi or r.accession]
            no_url  = [r for r in primary if not (r.url or r.doi or r.accession)]
            kept = (has_url + no_url)[:2]
            demoted = [r for r in primary if r not in kept]
            for r in demoted:
                r.is_primary = False
            primary = kept
            secondary = [r for r in total if not r.is_primary]
            heuristic_note = f"  (capped at 2 — {len(demoted)} additional LLM primary mark(s) demoted)"

        # If all LLM-marked primaries have no data link (journal DOI stripped, no URL/accession),
        # promote the best regex-found data repository ref instead.
        _REPO_PREF_ORDER = [
            ("pangaea", "zenodo", "figshare"),
            ("arctic_data_center", "dryad", "seanoe"),
            ("data_repository",),
            ("nasa_earthdata", "nsidc", "cmems", "copernicus_cds",
             "eumetsat", "noaa", "noaa_ncei", "ncar_rda", "other"),
        ]
        if primary and not any(r.url or r.doi or r.accession for r in primary):
            promoted = False
            for hint_group in _REPO_PREF_ORDER:
                for r in secondary:
                    if (r.url or r.doi or r.accession) and r.repository_hint in hint_group:
                        for p in primary:
                            p.is_primary = False
                        r.is_primary = True
                        primary = [r]
                        secondary = [x for x in secondary if x is not r]
                        heuristic_note = "  (promoted from secondary — LLM primary had no data link after filtering)"
                        promoted = True
                        break
                if promoted:
                    break

        print(f"\n  Active Dataset(s)  [{len(primary)}]{heuristic_note}")
        for r in primary:
            src_tag = f"[{r.source}]"
            hint = f"  {r.repository_hint}" if r.repository_hint else ""
            url_part = f"  {r.url}" if r.url else (f"  doi:{r.doi}" if r.doi else "")
            print(f"    ★  {r.name:<45} {src_tag:<7}{hint}{url_part}")

        print(f"\n  Secondary References  [{len(secondary)}]")
        if secondary:
            for r in secondary:
                src_tag = f"[{r.source}]"
                hint = f"  {r.repository_hint}" if r.repository_hint else ""
                url_part = f"  {r.url}" if r.url else (f"  doi:{r.doi}" if r.doi else "")
                print(f"    ·  {r.name:<45} {src_tag:<7}{hint}{url_part}")
        else:
            print(f"    (none)")

        print(f"{'═' * W}\n")
    else:
        print(f"  No datasets found.\n{'═' * W}\n")

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
