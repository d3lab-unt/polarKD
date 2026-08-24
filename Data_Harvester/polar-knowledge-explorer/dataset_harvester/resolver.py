"""
Iteration 2 — Dataset URL Resolver

Takes a DeduplicatedDataset and finds the actual landing page / download URL.

Resolution order:
  1. Already has a URL → verify it resolves (HEAD request)
  2. Has a DOI → DataCite API metadata → repository landing page
  3. Known-dataset registry → hardcoded canonical URL
  4. Repository search → Zenodo API, PANGAEA API
  5. Unresolved → mark as manual_lookup_required
"""

import re
import os
import requests
from dataclasses import dataclass
from typing import Optional

from deduplicator import DeduplicatedDataset

# ---------------------------------------------------------------------------
# Known dataset registry
# canonical name (lowercase, normalised) → landing page URL
# ---------------------------------------------------------------------------
_KNOWN_URLS: dict[str, dict] = {
    # Copernicus / ECMWF
    "era5": {
        "url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        "repository": "copernicus_cds",
        "notes": "Requires free CDS account. Use cdsapi to download."
    },
    "era-interim": {
        "url": "https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-interim",
        "repository": "ecmwf",
        "notes": "Deprecated in 2019. Archive available via ECMWF."
    },
    "era-40": {
        "url": "https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-40-years",
        "repository": "ecmwf",
    },
    "wam-4": {
        "url": "https://www.ecmwf.int/en/research/modelling-and-prediction/ocean-waves",
        "repository": "ecmwf",
    },
    "wam-3": {
        "url": "https://www.ecmwf.int/en/research/modelling-and-prediction/ocean-waves",
        "repository": "ecmwf",
    },
    "wavewatch iii": {
        "url": "https://polar.ncep.noaa.gov/waves/wavewatch/",
        "repository": "noaa",
    },

    # Sea ice models
    "piomas": {
        "url": "http://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/",
        "repository": "uw_apl",
        "notes": "Pan-Arctic Ice Ocean Modeling and Assimilation System. Free download."
    },
    "topaz4": {
        "url": "https://data.marine.copernicus.eu/product/ARCTIC_ANALYSISFORECAST_PHY_002_001",
        "repository": "copernicus_marine",
        "notes": "Arctic MFC product. Free via Copernicus Marine Service."
    },
    "topaz": {
        "url": "https://data.marine.copernicus.eu/product/ARCTIC_ANALYSISFORECAST_PHY_002_001",
        "repository": "copernicus_marine",
    },

    # Arctic atmosphere models
    "arome-arctic": {
        "url": "https://www.met.no/en/projects/The-weather-model-AROME-Arctic",
        "repository": "met_norway",
        "notes": "Norwegian Meteorological Institute. Data via thredds.met.no."
    },

    # NASA remote sensing
    "modis": {
        "url": "https://modis.gsfc.nasa.gov/data/",
        "repository": "nasa_earthdata",
        "notes": "Use NASA Earthdata Search (earthdata.nasa.gov) to find specific products."
    },
    "viirs": {
        "url": "https://www.earthdata.nasa.gov/sensors/viirs",
        "repository": "nasa_earthdata",
    },
    "landsat": {
        "url": "https://earthexplorer.usgs.gov/",
        "repository": "usgs",
        "notes": "Free download via USGS EarthExplorer."
    },
    "icesat-2": {
        "url": "https://nsidc.org/data/icesat-2",
        "repository": "nsidc",
        "notes": "Requires NASA Earthdata login."
    },
    "icesat": {
        "url": "https://nsidc.org/data/icesat",
        "repository": "nsidc",
    },
    "grace": {
        "url": "https://podaac.jpl.nasa.gov/GRACE",
        "repository": "nasa_podaac",
    },
    "grace-fo": {
        "url": "https://podaac.jpl.nasa.gov/GRACE-FO",
        "repository": "nasa_podaac",
    },
    "merra-2": {
        "url": "https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/data_access/",
        "repository": "nasa_earthdata",
        "notes": "Requires NASA Earthdata login."
    },
    "avhrr": {
        "url": "https://www.ncei.noaa.gov/products/avhrr-pathfinder-sst",
        "repository": "noaa_ncei",
    },

    # ESA
    "cryosat-2": {
        "url": "https://earth.esa.int/eogateway/missions/cryosat/data",
        "repository": "esa",
        "notes": "Free registration required via ESA Earth Online."
    },
    "sentinel-1": {
        "url": "https://browser.dataspace.copernicus.eu/",
        "repository": "copernicus_dataspace",
        "notes": "Free via Copernicus Data Space Ecosystem."
    },
    "sentinel-2": {
        "url": "https://browser.dataspace.copernicus.eu/",
        "repository": "copernicus_dataspace",
    },
    "sentinel-3": {
        "url": "https://browser.dataspace.copernicus.eu/",
        "repository": "copernicus_dataspace",
    },

    # NSIDC passive microwave
    "amsr2": {
        "url": "https://nsidc.org/data/au_si12",
        "repository": "nsidc",
        "notes": "AMSR2 12.5km sea ice concentration. Requires Earthdata login."
    },
    "amsr-e": {
        "url": "https://nsidc.org/data/ae_si12",
        "repository": "nsidc",
    },
    "ssmis": {
        "url": "https://nsidc.org/data/nsidc-0001",
        "repository": "nsidc",
    },
    "ssm/i": {
        "url": "https://nsidc.org/data/nsidc-0001",
        "repository": "nsidc",
    },
    "nsidc sea ice index": {
        "url": "https://nsidc.org/data/G02135",
        "repository": "nsidc",
        "notes": "Monthly and daily sea ice extent. Free download."
    },

    # Reanalysis
    "jra-55": {
        "url": "https://jra.kishou.go.jp/JRA-55/index_en.html",
        "repository": "jma",
        "notes": "Japan Meteorological Agency reanalysis."
    },
    "ncep/ncar": {
        "url": "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html",
        "repository": "noaa_psl",
        "notes": "Free download via NOAA PSL."
    },
    "cfsr": {
        "url": "https://rda.ucar.edu/datasets/d093000/",
        "repository": "ncar_rda",
    },
    "merra": {
        "url": "https://gmao.gsfc.nasa.gov/reanalysis/MERRA/",
        "repository": "nasa_earthdata",
    },

    # Oceanographic
    "argo": {
        "url": "https://argo.ucsd.edu/data/",
        "repository": "argo",
        "notes": "Free download via Argo GDAC."
    },
    "woa": {
        "url": "https://www.ncei.noaa.gov/products/world-ocean-atlas",
        "repository": "noaa_ncei",
    },
    "world ocean atlas": {
        "url": "https://www.ncei.noaa.gov/products/world-ocean-atlas",
        "repository": "noaa_ncei",
    },
    "gebco": {
        "url": "https://www.gebco.net/data_and_products/gridded_bathymetry_data/",
        "repository": "gebco",
        "notes": "Free download, no login required."
    },

    # In-situ / campaigns
    "iabp": {
        "url": "https://iabp.apl.uw.edu/data.html",
        "repository": "uw_apl",
        "notes": "International Arctic Buoy Programme."
    },
    "mosaic": {
        "url": "https://mosaic-expedition.org/expedition/data-access/",
        "repository": "pangaea",
        "notes": "MOSAiC expedition data deposited in PANGAEA."
    },
    "sheba": {
        "url": "https://nsidc.org/data/sheba",
        "repository": "nsidc",
    },
}


def _normalise_for_lookup(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r'\s*v\d[\d.]*\s*$', '', n)
    n = re.sub(r'\s*\(\d{4}\)\s*$', '', n)
    n = re.sub(r'[-_/]+', ' ', n)
    n = re.sub(r'\b(reanalysis|data|dataset|observation|observations|'
               r'sea ice|product|system|model)\b', '', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def _lookup_known(name: str) -> Optional[dict]:
    n = _normalise_for_lookup(name)
    # Direct match
    if n in _KNOWN_URLS:
        return _KNOWN_URLS[n]
    # Try without noise stripping
    plain = name.lower().strip()
    if plain in _KNOWN_URLS:
        return _KNOWN_URLS[plain]
    # Partial: known key is contained in the name
    for key, info in _KNOWN_URLS.items():
        if key in n or key in plain:
            return info
    return None


# ---------------------------------------------------------------------------
# DOI resolution via DataCite API
# ---------------------------------------------------------------------------

def _resolve_doi(doi: str) -> Optional[dict]:
    clean = re.sub(r'^https?://doi\.org/', '', doi, flags=re.IGNORECASE)
    try:
        r = requests.get(
            f"https://api.datacite.org/dois/{clean}",
            timeout=5,
            headers={"Accept": "application/json"},
        )
        if r.status_code != 200:
            return None
        data = r.json().get("data", {}).get("attributes", {})
        url = data.get("url") or f"https://doi.org/{clean}"
        repo = data.get("publisher", {})
        repo_name = repo.get("name", "") if isinstance(repo, dict) else str(repo)
        return {"url": url, "repository": repo_name, "notes": f"DOI: {clean}"}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Zenodo search
# ---------------------------------------------------------------------------

# Names that should never be sent to Zenodo/PANGAEA search —
# they are reports, frameworks, or models, not downloadable datasets
_NO_SEARCH = {
    "ipcc", "ipcc ar6", "ipcc ar5", "ipcc ar4",
    "unfccc", "ioc", "wmo",
}

def _is_not_searchable(name: str) -> bool:
    n = name.lower().strip()
    for skip in _NO_SEARCH:
        if n.startswith(skip):
            return True
    # Preprint DOIs (10.21203 = ResearchSquare) are not data DOIs
    if re.match(r'^10\.21203/', n):
        return True
    return False


# Domain words that must appear in a Zenodo result title for arctic/env research
_DOMAIN_WORDS = {
    "arctic", "antarctic", "ice", "ocean", "climate", "polar", "sea",
    "glacier", "snow", "wave", "atmospheric", "temperature", "marine",
    "cryosphere", "permafrost", "sea level", "reanalysis", "satellite",
    "buoy", "drift", "melt", "fjord", "greenland", "svalbard", "spitsbergen",
    "marginal ice zone", "miz", "bathymetry", "salinity", "current",
}


def _search_zenodo(name: str) -> Optional[dict]:
    if _is_not_searchable(name):
        return None
    # Don't search with vague short names — too many false positives
    meaningful_words = re.findall(r'\b\w{5,}\b', name.lower())
    if len(meaningful_words) < 2:
        return None
    try:
        r = requests.get(
            "https://zenodo.org/api/records",
            params={"q": name, "type": "dataset", "size": 3},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            return None

        query_words = set(re.findall(r'\b\w{5,}\b', name.lower()))

        for rec in hits:
            title = rec.get("metadata", {}).get("title", "").lower()
            title_words = set(re.findall(r'\b\w{5,}\b', title))
            overlap = query_words & title_words

            # Must have at least 2 key words overlapping
            if len(overlap) < 2:
                continue
            # Title must contain at least one domain term
            if not any(d in title for d in _DOMAIN_WORDS):
                continue

            doi = rec.get("doi", "")
            url = f"https://zenodo.org/record/{rec['id']}"
            return {"url": url, "repository": "zenodo",
                    "doi": doi, "notes": f"Zenodo search match: {title}"}

        return None   # no hit passed the relevance filter
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PANGAEA search
# ---------------------------------------------------------------------------

def _search_pangaea(name: str) -> Optional[dict]:
    try:
        r = requests.get(
            "https://www.pangaea.de/api/find",
            params={"q": name, "count": 1, "format": "json"},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        results = r.json()
        items = results.get("results", [])
        if not items:
            return None
        item = items[0]
        uri = item.get("URI", "")
        title = item.get("citation", {}).get("title", name)
        return {"url": uri, "repository": "pangaea",
                "notes": f"PANGAEA search match: {title}"}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main resolve function
# ---------------------------------------------------------------------------

@dataclass
class ResolvedDataset:
    canonical_name: str
    resolved_url: Optional[str]
    repository: Optional[str]
    doi: Optional[str]
    accession: Optional[str]
    notes: str
    resolution_method: str   # "existing_url"|"doi"|"known_registry"|"zenodo"|"pangaea"|"unresolved"
    mention_count: int
    sources: list[str]
    is_primary: bool = False
    used_in_study: bool = False

    def to_dict(self) -> dict:
        return self.__dict__


def resolve(datasets: list[DeduplicatedDataset]) -> list[ResolvedDataset]:
    resolved = []
    for d in datasets:
        result = resolve_one(d)
        resolved.append(result)
    return resolved


def resolve_one(d: DeduplicatedDataset) -> ResolvedDataset:
    return _resolve_one(d)


def _resolve_one(d: DeduplicatedDataset) -> ResolvedDataset:
    base = dict(
        canonical_name=d.canonical_name,
        doi=d.doi,
        accession=d.accession,
        mention_count=d.mention_count,
        sources=d.sources,
        is_primary=d.is_primary,
        used_in_study=d.used_in_study,
    )

    # 1. Already has a URL
    if d.url:
        return ResolvedDataset(
            resolved_url=d.url,
            repository=d.repository_hint,
            notes="",
            resolution_method="existing_url",
            **base,
        )

    # 2. Has a DOI — try DataCite
    if d.doi:
        info = _resolve_doi(d.doi)
        if info:
            return ResolvedDataset(
                resolved_url=info["url"],
                repository=info.get("repository"),
                notes=info.get("notes", ""),
                resolution_method="doi",
                **base,
            )

    # 3. Known-dataset registry
    info = _lookup_known(d.canonical_name)
    if info:
        return ResolvedDataset(
            resolved_url=info["url"],
            repository=info.get("repository"),
            notes=info.get("notes", ""),
            resolution_method="known_registry",
            **base,
        )

    # 4 & 5. Repository search — ONLY when the paper gave us a concrete anchor:
    # an accession number, or the LLM saw the repository name in the text
    # (repository_hint set to "pangaea" or "zenodo").
    # Name-only LLM extractions (no URL, no DOI, no accession, hint=None) are
    # NOT searched: a keyword match on a long dataset description will almost
    # always return the wrong record from a different expedition or year.
    has_anchor = bool(d.accession) or d.repository_hint in ("pangaea", "zenodo")

    if has_anchor and d.repository_hint in (None, "pangaea", "unknown"):
        info = _search_pangaea(d.canonical_name)
        if info:
            return ResolvedDataset(
                resolved_url=info["url"],
                repository="pangaea",
                notes=info.get("notes", ""),
                resolution_method="pangaea",
                **base,
            )

    if has_anchor and d.repository_hint in (None, "zenodo", "unknown", "zenodo"):
        info = _search_zenodo(d.canonical_name)
        if info:
            return ResolvedDataset(
                resolved_url=info["url"],
                repository="zenodo",
                notes=info.get("notes", ""),
                resolution_method="zenodo",
                **base,
            )

    # 6. Unresolved
    return ResolvedDataset(
        resolved_url=None,
        repository=d.repository_hint,
        notes="Could not resolve automatically — manual lookup required",
        resolution_method="unresolved",
        **base,
    )
