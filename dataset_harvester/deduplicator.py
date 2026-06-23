"""
Iteration 2 — Deduplication Engine

Merges DatasetRef objects that refer to the same dataset:
  - Exact DOI / URL / accession match
  - Known-alias match  (ERA-Interim reanalysis → ERA-Interim)
  - Fuzzy name match on stripped core name (>= SIMILARITY_THRESHOLD)
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from difflib import SequenceMatcher

from extractor import DatasetRef

SIMILARITY_THRESHOLD = 0.82

# Words that inflate the name but don't identify the dataset
_NOISE_WORDS = re.compile(
    r'\b(reanalysis|observation|observations|observation data|'
    r'sea[- ]ice|sea ice|data|dataset|datasets|product|products|'
    r'system|model|models|output|outputs|record|records|'
    r'global|regional|historical|gridded|monthly|daily|hourly|'
    r'version|v\d[\d.]*)\b',
    re.IGNORECASE
)

# Explicit alias table: any key normalises to its value before comparison
_ALIASES: dict[str, str] = {
    "era interim":              "era-interim",
    "era-interim reanalysis":   "era-interim",
    "era interim reanalysis":   "era-interim",
    "era5 reanalysis":          "era5",
    "era5 reanalysis data":     "era5",
    "merra 2":                  "merra-2",
    "merra2":                   "merra-2",
    "ncep ncar":                "ncep/ncar",
    "ncep/ncar reanalysis":     "ncep/ncar",
    "piomas sea volume":        "piomas",
    "piomas sea-ice volume":    "piomas",
    "piomas sea ice volume":    "piomas",
    "pan arctic ice ocean modeling and assimilation system": "piomas",
    "topaz ocean model":        "topaz4",
    "topaz4 ocean":             "topaz4",
    "topaz ocean model system": "topaz4",
    "topaz4 ocean-sea ice data assimilation system": "topaz4",
    "icesat 2":                 "icesat-2",
    "cryosat 2":                "cryosat-2",
    "grace fo":                 "grace-fo",
    "amsr e":                   "amsr-e",
    "ssm i":                    "ssm/i",
    "nsidc sea ice index":      "nsidc sea ice index",
    "wam 4":                    "wam-4",
    "wam4":                     "wam-4",
    "wam 3":                    "wam-3",
    "wave watch iii":           "wavewatch iii",
    "ww3":                      "wavewatch iii",
}


def _normalize_doi(doi: str) -> str:
    doi = doi.lower().strip()
    doi = re.sub(r'^(doi:|https?://doi\.org/)', '', doi)
    return doi


def _core_name(name: str) -> str:
    """Strip noise words and punctuation to get the identifying core."""
    n = name.lower().strip()
    # Version suffixes
    n = re.sub(r'\s*v\d[\d.]*\s*$', '', n)
    n = re.sub(r'\s*\(\d{4}\)\s*$', '', n)
    # Punctuation/whitespace normalise
    n = re.sub(r'[-_/]+', ' ', n)
    n = re.sub(r'\s+', ' ', n)
    # Check alias table first
    if n in _ALIASES:
        return _ALIASES[n]
    # Strip noise words
    stripped = _NOISE_WORDS.sub('', n).strip()
    stripped = re.sub(r'\s+', ' ', stripped).strip()
    return stripped if stripped else n


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _same_dataset(name_a: str, name_b: str) -> bool:
    ca, cb = _core_name(name_a), _core_name(name_b)
    # Exact core match
    if ca == cb:
        return True
    # One is a prefix of the other (handles "PIOMAS" vs "PIOMAS sea-ice volume")
    if ca.startswith(cb) or cb.startswith(ca):
        return True
    # Fuzzy on cores
    return _similarity(ca, cb) >= SIMILARITY_THRESHOLD


@dataclass
class DeduplicatedDataset:
    canonical_name: str
    url: Optional[str] = None
    doi: Optional[str] = None
    accession: Optional[str] = None
    repository_hint: Optional[str] = None
    mention_count: int = 1
    sources: list[str] = field(default_factory=list)
    raw_refs: list[DatasetRef] = field(default_factory=list)
    is_primary: bool = False
    used_in_study: bool = False

    def to_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "url": self.url,
            "doi": self.doi,
            "accession": self.accession,
            "repository_hint": self.repository_hint,
            "mention_count": self.mention_count,
            "sources": self.sources,
            "is_primary": self.is_primary,
            "used_in_study": self.used_in_study,
        }


def _merge_into(existing: DeduplicatedDataset, ref: DatasetRef, source: str):
    existing.mention_count += 1
    if source and source not in existing.sources:
        existing.sources.append(source)
    existing.raw_refs.append(ref)
    if not existing.url and ref.url:
        existing.url = ref.url
    if not existing.doi and ref.doi:
        existing.doi = _normalize_doi(ref.doi)
    if not existing.accession and ref.accession:
        existing.accession = ref.accession
    if not existing.repository_hint and ref.repository_hint:
        existing.repository_hint = ref.repository_hint
    # Any mention marked primary or used_in_study promotes the whole group
    if ref.is_primary:
        existing.is_primary = True
    if ref.used_in_study:
        existing.used_in_study = True
    # Prefer shorter canonical name (less verbose)
    if len(ref.name) < len(existing.canonical_name):
        existing.canonical_name = ref.name


def deduplicate(refs_by_source: dict[str, list[DatasetRef]]) -> list[DeduplicatedDataset]:
    deduped: list[DeduplicatedDataset] = []
    doi_index: dict[str, int] = {}
    url_index: dict[str, int] = {}
    acc_index: dict[str, int] = {}

    def _find_existing(ref: DatasetRef) -> Optional[int]:
        # 1. DOI exact match
        doi_to_check = ref.doi
        if not doi_to_check and ref.url:
            stripped = re.sub(r'^https?://doi\.org/', '', ref.url, flags=re.IGNORECASE)
            if stripped != ref.url:
                doi_to_check = stripped
        if doi_to_check:
            ndoi = _normalize_doi(doi_to_check)
            if ndoi in doi_index:
                return doi_index[ndoi]

        # 2. URL exact match
        if ref.url:
            url = ref.url.rstrip("/")
            if url in url_index:
                return url_index[url]

        # 3. Accession exact match
        if ref.accession and ref.accession.upper() in acc_index:
            return acc_index[ref.accession.upper()]

        # 4. Name match (alias table + fuzzy on core name)
        for i, d in enumerate(deduped):
            if _same_dataset(ref.name, d.canonical_name):
                return i

        return None

    for source, refs in refs_by_source.items():
        for ref in refs:
            idx = _find_existing(ref)
            if idx is not None:
                _merge_into(deduped[idx], ref, source)
            else:
                d = DeduplicatedDataset(
                    canonical_name=ref.name,
                    url=ref.url,
                    doi=_normalize_doi(ref.doi) if ref.doi else None,
                    accession=ref.accession,
                    repository_hint=ref.repository_hint,
                    mention_count=1,
                    sources=[source] if source else [],
                    raw_refs=[ref],
                    is_primary=ref.is_primary,
                    used_in_study=ref.used_in_study,
                )
                idx = len(deduped)
                deduped.append(d)
                if d.doi:
                    doi_index[d.doi] = idx
                if d.url:
                    url_index[d.url.rstrip("/")] = idx
                if d.accession:
                    acc_index[d.accession.upper()] = idx

    deduped.sort(key=lambda d: d.mention_count, reverse=True)
    return deduped
