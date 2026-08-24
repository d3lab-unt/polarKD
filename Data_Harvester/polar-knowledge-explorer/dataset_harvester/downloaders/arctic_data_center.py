"""
Arctic Data Center downloader — uses the DataONE Member Node REST API.

Arctic Data Center (arcticdata.io) hosts data openly — no authentication
required for public datasets. DOI prefix: 10.18739

How it works:
  1. DOI → DataONE Solr query to find the package identifier & resource map
  2. Resource map → list all DATA-type objects (the actual files)
  3. Download each file via GET {MN_BASE}/object/{pid}
"""

import os
import re
import urllib.parse
import requests

from .generic import download_file

_MN_BASE  = "https://arcticdata.io/metacat/d1/mn/v2"
_MAX_MB   = 200.0
_TIMEOUT  = 12   # seconds per API call


def _extract_doi(url: str = None, doi: str = None) -> str | None:
    """Return a bare DOI (e.g. '10.18739/A2FQ9Q664') from a URL or DOI string."""
    for s in [doi or "", url or ""]:
        m = re.search(r'(10\.18739/[A-Za-z0-9]+)', s, re.IGNORECASE)
        if m:
            return m.group(1).rstrip(".,;:")
        m = re.search(r'doi\.org/(10\.\d{4,9}/[^\s\)\]"\'<>]+)', s, re.IGNORECASE)
        if m:
            return m.group(1).rstrip(".,;:")
    return None


def _solr(params: dict, timeout: int = _TIMEOUT) -> dict:
    """Run a Solr query against the ADC Member Node. Returns parsed JSON or {}."""
    try:
        r = requests.get(
            f"{_MN_BASE}/query/solr/",
            params={**params, "wt": "json"},
            timeout=timeout,
            headers={"User-Agent": "polar-kg-harvester/1.0"},
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _find_resource_map(doi: str) -> tuple[str | None, str | None]:
    """
    Query DataONE Solr for the package resource map PID and dataset title.
    Returns (resource_map_pid, title) — both None if not found.
    """
    encoded = f"doi:{doi}"
    # Try seriesId first (preferred for versioned packages), then identifier
    for field in ("seriesId", "identifier"):
        data = _solr({
            "q":   f'{field}:"{encoded}"',
            "fl":  "id,resourceMap,title",
            "rows": 1,
        })
        docs = data.get("response", {}).get("docs", [])
        if docs:
            doc = docs[0]
            rm_list = doc.get("resourceMap", [])
            rm = rm_list[0] if rm_list else None
            return rm, doc.get("title", doi)
    return None, None


def _list_files(resource_map_pid: str) -> list[dict]:
    """
    Return [{pid, filename, size_bytes}] for all DATA objects in the package.
    Skips metadata and resource-map objects.
    """
    data = _solr({
        "q":   f'resourceMap:"{resource_map_pid}"',
        "fq":  "formatType:DATA",
        "fl":  "id,fileName,size",
        "rows": 200,
    })
    docs = data.get("response", {}).get("docs", [])
    files = []
    for doc in docs:
        pid = doc.get("id", "")
        # fileName may be absent — fall back to last segment of pid
        fname = (doc.get("fileName") or "").strip()
        if not fname:
            fname = urllib.parse.unquote(pid.split("/")[-1].split("%2F")[-1]) or "data_file"
        size_bytes = int(doc.get("size") or 0)
        files.append({"pid": pid, "filename": fname, "size_bytes": size_bytes})
    return files


# ── Public helpers ──────────────────────────────────────────────────────────

def is_available(url: str = None, doi: str = None) -> bool:
    """
    Quick check — returns True if a DataONE package exists for this DOI.
    Used by the frontend downloadability checker (should be fast).
    """
    raw_doi = _extract_doi(url=url, doi=doi)
    if not raw_doi:
        return False
    rm, _ = _find_resource_map(raw_doi)
    return rm is not None


def get_file_list(url: str = None, doi: str = None) -> list[dict]:
    """Return the list of data files in the package (for display/preview)."""
    raw_doi = _extract_doi(url=url, doi=doi)
    if not raw_doi:
        return []
    rm, _ = _find_resource_map(raw_doi)
    if not rm:
        return []
    return _list_files(rm)


def download(url: str = None, doi: str = None,
             dest_root: str = "downloads") -> list[str]:
    """
    Download all data files from an Arctic Data Center dataset.
    Returns list of local file paths on success.
    Raises RuntimeError with a user-facing message on failure.
    """
    raw_doi = _extract_doi(url=url, doi=doi)
    if not raw_doi:
        raise ValueError(f"Cannot extract ADC DOI from url={url!r} doi={doi!r}")

    # Step 1 — find the package resource map
    rm_pid, title = _find_resource_map(raw_doi)
    if not rm_pid:
        raise RuntimeError(
            f"DataONE package not found for {raw_doi}. "
            f"Visit https://arcticdata.io to access the data manually."
        )

    # Step 2 — list data files
    files = _list_files(rm_pid)
    if not files:
        raise RuntimeError(
            f"No data files in ADC package for {raw_doi} "
            f"(resource map: {rm_pid})."
        )

    # Step 3 — download
    safe_id = re.sub(r"[^\w\-.]", "_", raw_doi)
    dest_dir = os.path.join(dest_root, f"adc_{safe_id}")
    os.makedirs(dest_dir, exist_ok=True)

    local_paths, skipped = [], 0
    for f in files:
        size_mb = f["size_bytes"] / 1_048_576 if f["size_bytes"] else 0
        if size_mb > _MAX_MB:
            print(f"     [SKIP] {f['filename']}  {size_mb:.0f} MB > {_MAX_MB:.0f} MB limit")
            skipped += 1
            continue

        pid_enc  = urllib.parse.quote(f["pid"], safe="")
        file_url = f"{_MN_BASE}/object/{pid_enc}"
        size_str = f"{size_mb:.1f} MB" if size_mb else "? MB"
        print(f"     Downloading: {f['filename']}  ({size_str})")
        try:
            path = download_file(file_url, dest_dir, f["filename"])
            local_paths.append(path)
        except Exception as exc:
            print(f"     [WARN] {f['filename']} — {exc}")
            skipped += 1

    if not local_paths:
        msg = (f"All {skipped} file(s) exceeded the {_MAX_MB:.0f} MB limit or failed."
               if skipped else
               f"No downloadable files found in ADC package for {raw_doi}.")
        raise RuntimeError(msg)

    print(f"     {len(local_paths)} file(s) saved to {dest_dir}")
    return local_paths
