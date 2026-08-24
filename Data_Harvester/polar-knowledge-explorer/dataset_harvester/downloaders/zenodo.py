"""
Zenodo downloader — uses Zenodo REST API to list and download all files
for a given record ID or record URL.

Zenodo is fully open (no auth required for public records).
"""

import gzip
import os
import re
import shutil
import requests

from .generic import download_file

_API_BASE = "https://zenodo.org/api"


def _record_id_from_url(url: str) -> str | None:
    m = re.search(r'zenodo\.org/(?:record|records)/(\d+)', url, re.IGNORECASE)
    if m:
        return m.group(1)
    # Resolver form, e.g. "https://doi.org/10.5281/zenodo.1234567" — the
    # resolver often stores the DOI link itself in the URL field and leaves
    # the doi field empty, so this needs to work on a bare URL too.
    return _record_id_from_doi(url)


def _record_id_from_doi(doi: str) -> str | None:
    # DOI like 10.5281/zenodo.1234567, possibly with a doi.org/ prefix
    m = re.search(r'zenodo\.(\d+)\b', doi, re.IGNORECASE)
    return m.group(1) if m else None


def get_files(record_id: str) -> list[dict]:
    """Return list of {filename, url, size} for a Zenodo record."""
    # Try v2 API first (newer records)
    for api_url in [
        f"{_API_BASE}/records/{record_id}",
        f"{_API_BASE}/deposit/depositions/{record_id}",
    ]:
        try:
            r = requests.get(api_url, timeout=15,
                             headers={"Accept": "application/json"})
            if r.status_code != 200:
                continue
            data = r.json()
            files = data.get("files", [])
            result = []
            for f in files:
                # v2 format
                if "links" in f:
                    result.append({
                        "filename": f.get("key", f.get("filename", "file")),
                        "url": f["links"].get("self", f["links"].get("download", "")),
                        "size": f.get("size", 0),
                    })
                else:
                    result.append({
                        "filename": f.get("filename", "file"),
                        "url": f.get("links", {}).get("download", ""),
                        "size": f.get("filesize", 0),
                    })
            if result:
                return result
        except Exception:
            continue
    return []


def download(url: str = None, doi: str = None,
             dest_root: str = "downloads") -> list[str]:
    """
    Download all files from a Zenodo record.
    Returns list of local file paths.
    """
    record_id = None
    if url:
        record_id = _record_id_from_url(url)
    if not record_id and doi:
        record_id = _record_id_from_doi(doi)
    if not record_id:
        raise ValueError(f"Cannot extract Zenodo record ID from url={url} doi={doi}")

    files = get_files(record_id)
    if not files:
        raise RuntimeError(f"No files found for Zenodo record {record_id}")

    dest_dir = os.path.join(dest_root, f"zenodo_{record_id}")
    os.makedirs(dest_dir, exist_ok=True)

    MAX_MB = 200.0
    TEXT_EXTS = {".csv", ".tsv", ".txt", ".tab"}

    def _accept(filename: str) -> bool:
        """Accept plain text files and gzipped text files."""
        fl = filename.lower()
        if fl.endswith(".gz"):
            inner = os.path.splitext(fl[:-3])[1]  # e.g. ".csv" from "data.csv.gz"
            return inner in TEXT_EXTS or fl.endswith(".tab.gz")
        return os.path.splitext(fl)[1] in TEXT_EXTS

    def _decompress_gz(gz_path: str) -> str:
        """Decompress .gz → inner file, remove .gz original, return new path."""
        out_path = gz_path[:-3]  # strip ".gz"
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"     Decompressed → {os.path.basename(out_path)}")
        return out_path

    local_paths = []
    for f in files:
        if not _accept(f["filename"]):
            ext = os.path.splitext(f["filename"].lower())[1]
            print(f"     [SKIP] {f['filename']} — not a text/CSV file (ext={ext})")
            continue
        size_mb = f["size"] / 1024 / 1024 if f["size"] else 0
        if size_mb > MAX_MB:
            print(f"     [SKIP] {f['filename']} is {size_mb:.0f} MB — exceeds {MAX_MB:.0f} MB limit")
            continue
        print(f"     Downloading: {f['filename']}  ({size_mb:.1f} MB)")
        path = download_file(f["url"], dest_dir, f["filename"])
        if path.endswith(".gz"):
            try:
                path = _decompress_gz(path)
            except Exception as e:
                print(f"     [WARN] gzip decompression failed: {e}")
        local_paths.append(path)

    if not local_paths:
        raise RuntimeError(f"No downloadable text/CSV files found (all skipped or too large)")
    return local_paths
