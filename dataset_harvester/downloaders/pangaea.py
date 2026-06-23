"""
PANGAEA downloader — downloads datasets from pangaea.de.

PANGAEA is fully open for public datasets.
Data is usually a tab-separated .txt file downloadable directly via:
  https://doi.pangaea.de/10.1594/PANGAEA.XXXXXX?format=textfile
"""

import os
import re
import zipfile
import requests

from .generic import download_file

_BASE = "https://doi.pangaea.de"


def _pangaea_id_from_url(url: str) -> str | None:
    # e.g. https://doi.pangaea.de/10.1594/PANGAEA.123456
    # or   https://www.pangaea.de/...PANGAEA.123456
    m = re.search(r'PANGAEA\.(\d+)', url, re.IGNORECASE)
    return m.group(1) if m else None


def _pangaea_id_from_doi(doi: str) -> str | None:
    m = re.search(r'1594/PANGAEA\.(\d+)', doi, re.IGNORECASE)
    return m.group(1) if m else None


def get_metadata(pangaea_id: str) -> dict:
    """Fetch dataset metadata via PANGAEA REST API."""
    try:
        r = requests.get(
            f"https://www.pangaea.de/api/find",
            params={"q": f"PANGAEA.{pangaea_id}", "count": 1, "format": "json"},
            timeout=15,
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                return results[0]
    except Exception:
        pass
    return {}


def download(url: str = None, doi: str = None, accession: str = None,
             dest_root: str = "downloads") -> list[str]:
    """
    Download a PANGAEA dataset as tab-separated text file.
    Returns list of local file paths.
    """
    pangaea_id = None
    if url:
        pangaea_id = _pangaea_id_from_url(url)
    if not pangaea_id and doi:
        pangaea_id = _pangaea_id_from_doi(doi)
    if not pangaea_id and accession:
        m = re.search(r'(\d+)', accession)
        pangaea_id = m.group(1) if m else None
    if not pangaea_id:
        raise ValueError(f"Cannot extract PANGAEA ID from url={url} doi={doi}")

    dest_dir = os.path.join(dest_root, f"pangaea_{pangaea_id}")
    os.makedirs(dest_dir, exist_ok=True)

    # Direct text file download
    download_url = f"{_BASE}/10.1594/PANGAEA.{pangaea_id}?format=textfile"
    filename = f"PANGAEA_{pangaea_id}.txt"

    print(f"     Downloading: {filename}")
    path = download_file(download_url, dest_dir, filename)

    # PANGAEA returns application/zip even for ?format=textfile — extract if so
    with open(path, "rb") as _f:
        magic = _f.read(2)
    if magic == b"PK":
        zip_path = path[:-4] + ".zip" if path.endswith(".txt") else path + ".zip"
        os.rename(path, zip_path)
        extracted = []
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                zf.extract(name, dest_dir)
                extracted.append(os.path.join(dest_dir, name))
        os.remove(zip_path)
        print(f"     Extracted {len(extracted)} file(s) from ZIP")
        return extracted

    return [path]
