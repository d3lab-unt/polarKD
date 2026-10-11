"""
Generic HTTP downloader — streams any direct file URL to disk with retry.
Used as a fallback for any URL that isn't a recognised repository API.
"""

import os
import time
import requests

CHUNK_SIZE = 1024 * 256   # 256 KB
MAX_RETRIES = 3
RETRY_DELAY = 3           # seconds


def download_file(url: str, dest_dir: str, filename: str = None) -> str:
    """
    Download a single URL to dest_dir.
    Returns the local file path on success, raises on failure.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if not filename:
        filename = url.split("/")[-1].split("?")[0] or "download"

    dest_path = os.path.join(dest_dir, filename)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=60,
                              headers={"User-Agent": "arctic-dataset-harvester/1.0"}) as r:
                r.raise_for_status()
                # Try to get filename from Content-Disposition
                cd = r.headers.get("Content-Disposition", "")
                if "filename=" in cd:
                    cd_name = cd.split("filename=")[-1].strip().strip('"')
                    if cd_name:
                        dest_path = os.path.join(dest_dir, cd_name)

                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                if total:
                    print(f"     Downloaded {downloaded/1024/1024:.1f} MB")
                return dest_path
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"     Attempt {attempt} failed: {e} — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                raise
