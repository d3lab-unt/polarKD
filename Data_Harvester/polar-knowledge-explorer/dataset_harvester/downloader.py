"""
Iteration 3 — Dataset Downloader

Routes each resolved dataset to the correct downloader based on repository.
Tracks all attempts in manifest.json so nothing is downloaded twice.
"""

import os
import sys

from resolver import ResolvedDataset
from downloaders.manifest import already_downloaded, record, summary
from downloaders import zenodo, pangaea, generic
from downloaders.auth_required import print_instructions

# Repos that need auth — we skip download but print instructions
_AUTH_REQUIRED = {
    "nsidc", "nasa_earthdata", "copernicus_cds",
    "copernicus_marine", "ecmwf", "esa", "jma", "ncar_rda",
}

# Repos where the URL is a landing page, not a direct file
_LANDING_PAGE_ONLY = {
    "uw_apl", "met_norway", "noaa", "noaa_ncei", "noaa_psl",
    "usgs", "argo", "gebco", "other",
}

DOWNLOADS_DIR = os.path.join(os.path.dirname(__file__), "downloads")


MAX_FILE_SIZE_MB = 200.0   # skip anything larger than this per file


def download_all(resolved_datasets: list[ResolvedDataset],
                 skip_large_mb: float = MAX_FILE_SIZE_MB) -> None:
    """
    Attempt to download every resolved dataset.
    Skips already-downloaded ones (manifest check).
    """
    total = len(resolved_datasets)
    downloaded_count = 0
    skipped_count = 0
    auth_count = 0
    failed_count = 0

    for i, d in enumerate(resolved_datasets, 1):
        print(f"\n[{i}/{total}] {d.canonical_name}")
        print(f"        Repo: {d.repository or 'unknown'}  |  {d.resolved_url or 'no URL'}")

        if not d.resolved_url:
            print("        [SKIP] No URL resolved")
            record(d.canonical_name, "skipped", "", [], "no URL resolved")
            skipped_count += 1
            continue

        # Already downloaded?
        cached = already_downloaded(d.canonical_name)
        if cached:
            print(f"        [SKIP] Already downloaded: {cached['local_files']}")
            skipped_count += 1
            continue

        repo = (d.repository or "").lower()

        # Auth-required repositories
        if repo in _AUTH_REQUIRED:
            print_instructions(repo, d.canonical_name)
            record(d.canonical_name, "auth_required", d.resolved_url, [],
                   f"requires auth for {repo}")
            auth_count += 1
            continue

        # Landing-page-only — can't auto-download, just record
        if repo in _LANDING_PAGE_ONLY:
            print(f"        [INFO] Landing page only — visit to download manually:")
            print(f"               {d.resolved_url}")
            record(d.canonical_name, "skipped", d.resolved_url, [],
                   "landing page — manual download required")
            skipped_count += 1
            continue

        # Try to download
        try:
            local_files = _download_one(d)
            record(d.canonical_name, "downloaded", d.resolved_url, local_files)
            print(f"        [OK] Saved {len(local_files)} file(s):")
            for f in local_files:
                size = os.path.getsize(f) / 1024 / 1024
                print(f"             {f}  ({size:.2f} MB)")
            downloaded_count += 1
        except Exception as e:
            print(f"        [FAIL] {e}")
            record(d.canonical_name, "failed", d.resolved_url, [], str(e))
            failed_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"  Downloaded:    {downloaded_count}")
    print(f"  Auth required: {auth_count}  (see instructions above)")
    print(f"  Skipped:       {skipped_count}  (landing pages or already done)")
    print(f"  Failed:        {failed_count}")
    print(f"  Files saved to: {DOWNLOADS_DIR}")
    print(f"{'='*60}")


def _download_one(d: ResolvedDataset) -> list[str]:
    url = d.resolved_url
    repo = (d.repository or "").lower()

    # Zenodo
    if repo == "zenodo" or "zenodo.org" in url:
        return zenodo.download(url=url, doi=d.doi, dest_root=DOWNLOADS_DIR)

    # PANGAEA
    if repo == "pangaea" or "pangaea.de" in url:
        return pangaea.download(url=url, doi=d.doi,
                                accession=d.accession, dest_root=DOWNLOADS_DIR)

    # Generic direct file URL (ends with a known extension)
    if _looks_like_direct_file(url):
        dest_dir = os.path.join(DOWNLOADS_DIR, "direct")
        path = generic.download_file(url, dest_dir)
        return [path]

    # Unrecognised — try generic anyway
    dest_dir = os.path.join(DOWNLOADS_DIR, "other")
    path = generic.download_file(url, dest_dir)
    return [path]


def _looks_like_direct_file(url: str) -> bool:
    extensions = {
        ".nc", ".nc4", ".hdf", ".hdf5", ".h5",
        ".csv", ".txt", ".tsv", ".zip", ".tar", ".tar.gz",
        ".tif", ".tiff", ".geotiff", ".json", ".xml",
    }
    path = url.split("?")[0].lower()
    return any(path.endswith(ext) for ext in extensions)
