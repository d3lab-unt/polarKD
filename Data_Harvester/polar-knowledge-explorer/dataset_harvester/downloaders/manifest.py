"""
Download manifest — tracks every download attempt so we never re-download
and can resume interrupted sessions.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional


MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "downloads", "manifest.json")


def _load() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def already_downloaded(canonical_name: str) -> Optional[dict]:
    """Return manifest entry if dataset was already successfully downloaded."""
    data = _load()
    entry = data.get(canonical_name)
    if entry and entry.get("status") == "downloaded":
        return entry
    return None


def record(canonical_name: str, status: str, url: str,
           local_files: list[str], notes: str = ""):
    data = _load()
    data[canonical_name] = {
        "status": status,           # downloaded | failed | auth_required | skipped
        "url": url,
        "local_files": local_files,
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save(data)


def summary() -> dict:
    data = _load()
    counts = {"downloaded": 0, "failed": 0, "auth_required": 0, "skipped": 0}
    for entry in data.values():
        s = entry.get("status", "skipped")
        counts[s] = counts.get(s, 0) + 1
    return counts
