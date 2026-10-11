import streamlit as st
import tempfile
import os
import re
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from extractor import extract_from_pdfs
from deduplicator import deduplicate
from resolver import resolve_one, ResolvedDataset

st.set_page_config(page_title="Dataset Harvester", layout="wide")
st.title("Arctic Dataset Harvester")
st.caption("Upload a research PDF → extract dataset references → check which ones are downloadable")

# --------------------------------------------------------------------------
# Downloadability checker
# --------------------------------------------------------------------------

_AUTH_REQUIRED = {
    "nsidc", "nasa_earthdata", "copernicus_cds", "copernicus_marine",
    "ecmwf", "esa", "jma", "ncar_rda",
}
_LANDING_PAGE_ONLY = {
    "uw_apl", "met_norway", "noaa", "noaa_ncei", "noaa_psl",
    "usgs", "argo", "gebco", "other",
}
_CSV_EXTS = {".csv", ".txt", ".tsv", ".tab"}


def _pangaea_id(r: ResolvedDataset):
    for s in [r.resolved_url or "", r.doi or "", r.accession or ""]:
        m = re.search(r'PANGAEA\.(\d+)', s, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _zenodo_id(r: ResolvedDataset):
    for s in [r.resolved_url or "", r.doi or ""]:
        m = re.search(r'zenodo\.org/(?:record[s]?/)?(\d+)|zenodo\.(\d+)', s, re.IGNORECASE)
        if m:
            return m.group(1) or m.group(2)
    return None


def check_downloadable(r: ResolvedDataset):
    """Returns (status, label, detail)"""
    if not r.resolved_url:
        return "none", "No URL", "Dataset not resolved"

    repo = (r.repository or "").lower()

    if repo in _AUTH_REQUIRED:
        return "auth", "Login required", f"Requires account on {repo}"

    if repo in _LANDING_PAGE_ONLY:
        return "page", "Homepage only", "No direct file — manual download"

    if repo == "pangaea" or "pangaea" in (r.resolved_url or "").lower():
        pid = _pangaea_id(r)
        if pid:
            try:
                resp = requests.head(
                    f"https://doi.pangaea.de/10.1594/PANGAEA.{pid}?format=textfile",
                    timeout=5, allow_redirects=True
                )
                if resp.status_code == 200:
                    return "yes", "Downloadable", "PANGAEA tab-separated file"
                else:
                    return "collection", "Collection", "Parent record — contains multiple datasets"
            except Exception:
                return "unknown", "Unknown", "Could not reach PANGAEA"

    if repo == "zenodo" or "zenodo" in (r.resolved_url or "").lower():
        zid = _zenodo_id(r)
        if zid:
            try:
                resp = requests.get(f"https://zenodo.org/api/records/{zid}", timeout=5)
                if resp.status_code == 200:
                    files = resp.json().get("files", [])
                    csv_files = [
                        f for f in files
                        if os.path.splitext(f.get("key", "").lower())[1] in _CSV_EXTS
                    ]
                    if csv_files:
                        return "yes", "Downloadable", f"Zenodo: {len(csv_files)} CSV/text file(s)"
                    return "no_csv", "No CSV files", "Zenodo record has no CSV/text files"
            except Exception:
                return "unknown", "Unknown", "Could not reach Zenodo"

    return "page", "Homepage only", "Landing page — no direct file"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

# Session-state cache: keyed by (filename, size, version) — bump version to invalidate old runs
_CACHE_VERSION = 2
if "harvester_cache" not in st.session_state or st.session_state.get("harvester_cache_v") != _CACHE_VERSION:
    st.session_state.harvester_cache = {}
    st.session_state.harvester_cache_v = _CACHE_VERSION

uploaded = st.file_uploader("Upload a research PDF", type=["pdf"])

if uploaded:
    st.divider()

    cache_key = (uploaded.name, uploaded.size, _CACHE_VERSION)

    if cache_key in st.session_state.harvester_cache:
        rows = st.session_state.harvester_cache[cache_key]
        st.success(f"Showing cached results for {uploaded.name} — pipeline not re-run.")

    else:
        def _resolve_and_check(dataset):
            """Resolve URL + check downloadability for one dataset (runs in a thread)."""
            r = resolve_one(dataset)
            status_key, label, detail = check_downloadable(r)
            return r, status_key, label, detail

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, uploaded.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # Step 1 — Extract
            with st.status("Step 1 — Extracting dataset references from PDF...", expanded=True) as step1:
                refs_by_source = extract_from_pdfs([pdf_path], use_llm=True)
                raw_count = sum(len(v) for v in refs_by_source.values())
                step1.update(label=f"Step 1 — Found {raw_count} raw references", state="complete", expanded=False)

            # Step 2 — Deduplicate
            with st.status("Step 2 — Deduplicating...", expanded=True) as step2:
                datasets = deduplicate(refs_by_source)
                step2.update(label=f"Step 2 — {raw_count} raw → {len(datasets)} unique datasets", state="complete", expanded=False)

        # Steps 3+4 — Resolve + check downloadability in parallel (outside tmpdir — no file I/O needed)
        rows = [None] * len(datasets)
        with st.status(
            f"Steps 3+4 — Resolving URLs and checking downloadability for {len(datasets)} datasets...",
            expanded=True,
        ) as step34:
            progress = st.progress(0, text="Starting…")
            done_count = 0

            with ThreadPoolExecutor(max_workers=10) as pool:
                future_to_idx = {pool.submit(_resolve_and_check, d): i for i, d in enumerate(datasets)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        r, status_key, label, detail = future.result()
                    except Exception as exc:
                        d = datasets[idx]
                        r = ResolvedDataset(
                            canonical_name=d.canonical_name, resolved_url=None,
                            repository=None, doi=d.doi, accession=d.accession,
                            notes=str(exc), resolution_method="unresolved",
                            mention_count=d.mention_count, sources=d.sources,
                            is_primary=d.is_primary,
                        )
                        status_key, label, detail = "unknown", "Error", str(exc)
                    rows[idx] = {
                        "Dataset Name": r.canonical_name,
                        "URL": r.resolved_url or "",
                        "Repository": r.repository or "unknown",
                        "Status": label,
                        "Detail": detail,
                        "_status_key": status_key,
                        "_is_primary": r.is_primary,
                        "_used_in_study": r.used_in_study,
                        "_mention_count": r.mention_count,
                    }
                    done_count += 1
                    progress.progress(done_count / len(datasets), text=f"{done_count}/{len(datasets)} processed")

            resolved_count = sum(1 for row in rows if row["URL"])
            downloadable_count = sum(1 for row in rows if row["_status_key"] == "yes")
            step34.update(
                label=f"Steps 3+4 — {resolved_count} resolved, {downloadable_count} downloadable",
                state="complete", expanded=False,
            )

        # Cache results so re-upload is instant
        st.session_state.harvester_cache[cache_key] = rows

    # --- Identify actual dataset candidates used in the paper ---
    actual_rows = [r for r in rows if r["_used_in_study"] or r["_is_primary"]]
    if not actual_rows and rows:
        best = max(rows, key=lambda r: (r["_is_primary"], r["_used_in_study"], r["_mention_count"]))
        best["_used_in_study"] = True
        best["_primary_how"] = best.get("_primary_how", "fallback")
        actual_rows = [best]

    llm_picked_primary = any(r["_is_primary"] for r in rows)
    if not llm_picked_primary and rows and not actual_rows:
        best = max(rows, key=lambda r: r["_mention_count"])
        best["_is_primary"] = True
        best["_primary_how"] = "heuristic"
    for r in rows:
        if "_primary_how" not in r:
            r["_primary_how"] = "llm" if r["_is_primary"] else ""

    # Summary stats
    primary_rows = actual_rows
    secondary_rows = [r for r in rows if r not in actual_rows]
    resolved_count = sum(1 for r in rows if r["URL"])
    downloadable_count = sum(1 for r in rows if r["_status_key"] == "yes")

    # Metrics
    st.subheader(f"Results — {len(rows)} datasets found in {uploaded.name}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total found", len(rows))
    c2.metric("Actual dataset candidates", len(actual_rows))
    c3.metric("Other refs", len(secondary_rows))
    c4.metric("URLs resolved", resolved_count)
    c5.metric("Ready to download", downloadable_count)

    st.divider()

    # --- Actual dataset card(s) ---
    st.markdown("### Active Dataset(s) Used")
    for p in primary_rows:
        how_badge = " <span style='font-size:0.8rem;color:#888'>(identified by mention frequency)</span>" if p["_primary_how"] == "heuristic" else ""
        if p["_primary_how"] == "fallback":
            how_badge = " <span style='font-size:0.8rem;color:#888'>(fallback candidate)</span>"
        st.markdown(
            f"""
<div style="border:2px solid #667eea; border-radius:10px; padding:1rem 1.5rem;
            background:#f0f2ff; margin-bottom:0.75rem;">
  <strong style="font-size:1.1rem">{p['Dataset Name']}</strong>{how_badge}<br>
  <span style="color:#555">Repository:</span> <code>{p['Repository']}</code>
  &nbsp;|&nbsp; <span style="color:#555">Status:</span> {p['Status']}<br>
  <span style="color:#555">Detail:</span> {p['Detail']}<br>
  {"<a href='" + p['URL'] + "' target='_blank' style='color:#667eea'>" + p['URL'] + "</a>" if p['URL'] else "<em style='color:#888'>URL not resolved</em>"}
</div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Tables — Role column always visible ---
    def render_table(data):
        display = []
        for row in data:
            display.append({
                "Role": "Primary" if row["_is_primary"] else "Secondary",
                "Dataset": row["Dataset Name"][:80] + ("…" if len(row["Dataset Name"]) > 80 else ""),
                "URL": row["URL"],
                "Repository": row["Repository"],
                "Status": row["Status"],
                "Detail": row["Detail"],
            })
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Role": st.column_config.TextColumn("Role", width="small"),
                "URL": st.column_config.LinkColumn("URL", display_text="Open"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "Detail": st.column_config.TextColumn("Detail", width="large"),
            },
        )

    downloadable_rows = [r for r in rows if r["_status_key"] == "yes"]
    tab1, tab2 = st.tabs([
        f"All datasets ({len(rows)})",
        f"Downloadable ({downloadable_count})",
    ])

    with tab1:
        render_table(rows)

    with tab2:
        if downloadable_rows:
            render_table(downloadable_rows)
        else:
            st.info("No directly downloadable datasets found in this paper.")
