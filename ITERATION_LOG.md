# Iteration Log — PolarKD Project

Tracks feature additions and changes across development iterations.

---

## Iteration 1 — PDF Dataset Reference Extractor
**Module:** `dataset_harvester/extractor.py`
- Regex + LLM-based extraction of dataset references from uploaded PDFs
- Parallel extraction via ThreadPoolExecutor

## Iteration 2 — Deduplication & URL Resolution
**Modules:** `dataset_harvester/deduplicator.py`, `dataset_harvester/resolver.py`
- Deduplication engine merges duplicate dataset mentions across PDFs
- Resolver maps dataset names to canonical URLs (PANGAEA, Zenodo, etc.)

## Iteration 3 — Dataset Downloader
**Module:** `dataset_harvester/downloader.py`, `dataset_harvester/downloaders/`
- PANGAEA tab-separated file downloader
- Zenodo CSV/text file downloader
- Generic downloader with extension filtering
- Manifest JSON for tracking download status per dataset

## Iteration 4 — Linux Server Migration & UI Enhancements
**Date:** 2026-06-17
**Files changed:** `Knowledge_graph/Code/frontend_light_v2.py`

### Infrastructure
- Migrated complete codebase from local Windows VS Code to Linux server (`ci-l-84cl144.is.unt.edu`)
- Project relocated from `/mnt/storage/midas/polarKD_Gowtham` → `/mnt/storage/salif/polarKD_Gowtham`
- Dataset extraction time: **~4 seconds** on server (vs. significantly longer on local machine)
- GitHub backup maintained at `gowthamvsn/polarKD_Gowtham`

### Feature: Collection Expansion
- When a PANGAEA dataset resolves as a **collection** (parent record with child datasets), the UI now shows a **"List Child Datasets"** button
- Clicking fetches child dataset names and links from the PANGAEA search API
- Falls back to downloading the collection ZIP and reading filenames from its central directory if the API returns no results
- Each child dataset is listed individually with a direct link

### Feature: First-10-Rows Preview
- After a successful "Download to disk", the UI now displays the **first 10 rows** of the downloaded file as an interactive table
- Supports PANGAEA tab-separated `.txt`/`.tsv`/`.tab` files (skips PANGAEA header comment lines)
- Supports `.csv` files
- Binary formats (NetCDF, HDF5) show file path only

---

## Upcoming (Todo — Week 5)
- Data redistribution: users see dataset names only, server downloads and visualizes on their behalf
- User dataset upload via UI
- Variable tabulation (location, time, variables with units) — Panoply-NASA format
