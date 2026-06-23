"""
CLI runner — extraction + deduplication + resolution.

Usage:
  python run_extraction.py paper1.pdf paper2.pdf [--no-llm]
"""

import sys
import os
import json
import argparse
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "Knowledge_graph", ".env"))
load_dotenv()

from extractor import extract_from_pdfs
from deduplicator import deduplicate
from resolver import resolve
from downloader import download_all

def main():
    parser = argparse.ArgumentParser(description="Extract and resolve dataset references from PDFs")
    parser.add_argument("pdfs", nargs="+", help="PDF file paths")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM pass")
    parser.add_argument("--no-download", action="store_true", help="Skip download step")
    args = parser.parse_args()

    missing = [p for p in args.pdfs if not os.path.exists(p)]
    if missing:
        print(f"[error] Files not found: {missing}")
        sys.exit(1)

    # Step 1: Extract
    print("\n== STEP 1: EXTRACTING DATASET REFERENCES ==")
    refs_by_source = extract_from_pdfs(args.pdfs, use_llm=not args.no_llm)

    # Step 2: Deduplicate
    print("\n== STEP 2: DEDUPLICATING ==")
    datasets = deduplicate(refs_by_source)
    print(f"   {sum(len(v) for v in refs_by_source.values())} raw refs -> {len(datasets)} unique datasets")

    # Step 3: Resolve URLs
    print("\n== STEP 3: RESOLVING URLS ==")
    resolved = resolve(datasets)

    # --- Print results ---
    print(f"\n{'='*70}")
    print(f"  DATASETS FOUND: {len(resolved)}")
    print(f"{'='*70}")

    unresolved = []
    for i, r in enumerate(resolved, 1):
        status = "[OK]" if r.resolved_url else "[?]"
        print(f"\n{status} {i:>2}. {r.canonical_name}")
        if r.resolved_url:
            print(f"       URL:    {r.resolved_url}")
        if r.doi:
            print(f"       DOI:    {r.doi}")
        if r.accession:
            print(f"       Acc:    {r.accession}")
        print(f"       Repo:   {r.repository or 'unknown'}  |  via: {r.resolution_method}")
        print(f"       Mentioned {r.mention_count}x in: {', '.join(r.sources)}")
        if r.notes:
            print(f"       Note:   {r.notes}")
        if not r.resolved_url:
            unresolved.append(r.canonical_name)

    if unresolved:
        print(f"\n{'='*70}")
        print(f"  UNRESOLVED ({len(unresolved)}) — manual lookup needed:")
        for name in unresolved:
            print(f"    - {name}")

    # --- Save JSON ---
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "extraction_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in resolved], f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")

    # Step 4: Download
    if not args.no_download:
        print("\n== STEP 4: DOWNLOADING DATASETS ==")
        download_all(resolved)

if __name__ == "__main__":
    main()
