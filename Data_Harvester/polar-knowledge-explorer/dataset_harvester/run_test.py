"""
Test runner for extractor on a single PDF - step-by-step output.
Usage: python dataset_harvester/run_test.py <pdf_path>
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

PDF = sys.argv[1] if len(sys.argv) > 1 else "Knowledge_graph/docs/essd-16-471-2024.pdf"

import pdfplumber
from extractor import (
    _extract_page_text, _read_pdf,
    _section_ranges, _SECTION_HEADING_RE, _SECTION_END_RE,
    extract_regex,
)

W = 70
SEP = "=" * W
DIV = "-" * W

print(f"\n{SEP}")
print(f"  TEST RUN -- {os.path.basename(PDF)}")
print(f"{SEP}\n")

# -- STEP 1: Column detection -------------------------------------------------
print(DIV)
print("  STEP 1 -- Column-aware page extraction")
print(DIV)

with pdfplumber.open(PDF) as pdf:
    total = len(pdf.pages)
    two_col_pages = []
    for i, page in enumerate(pdf.pages, start=1):
        words = page.extract_words()
        mid = page.width / 2
        left  = [w for w in words if w["x1"] < mid - 10]
        right = [w for w in words if w["x0"] > mid + 10]
        two_col = (len(left) >= 5 and len(right) >= 5
                   and (len(left) + len(right)) >= len(words) * 0.6)
        if two_col:
            two_col_pages.append(i)
        mark = "TWO-COL" if two_col else "single "
        print(f"  Page {i:>2}: [{mark}]  words={len(words):>3}  "
              f"left={len(left):>3}  right={len(right):>3}")

print(f"\n  Total pages : {total}")
print(f"  Two-column  : {len(two_col_pages)} pages  {two_col_pages}")

# -- STEP 2: Full text extraction ----------------------------------------------
print(f"\n{DIV}")
print("  STEP 2 -- Full text (column-aware)")
print(DIV)

t0 = time.time()
text = _read_pdf(PDF)
print(f"  Extracted {len(text):,} chars in {time.time()-t0:.2f}s")

# -- STEP 3: Section detection -------------------------------------------------
print(f"\n{DIV}")
print("  STEP 3 -- Section heading detection")
print(DIV)

section_matches = list(_SECTION_HEADING_RE.finditer(text))
end_matches = list(_SECTION_END_RE.finditer(text))
ranges = _section_ranges(text)

print(f"  Section headings found: {len(section_matches)}")
for m in section_matches:
    line = m.group(0).strip()[:60]
    char_pos = m.start()
    pct = char_pos / max(len(text), 1) * 100
    print(f"    [{pct:5.1f}%  char {char_pos:>6}]  \"{line}\"")

print(f"\n  Hard end markers: {len(end_matches)}")
for m in end_matches:
    print(f"    [char {m.start():>6}]  \"{m.group(0).strip()[:40]}\"")

print(f"\n  Section ranges: {len(ranges)}")
for s, e in ranges:
    heading_line = text[s:s+60].split("\n")[0].strip()
    print(f"    [{s:>6} -> {e:>6}]  ({(e-s):,} chars)  \"{heading_line[:45]}\"")

# -- STEP 4: Content inside data sections -------------------------------------
print(f"\n{DIV}")
print("  STEP 4 -- Content inside detected sections (preview)")
print(DIV)

for s, e in ranges:
    heading_line = text[s:s+80].split("\n")[0].strip()
    snippet = text[s:min(s+600, e)]
    lines = [l for l in snippet.splitlines() if l.strip()]
    print(f"\n  SECTION: \"{heading_line[:60]}\"")
    print(f"  Range   : chars [{s} -> {e}]  ({e-s:,} chars)")
    for l in lines[:12]:
        print(f"    {l[:80]}")
    if len(lines) > 12:
        print(f"    ... ({len(lines)} lines total in preview, {e-s:,} chars in section)")

# -- STEP 5: Regex extraction --------------------------------------------------
print(f"\n{DIV}")
print("  STEP 5 -- Regex pass results")
print(DIV)

refs = extract_regex(text)
used = [r for r in refs if r.used_in_study]
unused = [r for r in refs if not r.used_in_study]

print(f"\n  used_in_study=True  : {len(used)}")
print(f"  used_in_study=False : {len(unused)}")

print(f"\n  === USED IN STUDY (will be candidates for download) ===")
for r in used:
    ident = r.url or r.doi or r.accession or r.name
    print(f"    [{r.repository_hint or '?':22s}]  {ident[:65]}")

print(f"\n  === ALL REFS (used + secondary) ===")
for r in refs:
    ident = r.url or r.doi or r.accession or r.name
    flag = "USED" if r.used_in_study else "    "
    print(f"    {flag}  [{r.repository_hint or '?':22s}]  {ident[:55]}")
