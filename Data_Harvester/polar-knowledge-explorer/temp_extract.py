from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'dataset_harvester'))
import extractor, deduplicator
from pdfplumber import open as pdf_open

pdf_path = Path('Knowledge_graph/docs/essd-14-4901-2022.pdf')
with pdf_open(str(pdf_path)) as pdf:
    text = '\n'.join(page.extract_text() or '' for page in pdf.pages)

refs = extractor.extract_regex(text)
print('regex refs', len(refs))
for r in refs:
    print('REF|', r.name, '|', r.doi, '|', r.url, '|', r.accession, '|', r.repository_hint, '|', r.used_in_study, '|', r.source)

from deduplicator import deduplicate

dedup = deduplicate({'regex': refs})
print('DEDUP', len(dedup))
for d in dedup:
    print('DEDUP|', d.canonical_name, '|', d.doi, '|', d.url, '|', d.repository_hint, '|', d.mention_count, '|', d.used_in_study, '|', d.is_primary, '| raw_count', len(d.raw_refs))
