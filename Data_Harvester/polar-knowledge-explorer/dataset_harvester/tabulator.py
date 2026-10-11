"""
Panoply/NASA-style variable tabulation from paper PDFs.
Extracts: location, time range, and measured variables with units.
"""

from __future__ import annotations
import re
import json
import os
import time
from pathlib import Path

import pdfplumber


# ── Column-aware PDF extraction (mirrors extractor.py iter 9.5) ──────────────

def _words_to_text(words: list) -> str:
    if not words:
        return ""
    sorted_w = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines, cur_top, cur_line = [], None, []
    for w in sorted_w:
        if cur_top is None or (w["top"] - cur_top) > 5:
            if cur_line:
                lines.append(" ".join(x["text"] for x in cur_line))
            cur_line = [w]
            cur_top = w["top"]
        else:
            cur_line.append(w)
    if cur_line:
        lines.append(" ".join(x["text"] for x in cur_line))
    return "\n".join(lines)


def _extract_page_text(page) -> str:
    words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    if not words:
        return page.extract_text() or ""
    mid = page.width / 2
    left = [w for w in words if w["x1"] < mid - 10]
    right = [w for w in words if w["x0"] > mid + 10]
    span = [w for w in words if not (w["x1"] < mid - 10 or w["x0"] > mid + 10)]
    two_col = (len(left) >= 5 and len(right) >= 5
               and (len(left) + len(right)) >= len(words) * 0.6)
    if two_col:
        parts = [_words_to_text(span), _words_to_text(left), _words_to_text(right)]
        return "\n".join(p for p in parts if p)
    return _words_to_text(words)


def _read_pdf(path) -> str:
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = [_extract_page_text(p) for p in pdf.pages]
            return "\n".join(p for p in pages if p)
    except Exception:
        return ""


# ── Physical unit whitelist ───────────────────────────────────────────────────
# Only patterns that represent real measured quantities pass through.

_UNIT_PATTERNS = [
    # Temperature (no standalone K — too many false positives with 'k' as count/constant)
    r"°[CF]",
    # Salinity / conductivity
    r"PSU", r"psu", r"PSS", r"g\s*/\s*kg",
    r"µS\s*/\s*cm", r"mS\s*/\s*cm",
    # Molar concentration (allow optional / separator: µmol/L or µmol L)
    r"µmol\s*/?\s*(?:L|kg|m[²2³3])?(?:\s*[⁻\-]1)?",
    r"mmol\s*/?\s*(?:L|kg|m[²2³3])?(?:\s*[⁻\-]1)?",
    r"nmol\s*/?\s*(?:L|kg|m[²2³3])?(?:\s*[⁻\-]1)?",
    r"pmol\s*/?\s*(?:L|kg)?(?:\s*[⁻\-]1)?",
    # Mass concentration
    r"mg\s*/?\s*(?:L|kg|g|m[23²³])?(?:\s*[⁻\-]1)?",
    r"µg\s*/?\s*(?:L|kg|m[23²³])?(?:\s*[⁻\-]1)?",
    r"ng\s*/?\s*(?:L|kg)?(?:\s*[⁻\-]1)?",
    # Dimensionless ratios
    r"ppm", r"ppb",
    r"%", r"‰",
    # Carbon mass
    r"PgC", r"TgC", r"GtC", r"Pg\s*C", r"Tg\s*C", r"Gt\s*C",
    r"mol\s*C\s*/?\s*m[23²³]",
    # Mass / volume (ice, water budget)
    r"Gt", r"Pg", r"Tg",
    r"km[³3]", r"km[²2]",
    # Pressure
    r"hPa", r"kPa", r"dbar", r"bar", r"mbar", r"µatm", r"matm", r"atm",
    # Length / depth / sea level
    r"km", r"m", r"cm", r"mm",
    r"mm\s*(?:SLE|w\.?e\.?)",   # sea level equivalent, water equivalent
    r"m\s*(?:SLE|w\.?e\.?)",
    r"cm\s*(?:SLE|w\.?e\.?)",
    # Velocity / rate — require explicit / separator or ⁻¹ notation
    r"m\s*/\s*s(?:\s*[⁻\-]1)?", r"m\s+s\s*[⁻\-]1",
    r"cm\s*/\s*s(?:\s*[⁻\-]1)?", r"cm\s+s\s*[⁻\-]1",
    r"m\s*/\s*yr(?:\s*[⁻\-]1)?", r"m\s+yr\s*[⁻\-]1",
    r"cm\s*/\s*yr(?:\s*[⁻\-]1)?", r"cm\s+yr\s*[⁻\-]1",
    r"mm\s*/\s*yr(?:\s*[⁻\-]1)?", r"mm\s+yr\s*[⁻\-]1",
    # Energy / heat flux
    r"W\s*/\s*m[2²]", r"MJ\s*/\s*m[2²]",
    r"ZJ",
    # Ocean transport (uppercase only — avoid PW/TW ambiguity with acronyms)
    r"Sv",
    # Flux rates
    r"mol\s*/?\s*m[2²]\s*/?\s*(?:yr|d|s)(?:\s*[⁻\-]1)?",
    r"µmol\s*/?\s*m[2²]\s*/?\s*(?:yr|d|s)(?:\s*[⁻\-]1)?",
    r"mmol\s*/?\s*m[2²]\s*/?\s*(?:yr|d|s)(?:\s*[⁻\-]1)?",
    r"g\s*/?\s*m[2²]\s*/?\s*(?:yr|d)?",
    r"µg\s*/?\s*m[2²]\s*/?\s*(?:yr|d)?",
    r"mg\s*/?\s*m[2²]\s*/?\s*(?:yr|d)?",
    r"kg\s*/?\s*m[2²]\s*/?\s*(?:yr|d)?",
    # Time rates
    r"yr[⁻\-]?1", r"/\s*yr", r"d[⁻\-]?1", r"/\s*d",
    # Paleoclimate / geochronology time units
    r"ka(?:\s*BP)?", r"Ma(?:\s*BP)?", r"kyr(?:\s*BP)?",
    # Velocity: m a⁻¹ / m a-1 (meters per annum, common in glaciology)
    r"m\s+a[⁻\-]1", r"km\s+a[⁻\-]1",
    # Water quality
    r"NTU", r"FTU",
    # Radioactivity
    r"Bq\s*/?\s*(?:m[23³]|kg|L)",
    # Isotope notation
    r"‰\s*VSMOW", r"‰\s*VPDB",
]

_UNIT_RE = re.compile(
    r"^(?:" + "|".join(f"(?:{p})" for p in _UNIT_PATTERNS) + r")$",
    re.UNICODE,  # NO IGNORECASE — scientific units are case-sensitive (Tg ≠ TG)
)


def _valid_unit(s: str) -> bool:
    return bool(_UNIT_RE.match(s.strip()))


# ── Patterns ──────────────────────────────────────────────────────────────────

# "Variable Name (unit)" or "acronym [unit]"
# - names: lowercase allowed, min 3 chars, [ -] only (no newlines in name)
# - unit content: max 30 chars, no newlines
_VAR_RE = re.compile(
    r"([A-Za-z][A-Za-z₀-₉₂₃₄]{2,}(?:[ \-][A-Za-z₀-₉₂₃₄]+){0,5})"
    r"\s*[\(\[]"
    r"([^\)\]\n]{1,30})"
    r"[\)\]]",
    re.MULTILINE,
)

# Words that are never variable names
_STOP_NAMES = frozenset({
    # Articles / prepositions / common quantifiers / adverbs
    "the", "and", "for", "but", "not", "are", "was", "were", "has", "had",
    "see", "note", "all", "each", "both", "from", "than", "with",
    "this", "that", "they", "their", "which", "when", "where", "here",
    "enough", "about", "within", "between", "around", "roughly", "approximately",
    "almost", "over", "under", "above", "below", "more", "less", "much",
    "just", "only", "even", "still", "yet", "already", "often", "sometimes",
    "many", "most", "some", "any", "few", "other", "another", "same",
    # Verb forms that describe a measurement action, not a variable
    "thus", "also", "shown", "used", "using", "based", "given", "defined",
    "expressed", "denoted", "calculated", "measured", "estimated", "derived",
    "plotted", "listed", "described", "indicated", "referred", "compared",
    "obtained", "computed", "observed", "modelled", "modeled", "simulated",
    # Adjectives describing size/magnitude (never variable names alone)
    "thin", "thick", "short", "long", "narrow", "wide", "shallow", "deep",
    "small", "large", "high", "low", "fast", "slow", "dense", "sparse",
    "upper", "lower", "inner", "outer", "central", "lateral",
    # Figure / table formatting words
    "fig", "figure", "table", "panel", "code", "label", "axis", "bars",
    "box", "cross", "cover", "line", "circle", "colour", "color",
    # Generic nouns / adjectives that appear near units but aren't variables
    "loss", "gain", "change", "error", "response", "carrier", "river",
    "maximum", "minimum", "million", "inches", "massive", "fissile",
    "suggests", "extension", "identifier",
    # Too generic measurement-adjacent words
    "content", "contents", "tests", "threshold", "species", "interval",
    "value", "values", "range", "ranges", "region", "regions",
    "explanatory", "freshened", "depths", "standard", "bottom", "top",
    "dataset", "datasets",
    # Statistical / methodological terms
    "variation", "variance", "percentage", "fraction", "proportion",
    "difference", "differences", "anomaly", "anomalies", "average",
    "typical", "relative", "decline", "reduction", "increase",
    # Adjective compounds / descriptive phrases masquerading as variables
    "low-salinity", "number", "samples", "clasts", "assemblage",
    "surface", "waters", "explanatory",
    # Generic geographic/topographic nouns — rejected as STANDALONE names only
    # (moved to _GEO_NOUNS below; NOT used in first/last-word checks so that
    # compound variable names like "ice thickness", "glacier velocity" are kept)
    "fjord", "coast", "peninsula",
    "summit", "mount", "mountain", "ridge", "peak", "nunatak",
    "plateau", "valley", "divide",
    # Adjectives that start descriptive phrases
    "although", "despite", "relative", "freshening", "warming",
    "tropical", "compiled", "typical",
})

# Geographic/topographic terms that are OK as part of a compound variable name
# (e.g. "ice thickness", "glacier velocity") but not as a standalone name.
# Only used for exact-match rejection — NOT checked as first/last word.
_GEO_NOUNS = frozenset({
    "glacier", "basin", "island", "shelf", "bay",
    "ice", "sheet", "outlet", "tributary", "channel",
    "slope", "range",
})

_YEAR_RANGE_RE = re.compile(
    r"\b(1[89]\d{2}|20[0-3]\d)\s*[–\-to]+\s*(1[89]\d{2}|20[0-3]\d)\b"
)

_SINGLE_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")

# Same as _YEAR_RANGE_RE but capturing the separator so we can tell a spelled
# "to"/"through" (a stated period, e.g. "1950 to 2021") from a bare dash.
# Not-preceded/followed-by-digit instead of \b: PDF text often runs words into
# numbers ("winter2019-2020"), where \b (letter|digit boundary) would not match.
_YEAR_RANGE_SEP_RE = re.compile(
    r"(?<!\d)(1[89]\d{2}|20[0-3]\d)\s*([–\-]|to|through|until|thru|and)\s*"
    r"(1[89]\d{2}|20[0-3]\d)(?!\d)"
)

# A year-range that is really a journal citation's "volume, pages" (e.g.
# "Geophys. Res. Lett., 41, 2011–2018") or explicit page markers. Detected by the
# text immediately BEFORE the first year.
_CITATION_PAGE_RE = re.compile(
    r"(?:,\s*\d{1,4}\s*,|\bpp?\.\s*\d{0,4}|\bvol\.?\s*\d{1,4}\s*,?|\bno\.?\s*\d{1,4}\s*,?)\s*$",
    re.IGNORECASE,
)

# The paper's own running self-citation footer, e.g.
# "Earth Syst. Sci. Data, 14, 4901-4921, 2022" — a page range (too big to be a
# year, so _YEAR_RANGE_SEP_RE never sees it) immediately followed by the
# publication year. Only the single-year fallback below needs this: the
# journal name/volume/page-range portion doesn't look like a year-range so
# _CITATION_PAGE_RE (anchored on the FIRST year of a range) never fires on it.
_CITATION_YEAR_TAIL_RE = re.compile(r"\d{2,5}\s*[–\-]\s*\d{2,5}\s*,\s*$")

# Same idea but for the bare DOI-slug form with no comma at all, e.g.
# ".../essd-14-4901-2022" — a number glued to the year by a lone hyphen.
# Only fires when the glued number is NOT itself a plausible year, so a real
# hyphen-joined study range ("1990-1991") is never caught by this.
_TAIL_NUM_DASH_RE = re.compile(r"(?<!\d)(\d{1,6})\s*[–\-]\s*$")


def _is_pagenum_glued_year(text: str, start: int) -> bool:
    m = _TAIL_NUM_DASH_RE.search(text[max(0, start - 12):start])
    if not m:
        return False
    n = int(m.group(1))
    return not (1800 <= n <= 2039)

# Substrings (matched WITHOUT word boundaries because PDF text often glues words
# together, e.g. "thisstudy") that mark a range as the study's OWN period.
_STUDY_CUE = ("this study", "thisstudy", "study period", "studyperiod",
              "our study", "ourstudy", "we use", "weuse", "we analy", "weanaly",
              "we simulate", "conclusion", "in this")

# Distinctive observation-window cues — deliberately NARROW (expedition/season/
# month, plus "during"). Broad words like "from"/"over"/"between"/"study" are
# excluded on purpose: they fire on climate-trend ranges and citations ("per
# decade over 1968-2022", "CASES Study in 2003-2004") and would create false
# study periods. A singly-stated range next to one of these still counts.
_PERIOD_CUE = ("during", "winter", "summer", "spring", "autumn", "expedition",
               "campaign", "cruise", "deployed", "deployment", "fieldwork",
               "overwinter", "collected", "sampled", "january", "february",
               "march", "april", "june", "july", "august", "september",
               "october", "november", "december")


def _extract_time_range(text: str) -> tuple[int | None, int | None, str]:
    """Pick the study/data time window from a paper.

    Strategy (see validation 2026-07-07): the old min/max-over-all-year-ranges rule
    leaked citation years and journal page ranges into the answer. Instead we:
      1. drop year-ranges that are journal "volume, pages" citations;
      2. count the remaining plausible ranges — a study's own data period is stated
         repeatedly, so the most-frequent exact (start, end) pair wins;
      3. break ties toward a range flagged as the study's own, then the wider span.
    If no plausible range survives we return (None, None) rather than guess, so
    paleo / single-year / undated papers correctly get no range.
    """
    from collections import Counter
    counts: Counter = Counter()
    study_pairs: set[tuple[int, int]] = set()
    period_pairs: set[tuple[int, int]] = set()
    dropped = 0
    for m in _YEAR_RANGE_SEP_RE.finditer(text):
        a, b = int(m.group(1)), int(m.group(3))
        lo, hi = min(a, b), max(a, b)
        if hi - lo > 130 or hi < 1850:          # implausible for a data window
            continue
        before = text[max(0, m.start() - 45):m.start()]
        if _CITATION_PAGE_RE.search(before):    # journal volume/pages, not years
            dropped += 1
            continue
        counts[(lo, hi)] += 1
        ctx = (before + text[m.end():m.end() + 15]).lower()
        if any(c in ctx for c in _STUDY_CUE):
            study_pairs.add((lo, hi))
        if any(c in ctx for c in _PERIOD_CUE):
            period_pairs.add((lo, hi))
    # A real study period is stated more than once OR sits next to a study/period
    # cue. A single, uncued range is almost always bibliography scatter — don't guess.
    eligible = [p for p in counts
                if counts[p] >= 2 or p in study_pairs or p in period_pairs]
    if not eligible:
        return None, None, (f"no repeated/cued range "
                            f"({len(counts)} lone range(s), {dropped} page-cite dropped)")

    def rank(pair: tuple[int, int]):
        lo, hi = pair
        cue = (2 if pair in study_pairs else 0) + (1 if pair in period_pairs else 0)
        return (counts[pair] + cue, hi - lo)

    best = max(eligible, key=rank)
    if best in study_pairs:
        tag = "study-cued"
    elif counts[best] >= 2:
        tag = f"most frequent x{counts[best]}"
    else:
        tag = "period-cued"
    return best[0], best[1], f"{tag}; {len(counts)} distinct range(s), {dropped} page-cite dropped"


_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?"
    r"(abstract|introduction|study\s+(?:area|site|region)|"
    r"data(?:\s+and\s+methods?)?|methods?(?:\s+and\s+materials?)?|"
    r"materials?\s+and\s+methods?|measurements?|observations?|"
    r"datasets?|variables?|parameters?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_REF_RE = re.compile(
    r"^\s*(?:References?|Bibliography|Works\s+Cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _target_sections(text: str) -> tuple[str, bool]:
    """Returns (section_text, used_fallback).
    used_fallback is True when no data/methods section headers were found
    and we fell back to text[:10000] (abstract/intro region).
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return text[:10000], True
    target_words = {"study", "data", "method", "material", "measurement",
                    "observation", "variable", "parameter", "abstract"}
    spans = []
    for i, m in enumerate(matches):
        if any(t in m.group(1).lower() for t in target_words):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else min(start + 8000, len(text))
            spans.append(text[start:end])
    if spans:
        return "\n".join(spans)[:10000], False
    return text[:10000], True


def _trim_refs(text: str) -> str:
    m = _REF_RE.search(text)
    return text[:m.start()] if m else text


# Section headers that mark the end of the introduction — study area, methods,
# results, discussion, conclusions. Anything from these onward is citation-heavy
# (prior work, comparison regions) and is NOT read for location detection.
_INTRO_END_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?"
    r"(study\s+(?:area|site|region)|"
    r"data(?:\s+and\s+methods?)?|methods?(?:\s+and\s+materials?)?|"
    r"materials?\s+and\s+methods?|measurements?|observations?|"
    r"datasets?|variables?|parameters?|results?|discussion|conclusions?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _title_abstract_intro(text: str) -> str:
    """Title + abstract + introduction ONLY — the only place a paper reliably
    states its own study location. Methods/Results/Discussion/References cite
    many other regions (prior studies, comparisons) and cause location leaks
    (e.g. "Canada"/"Alaska" on an Antarctic-focused paper), so we deliberately
    don't read past the introduction for location detection."""
    end_match = _INTRO_END_RE.search(text)
    if end_match:
        return text[:end_match.start()]
    # No structured section headers found — fall back to an early window
    # (title/abstract/intro reliably fall within a paper's first ~4000 chars).
    return text[:4000]


# ── Polar keyword location search (faster + more reliable than NER alone) ────
# IMPORTANT: longer/more-specific patterns MUST appear before shorter ones within
# each group so re.finditer always matches the longest form (e.g. "Greenland Sea"
# before bare "Greenland", "Antarctica" before "Antarctic").
_POLAR_PLACE_RE = re.compile(
    r"\b("
    # Compound ice sheets / shelves first (before bare continent names)
    r"Greenland\s+Ice\s+Sheet|Antarctic\s+Ice\s+Sheet|"
    r"West\s+Antarctica|East\s+Antarctica|"
    r"Antarctic\s+Peninsula|"
    r"Thwaites\s+Glacier|Pine\s+Island\s+Glacier|"
    r"Lambert\s+Glacier|Totten\s+Glacier|"
    r"Filchner.Ronne|Filchner\s+Ice\s+Shelf|Ronne\s+Ice\s+Shelf|"
    r"Shackleton\s+Range|Transantarctic\s+Mountains|"
    r"Ellsworth\s+Mountains|Heritage\s+Range|"
    # Antarctic ocean sectors (before bare "Antarctica")
    r"Weddell\s+Sea|Ross\s+Sea|Amundsen\s+Sea|Bellingshausen\s+Sea|"
    r"Prydz\s+Bay|Queen\s+Maud\s+Land|Marie\s+Byrd\s+Land|Victoria\s+Land|"
    # Continent names (after compounds)
    r"Antarctica|"          # noun form — matches "Antarctica" but not "Antarctic"
    r"Antarctic(?!a)|"      # adjective form — only if "Antarctica" didn't match
    # Arctic Ocean subdivisions (before bare "Arctic")
    r"Greenland\s+Sea|Barents\s+Sea|Kara\s+Sea|Laptev\s+Sea|"
    r"East\s+Siberian\s+Sea|Chukchi\s+Sea|Beaufort\s+Sea|Lincoln\s+Sea|"
    r"Norwegian\s+Sea|Bering\s+Sea|Hudson\s+Bay|Baffin\s+Bay|"
    r"Davis\s+Strait|Fram\s+Strait|Denmark\s+Strait|"
    r"Canadian\s+Arctic|Northwest\s+Passage|Canada|"
    r"Russia|"
    # Arctic continent/region (after ocean subdivisions)
    r"Arctic|"
    # Greenland specifics (after "Greenland Sea" and "Greenland Ice Sheet")
    r"Greenland|"
    # Svalbard / Norway
    r"Svalbard|Spitsbergen|Hornsund|Isfjorden|Kongsfjorden|"
    r"Ny-?[ÅA]lesund|Longyearbyen|Nordaustlandet|Edgeøya|"
    # Canada / Alaska
    r"Alaska|Baffin\s+Island|Ellesmere\s+Island|Banks\s+Island|"
    r"Victoria\s+Island|"
    # Siberia / Russia
    r"Siberia|Novaya\s+Zemlya|Franz\s+Josef\s+Land|Severnaya\s+Zemlya|"
    # Greenland coasts / fjords
    r"Ilulissat|Jakobshavn|Kangerdlugssuaq|Helheim|Sermeq|"
    r"Disko\s+(?:Bay|Island)|"
    # Antarctic stations
    r"McMurdo|Concordia|Halley|Dome\s+(?:C|Fuji|A)|"
    # Sub-polar / mountain glaciers
    r"Himalayas|Hindu\s+Kush|Karakoram|Tibetan\s+Plateau|Patagonia|"
    r"Andes|Alps|Caucasus"
    r")\b",
    re.IGNORECASE,
)


# A place name that sits at the tail of an author-affiliation address, e.g.
# "Geophysical Institute, Fairbanks, Alaska." — an institution word, then a city,
# then our place. This is a bibliography address, not the study region. Kept tight
# (requires the "…, City," shape right before the place) so it won't drop a real
# study place that merely follows a "University"/"Institute" mention in prose.
_AFFILIATION_RE = re.compile(
    r"(?:Institute|University|Department|Laborator|Observatory|College|"
    r"Academy|Faculty|Division|Survey)[^.]{0,30},\s*[A-Za-z.\-]+,\s*$",
    re.IGNORECASE,
)

# Same address shape but only ONE comma deep — catches the CITY itself in
# "Alfred Wegener Institute ..., Bremerhaven, Germany" (NER tags "Bremerhaven"
# as its own GPE entity, sitting right after the institute name with no
# second comma yet for _AFFILIATION_RE to anchor on).
_AFFIL_CITY_RE = re.compile(
    r"(?:Institute|University|Department|Laborator|Observatory|College|"
    r"Academy|Faculty|Division|Survey)[^.]{0,30},\s*$",
    re.IGNORECASE,
)


def _places_keywords(text: str) -> list[str]:
    """Fast keyword-based polar location extraction — more reliable than NER for place names."""
    body = _trim_refs(text)[:80000]
    seen: set[str] = set()
    out: list[str] = []
    for m in _POLAR_PLACE_RE.finditer(body):
        val = m.group(1).strip()
        # Skip places that are the tail of an author-affiliation address.
        if _AFFILIATION_RE.search(body[max(0, m.start() - 45):m.start()]):
            continue
        key = re.sub(r"\s+", " ", val).strip()
        # Normalise "Antarctic" → "Antarctica" when both might appear
        canonical = {"Antarctic": "Antarctica"}.get(key, key)
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            out.append(canonical)
        if len(out) >= 8:
            break
    return out


# ── LLM variable extraction (pass 3 fallback) ────────────────────────────────

_VARS_SYSTEM_PROMPT = """\
You are a scientific data extraction assistant for polar and cryosphere research papers.
Your job is to extract EVERY measurable quantity that is reported, measured, modelled, or sampled.

Return a JSON array ONLY — no prose before or after.
Each object: {"name": "<quantity name>", "unit": "<unit as it appears in the paper, or '-' if dimensionless>"}

INCLUDE all physical and environmental quantities:
- Atmospheric: temperature, pressure, wind speed, humidity, radiation, albedo, cloud fraction
- Cryosphere: ice thickness, ice velocity, mass balance, melt rate, snow depth, sea ice extent/concentration, ice discharge, accumulation rate, firn density
- Ocean: salinity, sea level, ocean temperature, currents, wave height
- Geochemical / geochronology (IMPORTANT — these appear in paleoclimate and dating papers):
  * Cosmogenic nuclide concentrations: 10Be, 26Al, 36Cl, 3He, 21Ne — unit is typically atoms/g or 10^n atoms/g
  * Isotope ratios: d18O, d13C, dD — unit is ‰
  * Radiocarbon age (14C age) — unit ka or yr BP
  * Exposure age — unit ka or yr
  * Burial age, production rate, erosion rate, denudation rate
  * Beryllium, Aluminum, Chlorine, Helium concentrations
- Glaciology: bedrock elevation, surface elevation, ice sheet thickness, grounding line position, retreat rate, advance rate
- General: concentration, flux, volume, area, distance, depth, elevation, velocity

ALSO INCLUDE derived quantities, indices, ratios, and model outputs.
If a unit is ambiguous or not shown in the text, skip that variable entirely — only include variables whose unit is clearly stated or is conventionally fixed (e.g. ‰ for isotope ratios, atoms/g for cosmogenic nuclides).
For dimensionless quantities (albedo, reflectance, ratio, fraction) use "-".

EXCLUDE ONLY: model names (WRF, ROMS, CESM), instrument brand names, author names, figure panel labels (a, b, c), pure geographic place names.

Return [] ONLY if the text is entirely non-scientific (e.g. acknowledgements, references list, boilerplate).
Output: JSON array only.\
"""


def _clean_for_llm(text: str) -> str:
    """Remove PDF font-encoding artifacts and normalise whitespace for LLM input."""
    # Replace (cid:XX) artifacts — these are undecodable glyph IDs from PDF fonts,
    # typically covering Greek letters, superscripts, subscripts, and special symbols.
    # Replacing with a space preserves word boundaries without confusing the LLM.
    text = re.sub(r'\(cid:\d+\)', ' ', text)
    # Collapse runs of whitespace/newlines to single spaces so the LLM sees prose
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _vars_llm(text: str) -> list[dict]:
    """LLM pass: extract variables from prose text. Returns [{name, unit}]."""
    # Load .env if keys not already in environment
    if not os.getenv("AZURE_OPENAI_KEY"):
        try:
            from dotenv import load_dotenv
            from pathlib import Path as _P
            _env = _P(__file__).parent.parent / "Knowledge_graph" / ".env"
            load_dotenv(_env)
        except Exception:
            pass

    # Clean cid-artifacts BEFORE trimming refs so garbled text doesn't hide content
    text_clean = _clean_for_llm(_trim_refs(text))
    # Increased from 6000 → 10000 to cover more of the paper per call
    text_trimmed = text_clean[:10000]
    if len(text_trimmed) < 150:
        return []
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": _VARS_SYSTEM_PROMPT},
                {"role": "user",   "content": text_trimmed},
            ],
            temperature=0,
            max_tokens=1500,
        )
        raw = resp.choices[0].message.content or ""
        print(f"[tabulator]   LLM raw   : {raw[:120].strip()!r}{'…' if len(raw) > 120 else ''}")
    except Exception as e:
        print(f"[tabulator]   LLM error : {e}")
        return []

    # Parse — tolerate markdown fences and extra prose
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except Exception:
            return []

    # Standalone coordinate/methodological terms that aren't environmental variables
    _LLM_META_NAMES = frozenset({
        "latitude", "longitude", "elevation", "altitude",
        "sample volume", "sample size", "sample weight",
        "spatial resolution", "temporal resolution", "pixel size",
        # Aircraft/platform navigation variables — not environmental measurements
        "flight altitude", "aircraft altitude", "aircraft pitch angle", "aircraft roll angle",
        "pitch angle", "roll angle", "heading",
        # Solar geometry — inputs, not outputs
        "solar zenith angle", "solar azimuth angle", "zenith angle",
        # Universal physical constants — never a study variable
        "gravitational acceleration", "gravity", "gas constant",
    })

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        unit = (item.get("unit") or "").strip()
        # Allow "-" / "dimensionless" as valid units for albedo, reflectance, fraction, etc.
        if not unit:
            unit = "-"
        if not name or len(name) < 2 or len(name) > 65 or len(unit) > 30:
            continue
        name_lower = name.lower()
        first_word = name_lower.split()[0]
        if first_word in _STOP_NAMES or name_lower in _STOP_NAMES:
            continue
        if name_lower in _LLM_META_NAMES:
            continue
        out.append({"name": name, "unit": unit})
    return out


# ── spaCy NER (optional) ──────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = False
    return _nlp if _nlp else None


# reject spaCy GPE false positives that are really cited authors. Checked
# against EVERY occurrence of the candidate in the document, not just the one
# spaCy happened to tag — column-joined PDFs scatter the same surname across
# many contexts and only some are citations.
_AUTHOR_CITE_RE = re.compile(
    r'^\s*(?:'
    r'et\s*al'                          # "Krumpen et al"
    r'|,?\s*[A-Z]\.'                    # "Krumpen, T." (initials)
    r'|\(\s*(?:19|20)\d{2}'             # "Krumpen (2021)"
    r'|,?\s*personal\s*communica'       # "Hutchings, personal communication"
    r'|,?\s*pers\.?\s*comm'             # "Hutchings, pers. comm."
    r'|\s+and\s+[A-Z][a-z]+,?\s*\(?\s*(?:19|20)\d{2}'  # "Krumpen and Hutchings (2020)"
    r')',
    re.IGNORECASE,
)


def _is_author_citation(name: str, doc_text: str) -> bool:
    """True if any occurrence of `name` in the document is in a citation context."""
    for m in re.finditer(re.escape(name), doc_text):
        if _AUTHOR_CITE_RE.match(doc_text[m.end():m.end() + 35]):
            return True
    return False


def _places_ner(text: str) -> list[str]:
    nlp = _get_nlp()
    if not nlp:
        return []
    doc = nlp(_trim_refs(text)[:50000])
    seen: set[str] = set()
    out: list[str] = []
    for ent in doc.ents:
        if ent.label_ not in ("GPE", "LOC"):
            continue
        val = ent.text.strip()
        # Skip places that are the tail of an author-affiliation address, e.g.
        # "Alfred Wegener Institute, Bremerhaven, Germany" — same guard as
        # _places_keywords; NER has no polar-place allowlist to filter these out.
        _before_ent = doc.text[max(0, ent.start_char - 45):ent.start_char]
        if _AFFILIATION_RE.search(_before_ent) or _AFFIL_CITY_RE.search(_before_ent):
            continue
        _NER_STOP = frozenset({
            "Sect", "Figs", "Fig", "Tab", "Eq", "Ref", "Refs", "App",  # section/figure refs
            "Polygon", "Polygons", "Station", "Delta", "Source",        # geometry / generic nouns
            "Carbon", "Nitrogen", "Oxygen", "Hydrogen",                 # element names (not places)
            "NSIDC", "NASA", "NOAA", "ESA", "AMSR", "NCEP", "ECMWF",     # org/product acronyms
            "EUMETSAT", "JAXA", "Copernicus",
        })
        # Bare compass directions are not place names on their own ("East"), but
        # keep compound forms ("East Siberian Sea", "Central Arctic").
        _BARE_DIR = frozenset({"East", "West", "North", "South",
                               "Eastern", "Western", "Northern", "Southern", "Central"})
        first_tok = val.split()[0]
        if (len(val) >= 4
                and val[0].isupper()                          # must start uppercase
                and not (val.isupper() and " " not in val and len(val) <= 6)  # short acronym: HCSL, NSF
                and val not in _BARE_DIR                       # bare compass direction
                # camelCase single token = extraction fragment, e.g. "SAiC" (⊂ MOSAiC)
                and not (" " not in val and re.search(r"[a-z][A-Z]", val))
                and first_tok not in _NER_STOP               # section/figure/org abbreviations
                and not re.search(r"\d", val)
                and not re.search(r"^[A-Z]\.", val)
                and not re.search(r"[A-Z]\.[,\s]", val)
                and not re.search(r"\bet\s+al", val, re.IGNORECASE)  # citation "et al."
                and not (len(val) > 20 and " " not in val)   # all-concatenated
                and not (len(first_tok) > 15)                 # concatenated first word
                and "(" not in val                            # citation artifact
                and "\n" not in val
                and val not in seen
                and not (                                     # author name: "et [al.] + year"
                    re.search(r"\bet\s*(?:al\.?)?", doc.text[ent.end_char:ent.end_char + 40])
                    and re.search(r"(?:19|20)\d{2}", doc.text[ent.end_char:ent.end_char + 40])
                )
                and not re.search(                            # single-author (YYYY) pattern
                    r"\(\s*(?:19|20)\d{2}",
                    doc.text[ent.end_char:ent.end_char + 25]
                )
                and not re.search(                            # reference-list entry:
                    r"^,\s*[A-Z]\.(?:\s*[A-Z]\.)?\s*[,:]",     # "Krumpen, T.," / "Hutchings, J. K.:"
                    doc.text[ent.end_char:ent.end_char + 15]
                )
                and not _is_author_citation(val, doc.text)):  # any citation context in doc
            seen.add(val)
            out.append(val)
    return out[:10]


# ── Core extraction ───────────────────────────────────────────────────────────

def tabulate_paper(pdf_path: str | Path, compare_llm: bool = False) -> dict:
    """
    Return a Panoply-style dict for one PDF:
    {paper, location, time_start, time_end, variables: [{name, unit}]}
    """
    pdf_path = Path(pdf_path)
    print(f"\n[tabulator] {'─' * 60}")
    print(f"[tabulator] {pdf_path.name}")

    result: dict = {
        "paper": pdf_path.name,
        "location": None,
        "time_start": None,
        "time_end": None,
        "variables": [],
    }

    full_text = _read_pdf(pdf_path)
    if len(full_text) < 300:
        result["error"] = "Too little text (likely scanned PDF)"
        print(f"[tabulator]   ERROR: too little text ({len(full_text)} chars) — likely scanned PDF")
        return result

    print(f"[tabulator]   full text : {len(full_text):,} chars")

    section_text, _section_fallback = _target_sections(full_text)
    print(f"[tabulator]   sections  : {len(section_text):,} chars  "
          f"({'targeted' if not _section_fallback else 'fallback — no section headers found'})")

    # ── Place names: title/abstract/intro ONLY, keyword regex first, NER as ──
    # supplement. Restricting to the intro is deliberate: Methods/Results/
    # Discussion/References cite many other regions (prior studies, comparison
    # sites) and were the source of location leaks on single-region papers.
    intro_text = _title_abstract_intro(full_text)
    print(f"[tabulator]   intro     : {len(intro_text):,} chars  (title/abstract/introduction only)")
    kw_places = _places_keywords(intro_text)
    ner_places = _places_ner(intro_text)
    # Supplement with NER only when keyword detection is sparse
    if len(kw_places) < 3:
        seen_lc: set[str] = {p.lower() for p in kw_places}
        for p in ner_places:
            p_lower = p.lower()
            # Skip NER entries that contain (or are contained by) a keyword result
            if any(kw.lower() in p_lower or p_lower in kw.lower()
                   for kw in kw_places):
                continue
            if p_lower not in seen_lc:
                kw_places.append(p)
                seen_lc.add(p_lower)
    places = kw_places[:5]
    if places:
        source = "keyword" if kw_places else "spaCy NER"
        result["location"] = ", ".join(places)
        print(f"[tabulator]   places    : {result['location']}  [{source}+NER, {len(kw_places)} found]")
    else:
        print(f"[tabulator]   places    : none detected")

    # ── Time range ────────────────────────────────────────────────────────────
    # Frequency-based selection over the paper body — see _extract_time_range.
    # Trim the reference list so its page/year ranges don't compete, but if
    # trimming would gut the paper (a spuriously-early "References" line in a long
    # PDF cut it to a fraction of its size), fall back to the full text so the
    # study period isn't lost.
    time_body = _trim_refs(full_text)
    if len(time_body) < 0.4 * len(full_text):
        time_body = full_text
    start, end, tnote = _extract_time_range(time_body)
    if start is not None:
        result["time_start"] = start
        result["time_end"] = end
        print(f"[tabulator]   time      : {start}–{end}  ({tnote})")
    else:
        # Fall back to a compact cluster of single years before giving up.
        # Skip years sitting right after a "<journal>, <vol>, <pages>," running
        # citation footer — those are the paper's own publication year, not a
        # data-collection year, and would otherwise leak in on every page.
        singles_set: set[int] = set()
        for _m in _SINGLE_YEAR_RE.finditer(section_text):
            _before = section_text[max(0, _m.start() - 40):_m.start()]
            if (_CITATION_YEAR_TAIL_RE.search(_before)
                    or _CITATION_PAGE_RE.search(_before)
                    or _is_pagenum_glued_year(section_text, _m.start())):
                continue
            singles_set.add(int(_m.group(1)))
        singles = sorted(singles_set)
        if singles and singles[-1] - singles[0] <= 20:
            result["time_start"] = singles[0]
            result["time_end"] = singles[-1]
            print(f"[tabulator]   time      : {singles[0]}–{singles[-1]}  "
                  f"(single years only, {len(singles)} found)")
        else:
            print(f"[tabulator]   time      : not detected  ({tnote})")

    # ── Variables ─────────────────────────────────────────────────────────────
    variables: list[dict] = []
    seen_keys: set[tuple] = set()

    # Method 1: explicit PDF tables
    table_count = 0
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages[:20]:
                for table in (page.extract_tables() or []):
                    for row in table:
                        if row and len(row) >= 2:
                            name_cell = (row[0] or "").strip()
                            unit_cell = (row[1] or "").strip()
                            if name_cell and unit_cell and _valid_unit(unit_cell):
                                key = (name_cell.lower()[:25], unit_cell)
                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    variables.append({"name": name_cell, "unit": unit_cell, "source": "table"})
                                    table_count += 1
    except Exception:
        pass
    print(f"[tabulator]   pass 1    : PDF tables  → {table_count} variable(s)")

    # Method 2 helper — shared logic for both passes
    def _sweep(text: str, label: str) -> tuple[int, int]:
        candidates = accepted = 0
        cid_skipped = stop_skipped = 0
        for m in _VAR_RE.finditer(text):
            name_raw = re.sub(r'\s+', ' ', m.group(1)).strip()   # normalise whitespace
            inner    = m.group(2).strip()
            candidates += 1
            # Skip PDF font-encoding artifacts like (cid:19)
            if re.match(r"cid:\d+", inner):
                cid_skipped += 1
                continue
            # Skip single-letter figure refs like (a), (j)
            if len(inner) == 1 and inner.isalpha():
                stop_skipped += 1
                continue
            # Skip if whole name, first word, or last word is a stop word
            words = name_raw.split()
            first_word = words[0].lower()
            last_word  = words[-1].lower()
            name_lower = name_raw.lower()
            # Single generic measurement-adjacent words (fine as last word in a phrase but
            # not as standalone variable names)
            _SOLO_STOPS = frozenset({"coverage", "extent", "concentration", "area",
                                     "thickness", "depth", "flux", "discharge"})
            if (name_lower in _STOP_NAMES
                    or name_lower in _SOLO_STOPS
                    or name_lower in _GEO_NOUNS        # exact-match only for geo terms
                    or first_word in _STOP_NAMES       # _STOP_NAMES no longer has "ice" etc.
                    or last_word in _STOP_NAMES
                    or (len(last_word) == 1 and last_word.isalpha())):
                stop_skipped += 1
                continue
            # Skip repeated word — column-join duplicated: "airtemperature airtemperature"
            if len(words) > 1 and any(
                words[i].lower() == words[i + 1].lower() for i in range(len(words) - 1)
            ):
                stop_skipped += 1
                continue
            # Skip concatenated names — PDF column-join or camelCase artifacts
            if " " not in name_raw and len(name_raw) > 11:
                stop_skipped += 1
                continue
            # SICsub-range style: uppercase run immediately followed by lowercase body
            if re.search(r'[A-Z]{2,}[a-z]', name_raw) and " " not in name_raw:
                stop_skipped += 1
                continue
            # conjunction embedded without spaces: "gainedorlost", "andsalinity"
            if " " not in name_raw and re.search(
                r'(^|[a-z])(or|and|to|of|in)[a-z]', name_raw.lower()
            ):
                stop_skipped += 1
                continue
            # Skip sentence fragments (too long to be a variable name)
            if len(name_raw) > 55:
                stop_skipped += 1
                continue
            # Skip names that contain sentence-fragment phrases
            name_lower = name_raw.lower()
            if re.search(
                r'\b(in its|in the|are a|is a|was a|with a|bars are|that is'
                r'|as it flows|slightly as|contain a|characterised by|since\b'
                r'|flows into|low\b.{0,10}[%]|over the|although\b'
                r'|integrated over|substantially|increases\b|decreases\b)\b',
                name_lower
            ):
                stop_skipped += 1
                continue
            if _valid_unit(inner):
                key = (name_raw.lower()[:25], inner)
                if key not in seen_keys:
                    seen_keys.add(key)
                    variables.append({"name": name_raw, "unit": inner, "source": label})
                    accepted += 1
            else:
                parts = re.split(r"[,;\s]+", inner)
                if parts and _valid_unit(parts[-1]):
                    key = (name_raw.lower()[:25], parts[-1])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        variables.append({"name": name_raw, "unit": parts[-1], "source": label})
                        accepted += 1
        rejected = candidates - accepted - cid_skipped - stop_skipped
        parts = []
        if cid_skipped:  parts.append(f"{cid_skipped} cid-artifacts")
        if stop_skipped: parts.append(f"{stop_skipped} filtered")
        note = f"  ({', '.join(parts)} skipped)" if parts else ""
        print(f"[tabulator]   {label:<10}: bracket regex  → {candidates} candidates, "
              f"{accepted} passed unit whitelist, {rejected} rejected{note}")
        if cid_skipped > 0 and candidates > 0 and cid_skipped / candidates > 0.25:
            print(f"[tabulator]   WARNING   : {cid_skipped}/{candidates} bracket matches are PDF "
                  f"font-encoding artifacts (cid:N) — unit symbols like °C, µmol/L may not be "
                  f"decodeable from this PDF's embedded fonts")
        return candidates, accepted

    # Pass 2a — targeted sections only
    _sweep(section_text, "pass 2a")

    # Pass 2b — full text sweep to catch appendix tables, figure captions, etc.
    before = len(variables)
    _sweep(full_text, "pass 2b")
    extra = len(variables) - before
    if extra:
        print(f"[tabulator]             {extra} additional variable(s) found in full text")

    heuristic_total = len(variables)
    print(f"[tabulator]   total     : {heuristic_total} variable(s) from heuristics")
    for v in variables:
        print(f"[tabulator]     {v['name']}  [{v['unit']}]  via {v['source']}")

    # Pass 3 — LLM fallback (always runs in compare mode; otherwise only when heuristics silent)
    if not os.getenv("AZURE_OPENAI_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).parent.parent / "Knowledge_graph" / ".env")
        except Exception:
            pass
    azure_ready = bool(os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
    run_llm = azure_ready and (compare_llm or heuristic_total < 5)
    if run_llm:
        mode = "compare" if compare_llm and heuristic_total > 0 else "fallback"
        print(f"[tabulator]   pass 3    : LLM variable extraction ({mode})")
        t0 = time.time()
        # When section detection fell back to abstract/intro (no section headers found),
        # send the post-intro window instead — methods sections for most academic papers
        # start around char 8000-18000, well past the abstract/intro region.
        if _section_fallback and len(full_text) > 25000:
            llm_input = full_text[8000:18000]
            print(f"[tabulator]   pass 3    : section fallback — using post-intro window [8000:18000]")
        else:
            llm_input = section_text or full_text[:6000]
        llm_vars = _vars_llm(llm_input)

        # If the first window yielded nothing, retry on a different region.
        # Papers vary widely in structure — cosmogenic-dating / paleoclimate papers
        # often describe their variables in the abstract/intro rather than in a
        # labelled Methods block, so a second pass on the beginning of the paper
        # catches what the post-intro window missed.
        if not llm_vars and _section_fallback and len(full_text) > 25000:
            print(f"[tabulator]   pass 3b   : 0 candidates — retrying with intro window [0:6000]")
            llm_vars_b = _vars_llm(full_text[:6000])
            if llm_vars_b:
                llm_vars = llm_vars_b
                print(f"[tabulator]   pass 3b   : {len(llm_vars)} candidate(s) from intro window")
            else:
                # One more attempt: the later results section
                print(f"[tabulator]   pass 3c   : 0 candidates — retrying with results window [18000:24000]")
                llm_vars_c = _vars_llm(full_text[18000:24000])
                if llm_vars_c:
                    llm_vars = llm_vars_c
                    print(f"[tabulator]   pass 3c   : {len(llm_vars)} candidate(s) from results window")

        elapsed = time.time() - t0

        new_from_llm = 0
        llm_accepted = []
        # Prefix-based dedup: "basal melting" vs "basal melting rate" → keep first
        llm_seen_names: list[str] = [v["name"].lower() for v in variables]
        for item in llm_vars:
            name, unit = item["name"], item["unit"]
            name_lower = name.lower()
            key = (name_lower[:25], unit)
            if key in seen_keys:
                continue
            # Skip if this name is a prefix-extension of an already-accepted name, or vice versa
            if any(
                name_lower.startswith(p + " ") or p.startswith(name_lower + " ")
                for p in llm_seen_names
            ):
                continue
            seen_keys.add(key)
            llm_seen_names.append(name_lower)
            entry = {"name": name, "unit": unit, "source": "llm"}
            variables.append(entry)
            llm_accepted.append(entry)
            new_from_llm += 1

        if compare_llm and heuristic_total > 0:
            # Side-by-side comparison output
            print(f"[tabulator]   pass 3    : LLM found {len(llm_vars)} candidate(s)  ({elapsed:.1f}s)")
            if llm_vars:
                print(f"[tabulator]   LLM vars  :")
                for item in llm_vars:
                    overlap = any(v["name"].lower()[:20] == item["name"].lower()[:20]
                                  for v in variables if v["source"] != "llm")
                    marker = "=" if overlap else "+"
                    print(f"[tabulator]     [{marker}] {item['name']}  [{item['unit']}]")
                print(f"[tabulator]   [=] = matches heuristic  [+] = new from LLM  "
                      f"({new_from_llm} new added)")
        else:
            print(f"[tabulator]   pass 3    : LLM → {len(llm_vars)} candidate(s), "
                  f"{new_from_llm} accepted  ({elapsed:.1f}s)")
            for v in llm_accepted:
                print(f"[tabulator]     {v['name']}  [{v['unit']}]  via llm")

    total = len(variables)
    print(f"[tabulator]   FINAL     : {total} variable(s)")
    result["variables"] = variables
    return result


def tabulate_papers(pdf_paths: list, compare_llm: bool = False) -> list[dict]:
    print(f"\n[tabulator] ={'=' * 60}")
    print(f"[tabulator] Starting variable tabulation for {len(pdf_paths)} paper(s)")
    print(f"[tabulator] ={'=' * 60}")
    results = [tabulate_paper(p, compare_llm=compare_llm) for p in pdf_paths]
    total_vars = sum(len(r.get("variables", [])) for r in results)
    print(f"\n[tabulator] ={'=' * 60}")
    print(f"[tabulator] Done — {len(pdf_paths)} paper(s), {total_vars} variable(s) total")
    print(f"[tabulator] ={'=' * 60}\n")
    return results
