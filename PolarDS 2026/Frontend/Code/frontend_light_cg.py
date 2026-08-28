import sys
import os

# This app lives in Frontend/Code/, but its Knowledge-Graph and Causal-Graph
# modules live in sibling top-level folders (Knowledge_graph/Code/,
# Causal_graph/Code/), so both must be added to sys.path before the
# project-internal imports below can resolve.
_KG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Knowledge_graph", "Code")
_CG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Causal_graph", "Code")
for _dir in (_KG_DIR, _CG_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

import streamlit as st
from keywords_extraction import process
from scg_keyword_extraction import process_enhanced, extract_dataset_variables
from neo4j_storage import Neo4jConnector
from qa_module import qa_system
from frontend_dataset_display import (
    display_gpt4_toggle,
    display_dataset_filter,
    display_datasets_section,
    display_cost_summary,
    export_datasets_to_csv
)
from causal_graph import (
    extract_causal_relations, generate_causal_graph, run_causal_discovery, is_valid_node,
    TAU_MAX_METHODS, PC_ALPHA_METHODS,
)
from causal_discovery import (
    load_dataset, detect_time_column, numeric_columns, suggest_variable_mapping, clean_node_name,
    prettify_node_name, prettify_column_name, plot_full_causal_graph,
)
import json
import pandas as pd

# Page config
st.set_page_config(
    page_title="PolarKD — Polar Knowledge Discovery Toolkit",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Display-only names for causal_discovery.py's method keys — used in the
# Enhanced Structural Causal Graph's (Step 04) method multiselect and
# results-expander headers. Not a generic casing rule since these are
# algorithm names/initialisms, not
# ordinary words (e.g. LiNGAM's internal capitalization is how the
# algorithm's own authors stylize it). Falls back to method.upper() for any
# key not listed here, so a new method added to causal_discovery.py without
# an entry here still gets a reasonable display name.
METHOD_DISPLAY_NAMES = {
    'pc': 'PC',
    'pcmci': 'PCMCI',
    'pcmci_plus': 'PCMCI+',
    'tcdf': 'TCDF',
    'lingam': 'LiNGAM',
    'fci': 'FCI',
    'cdnod': 'CD-NOD',
    'daggnn': 'DAG-GNN',
    'lpcmci': 'LPCMCI',
    'ges': 'GES',
}


def _method_display(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method.upper())

# ─── CACHED HELPERS ─────────────────────────────────────────────────────────
# Streamlit reruns this whole script on every widget interaction anywhere on
# the page. Without caching, the Enhanced Structural Causal Graph's PyVis
# rendering and fuzzy variable-name matching (Step 04) would rebuild on
# every unrelated click. These are cached by input content, so they only
# recompute when the underlying relations/nodes/columns actually change.

@st.cache_data(show_spinner=False)
def _cached_causal_graph_html(causal_rels: list, output_path: str) -> str:
    _, html = generate_causal_graph(causal_rels, output_path=output_path)
    return html


@st.cache_data(show_spinner=False)
def _cached_variable_mapping(kg_nodes: list, dataset_columns: list) -> dict:
    return suggest_variable_mapping(kg_nodes, dataset_columns)


@st.cache_data(show_spinner=False)
def _cached_kg_graph_html(all_nodes: list, all_relations: list, all_datasets: list, graph_type: str) -> str:
    """Stores/retrieves from Neo4j and builds the KG PyVis graph. Cached so
    unrelated reruns (any widget interaction anywhere in the app) don't
    re-create every node/relation in Neo4j again."""
    neo = Neo4jConnector()
    try:
        neo.store_keywords_and_relations(all_nodes, all_relations, all_datasets)
        rels = neo.retrieve_relations()
        graph, expansion_js = neo.generate_graph(rels, graph_type=graph_type)
        graph.save_graph("graph.html")
        with open("graph.html", "r") as f:
            html_content = f.read()
        html_content = html_content.replace("</body>", expansion_js + "</body>")
        with open("graph.html", "w") as f:
            f.write(html_content)
        with open("graph.html", "r") as f:
            return f.read()
    finally:
        neo.close()

# ─── EDITORIAL CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Reset & Base ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    html, body, .stApp {
        background-color: #FFFFFF !important;
        color: #0D3D74 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .main .block-container {
        background-color: transparent !important;
        padding: 0 2rem 4rem 2rem !important;
        max-width: 1280px !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Typography overrides ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #0D3D74 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    p, span, div, label, li {
        color: #0D3D74 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Ensure button inner elements are never hijacked by the rule above */
    .stButton > button, .stButton > button *,
    .stDownloadButton > button, .stDownloadButton > button *,
    .stFormSubmitButton > button, .stFormSubmitButton > button *,
    [data-testid="baseButton-primary"], [data-testid="baseButton-primary"] *,
    [data-testid="baseButton-secondary"], [data-testid="baseButton-secondary"] * {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Hero Banner ── */
    .polar-hero {
        background: linear-gradient(135deg, #0D3D74 0%, #0D3D74 45%, #377FD0 100%);
        padding: 4rem 4rem 3rem 4rem;
        margin: -1rem -2rem 0 -2rem;
        position: relative;
        overflow: hidden;
    }

    .polar-hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 320px; height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(55,127,208,0.15) 0%, transparent 70%);
        pointer-events: none;
    }

    .polar-hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 10%;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(55,127,208,0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #377FD0 !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .hero-eyebrow::before, .hero-eyebrow::after {
        content: '';
        display: inline-block;
        width: 36px; height: 1px;
        background: #377FD0;
        opacity: 0.6;
    }

    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: clamp(2.6rem, 5vw, 4rem) !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        line-height: 1.08 !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.5rem;
    }

    .hero-title em {
        font-style: italic;
        color: #377FD0 !important;
    }

    .hero-subtitle {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.25rem !important;
        color: rgba(245,248,252,0.65) !important;
        font-weight: 300 !important;
        margin-top: 1.25rem !important;
        max-width: 580px;
        line-height: 1.6;
    }

    .hero-meta {
        margin-top: 2.5rem;
        display: flex;
        gap: 2.5rem;
        align-items: center;
        flex-wrap: wrap;
    }

    .hero-stat {
        text-align: left;
    }

    .hero-stat-num {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #377FD0 !important;
        font-weight: 600;
        line-height: 1;
    }

    .hero-stat-label {
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: rgba(245,248,252,0.45) !important;
        margin-top: 0.2rem;
    }

    .hero-divider {
        width: 1px; height: 40px;
        background: rgba(55,127,208,0.25);
    }

    /* ── Navigation Pills ── */
    .polar-nav {
        display: flex;
        gap: 0.5rem;
        padding: 1.25rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(13,61,116,0.1);
        margin-bottom: 2rem;
    }

    .polar-nav-pill {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.55rem 1.25rem;
        border-radius: 2px;
        color: #0D3D74 !important;
        background: transparent;
        border: 1px solid rgba(13,61,116,0.15);
        cursor: pointer;
        transition: all 0.25s ease;
    }

    .polar-nav-pill:hover, .polar-nav-pill.active {
        background: #0D3D74;
        color: #FFFFFF !important;
        border-color: #0D3D74;
    }

    /* ── Section Labels ── */
    .section-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #377FD0 !important;
        margin-bottom: 0.6rem;
        display: block;
    }

    .section-heading {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.9rem !important;
        font-weight: 600 !important;
        color: #0D3D74 !important;
        letter-spacing: -0.025em !important;
        margin-bottom: 1.75rem !important;
        line-height: 1.15 !important;
    }

    .section-heading em {
        font-style: italic;
        color: #377FD0 !important;
    }

    /* ── Upload Zone ── */
    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;
        border: 1.5px dashed rgba(46,95,160,0.3) !important;
        border-radius: 4px !important;
        padding: 3rem 2rem !important;
        text-align: center !important;
        min-height: 180px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #377FD0 !important;
        background: #EAF2FD !important;
        box-shadow: 0 0 0 4px rgba(55,127,208,0.08) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #377FD0 !important;
        font-size: 0.9rem !important;
        font-weight: 400 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "↑";
        display: block;
        font-size: 2rem;
        margin-bottom: 0.75rem;
        color: #377FD0 !important;
        font-weight: 300;
    }

    /* ── Buttons ── */
    .stButton > button,
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"] {
        background: #0D3D74 !important;
        color: #FFFFFF !important;
        border: 1px solid #0D3D74 !important;
        padding: 0.7rem 1.75rem !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        transition: all 0.25s ease !important;
        box-shadow: none !important;
        white-space: nowrap !important;
    }

    /* Force ALL child text inside buttons to stay cream — overrides global p/span/div rules */
    .stButton > button *,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div,
    [data-testid="baseButton-primary"] *,
    [data-testid="baseButton-secondary"] * {
        color: #FFFFFF !important;
    }

    .stButton > button:hover,
    .stButton > button:hover *,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="baseButton-primary"]:hover *,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover * {
        background: #377FD0 !important;
        border-color: #377FD0 !important;
        color: #0D3D74 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(55,127,208,0.3) !important;
    }

    .stDownloadButton > button {
        background: transparent !important;
        color: #0D3D74 !important;
        border: 1px solid rgba(13,61,116,0.4) !important;
        padding: 0.7rem 1.75rem !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        transition: all 0.25s ease !important;
    }

    .stDownloadButton > button *,
    .stDownloadButton > button p,
    .stDownloadButton > button span {
        color: #0D3D74 !important;
    }

    .stDownloadButton > button:hover,
    .stDownloadButton > button:hover * {
        background: #0D3D74 !important;
        color: #FFFFFF !important;
        border-color: #0D3D74 !important;
    }

    .stFormSubmitButton > button {
        background: #377FD0 !important;
        color: #0D3D74 !important;
        border: 1px solid #377FD0 !important;
        padding: 0.7rem 2rem !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        transition: all 0.25s ease !important;
        white-space: nowrap !important;
        min-width: 90px !important;
    }

    .stFormSubmitButton > button *,
    .stFormSubmitButton > button p,
    .stFormSubmitButton > button span {
        color: #0D3D74 !important;
    }

    .stFormSubmitButton > button:hover,
    .stFormSubmitButton > button:hover * {
        background: #0D3D74 !important;
        color: #FFFFFF !important;
        border-color: #0D3D74 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Cards ── */
    .polar-card {
        background: #FFFFFF;
        border: 1px solid rgba(13,61,116,0.08);
        border-radius: 4px;
        padding: 1.75rem;
        margin-bottom: 1rem;
    }

    .polar-card-dark {
        background: #0D3D74;
        border: none;
        border-radius: 4px;
        padding: 1.75rem;
        margin-bottom: 1rem;
    }

    .polar-card-dark p, .polar-card-dark span, .polar-card-dark div, .polar-card-dark label {
        color: rgba(245,248,252,0.75) !important;
    }

    .polar-card-dark h3 {
        color: #FFFFFF !important;
    }

    .polar-info-row {
        background: #FFFFFF;
        border-left: 3px solid #377FD0;
        padding: 1rem 1.25rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        color: #0D3D74 !important;
    }

    /* ── Alert / Info Boxes ── */
    .stAlert, [data-testid="stAlert"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(55,127,208,0.4) !important;
        border-left: 3px solid #377FD0 !important;
        border-radius: 4px !important;
        color: #0D3D74 !important;
    }

    .stSuccess {
        background: #F2FBF4 !important;
        border-left-color: #4A9B6F !important;
    }

    .stWarning {
        background: #FFFBF0 !important;
        border-left-color: #D4A017 !important;
    }

    .stError {
        background: #FFF5F5 !important;
        border-left-color: #C0392B !important;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(13,61,116,0.1) !important;
        margin: 2.5rem 0 !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(13,61,116,0.07) !important;
        border-radius: 4px !important;
        padding: 1.25rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: #84888D !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
        color: #0D3D74 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #4A9B6F !important;
    }

    /* ── Keyword Tags ── */
    .kw-tag {
        display: inline-block;
        background: #0D3D74;
        color: #FFFFFF !important;
        padding: 0.3rem 0.85rem;
        border-radius: 2px;
        margin: 0.2rem;
        font-size: 0.72rem;
        font-weight: 400;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }

    .kw-tag-light {
        display: inline-block;
        background: transparent;
        color: #0D3D74 !important;
        border: 1px solid rgba(13,61,116,0.25);
        padding: 0.3rem 0.85rem;
        border-radius: 2px;
        margin: 0.2rem;
        font-size: 0.72rem;
        font-weight: 400;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Causal Tags ── */
    .causal-cause-tag {
        display: inline-block;
        background: #C0392B;
        color: #FFFFFF !important;
        padding: 0.3rem 0.85rem;
        border-radius: 2px;
        margin: 0.2rem;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }

    .causal-effect-tag {
        display: inline-block;
        background: #E67E22;
        color: #FFFFFF !important;
        padding: 0.3rem 0.85rem;
        border-radius: 2px;
        margin: 0.2rem;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Chat ── */
    .chat-bubble-user {
        background: #0D3D74;
        color: #FFFFFF !important;
        padding: 1rem 1.25rem;
        border-radius: 4px 4px 4px 0;
        margin: 0.75rem 0;
        font-size: 0.875rem;
        line-height: 1.6;
    }

    .chat-bubble-user strong, .chat-bubble-user span, .chat-bubble-user div {
        color: #FFFFFF !important;
    }

    .chat-bubble-assistant {
        background: #FFFFFF;
        border: 1px solid rgba(13,61,116,0.09);
        padding: 1rem 1.25rem;
        border-radius: 4px 4px 0 4px;
        margin: 0.75rem 0;
        font-size: 0.875rem;
        line-height: 1.6;
        border-left: 3px solid #377FD0;
    }

    .chat-label {
        font-size: 0.62rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-weight: 500;
        margin-bottom: 0.3rem;
        display: block;
    }

    .chat-label-user { color: #377FD0 !important; }
    .chat-label-ai { color: #84888D !important; }

    /* ── Text Input ── */
    .stTextInput > div > div > input {
        background: #FFFFFF !important;
        color: #0D3D74 !important;
        border: 1px solid rgba(13,61,116,0.2) !important;
        border-radius: 2px !important;
        padding: 0.65rem 1rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
        transition: border-color 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #377FD0 !important;
        box-shadow: 0 0 0 3px rgba(55,127,208,0.12) !important;
        outline: none !important;
    }

    /* ── Selectbox — trigger box ── */
    .stSelectbox > div > div,
    [data-testid="stSelectbox"] > div > div {
        background: #FFFFFF !important;
        border: 1px solid rgba(13,61,116,0.2) !important;
        border-radius: 2px !important;
    }

    /* Text shown inside the selectbox trigger */
    .stSelectbox [data-baseweb="select"] > div,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] input {
        background: #FFFFFF !important;
        color: #0D3D74 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
    }

    /* Dropdown list container (the dark popover) */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"],
    [role="listbox"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(13,61,116,0.12) !important;
        border-radius: 4px !important;
        box-shadow: 0 8px 24px rgba(13,61,116,0.12) !important;
    }

    /* Each dropdown option */
    [role="option"],
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] ul li {
        background: #FFFFFF !important;
        color: #0D3D74 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
    }

    /* Hovered / focused option */
    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    [data-baseweb="menu"] li:hover {
        background: #DCE9FB !important;
        color: #0D3D74 !important;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] {
        padding-top: 0.25rem;
    }

    .stSlider [role="slider"] {
        background: #0D3D74 !important;
    }

    .stSlider [data-testid="stThumbValue"] {
        color: #0D3D74 !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div {
        background: #377FD0 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #FFFFFF !important;
        border: 1px solid rgba(13,61,116,0.1) !important;
        border-radius: 4px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        color: #0D3D74 !important;
    }

    /* ── Database Document Item ── */
    .doc-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: #FFFFFF;
        border: 1px solid rgba(13,61,116,0.08);
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.83rem;
        color: #0D3D74 !important;
    }

    .doc-item::before {
        content: '↗';
        color: #377FD0;
        font-size: 1rem;
        flex-shrink: 0;
    }

    /* ── Graph Legend ── */
    .graph-legend {
        display: flex;
        gap: 2rem;
        padding: 1rem 1.5rem;
        background: #FFFFFF;
        border: 1px solid rgba(13,61,116,0.08);
        border-radius: 4px;
        margin-bottom: 1.25rem;
        align-items: center;
    }

    .graph-legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #0D3D74 !important;
        font-weight: 500;
    }

    .graph-legend-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Empty State ── */
    .empty-state {
        background: #FFFFFF;
        border: 1px dashed rgba(13,61,116,0.15);
        border-radius: 4px;
        padding: 4rem 2rem;
        text-align: center;
    }

    .empty-state-glyph {
        font-size: 2rem;
        color: #377FD0 !important;
        margin-bottom: 1rem;
        display: block;
        font-weight: 300;
    }

    .empty-state-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #0D3D74 !important;
        margin-bottom: 0.5rem;
    }

    .empty-state-text {
        font-size: 0.82rem;
        color: #84888D !important;
        max-width: 320px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Footer ── */
    .polar-footer {
        margin-top: 4rem;
        padding: 2.5rem 0;
        border-top: 1px solid rgba(13,61,116,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .polar-footer-brand {
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        color: #0D3D74 !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .polar-footer-links {
        display: flex;
        gap: 1.5rem;
    }

    .polar-footer-links a {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #84888D !important;
        text-decoration: none;
        transition: color 0.2s;
    }

    .polar-footer-links a:hover { color: #0D3D74 !important; }

    .polar-footer-copy {
        font-size: 0.72rem;
        color: #84888D !important;
        letter-spacing: 0.05em;
    }

    /* ── Column padding ── */
    [data-testid="column"] { padding: 0 0.75rem !important; }

    /* ── Checkbox ── */
    .stCheckbox span {
        font-size: 0.83rem !important;
        color: #0D3D74 !important;
    }

    /* ── Tooltip ── */
    .stTooltipIcon { color: #377FD0 !important; }

    /* ── Subtle horizontal rule inside sections ── */
    .inner-rule {
        border: none;
        border-top: 1px solid rgba(13,61,116,0.08);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOGO LOADING (GitHub raw URLs) ────────────────────────────────────────
IHARP_LOGO_URL = "https://raw.githubusercontent.com/d3lab-unt/polarKD/main/Knowledge_graph/images/iharp%20logo.png"
UNT_LOGO_URL   = "https://raw.githubusercontent.com/d3lab-unt/polarKD/main/Knowledge_graph/images/university-of-north-texas-seeklogo.png"


# ─── SESSION STATE ─────────────────────────────────────────────────────────
for key, default in [
    # Steps 01-03 (Upload / Q&A / Knowledge Graph) pipeline state.
    # uploaded_files/databases/chat_history back the Q&A flow; processed_pdfs/
    # current_graph back the Knowledge Graph flow; the two show_*_dialog flags
    # track which of Step 01's two inline confirm dialogs (if any) is open —
    # see the dialog/rerun pattern comment at show_qa_dialog's first use below.
    ('uploaded_files', []),
    ('databases', []),
    ('chat_history', []),
    ('processed_pdfs', {}),
    ('current_graph', None),
    ('show_qa_dialog', False),
    ('show_kg_dialog', False),
    # Enhanced Structural Causal Graph (Step 04) pipeline state — entirely
    # separate from the keys above by design, so this pipeline never reads or
    # writes Step 01/03's state and vice versa.
    ('show_scg_kg_dialog', False),
    ('scg_kg_model_selected', None),
    ('scg_kg_graph_type_selected', None),
    ('scg_processed_pdfs', {}),
    ('show_scg_cg_dialog', False),
    ('scg_cg_model_selected', None),
    ('scg_cg_relations', []),
    ('scg_validation_dataset', None),
    ('scg_validation_dataset_name', None),
    ('scg_validation_dataset_file_id', None),
    ('show_scg_dataset_dialog', False),
    ('scg_methods_selected', None),
    ('scg_pc_alpha_selected', 0.05),
    ('scg_tau_max_selected', 21),
    ('show_scg_discovery_dialog', False),
    ('scg_discovery_dialog_handled_for', None),
    ('scg_run_request', None),
    ('scg_results', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── HERO ──────────────────────────────────────────────────────────────────
iharp_logo_html = f'<img src="{IHARP_LOGO_URL}" style="height:48px;width:auto;margin-bottom:2rem;opacity:0.9;" alt="iHARP Logo">'

unt_logo_html = (
    '<div style="position:absolute;top:1.5rem;right:2rem;'
    'background:white;border-radius:50%;padding:6px;'
    'box-shadow:0 2px 12px rgba(0,0,0,0.2);'
    'display:flex;align-items:center;justify-content:center;">'
    f'<img src="{UNT_LOGO_URL}" '
    'style="height:80px;width:80px;object-fit:contain;border-radius:50%;" alt="UNT Logo">'
    '</div>'
)

hero_html = (
    '<div class="polar-hero">'
    + unt_logo_html
    + iharp_logo_html
    + '<div class="hero-eyebrow">iHARP Research Initiative</div>'
    + '<div class="hero-title">Polar <em>Knowledge</em><br>Discovery Toolkit</div>'
    + '<div class="hero-subtitle">Extract climate variables, build semantic knowledge graphs, discover causal relationships, and interrogate polar science literature — all within a single intelligent workspace.</div>'
    + '<div class="hero-meta">'
    + '<a href="#section-upload" style="text-decoration:none;">'
    +   '<div class="hero-stat" style="cursor:pointer;" onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
    +     '<div class="hero-stat-num">PDF</div>'
    +     '<div class="hero-stat-label">Ingestion</div>'
    +   '</div>'
    + '</a>'
    + '<div class="hero-divider"></div>'
    + '<a href="#section-qa" style="text-decoration:none;">'
    +   '<div class="hero-stat" style="cursor:pointer;" onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
    +     '<div class="hero-stat-num">Q&amp;A</div>'
    +     '<div class="hero-stat-label">Document QA</div>'
    +   '</div>'
    + '</a>'
    + '<div class="hero-divider"></div>'
    + '<a href="#section-kg" style="text-decoration:none;">'
    +   '<div class="hero-stat" style="cursor:pointer;" onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
    +     '<div class="hero-stat-num">KG</div>'
    +     '<div class="hero-stat-label">Knowledge Graph</div>'
    +   '</div>'
    + '</a>'
    + '<div class="hero-divider"></div>'
    + '<a href="#section-scg" style="text-decoration:none;">'
    +   '<div class="hero-stat" style="cursor:pointer;" onmouseover="this.style.opacity=\'0.7\'" onmouseout="this.style.opacity=\'1\'">'
    +     '<div class="hero-stat-num">CG</div>'
    +     '<div class="hero-stat-label">Causal Graph</div>'
    +   '</div>'
    + '</a>'
    + '</div>'
    + '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div id="section-upload"></div>', unsafe_allow_html=True)
st.markdown('<span class="section-label">Step 01</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Upload <em>Documents</em></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Drag & Drop PDFs here or Click to Upload",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="visible",
        help="Select multiple PDF files"
    )

    if uploaded_files is not None and len(uploaded_files) > 0:
        st.success(f"✓ {len(uploaded_files)} file(s) ready")
        for i, file in enumerate(uploaded_files, 1):
            st.markdown(f'<div class="doc-item">{i}. {file.name} &nbsp;·&nbsp; {file.size // 1024} KB</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin-top:0.5rem;padding:0.75rem 1rem;background:#FFFFFF;border-radius:4px;border:1px solid rgba(13,61,116,0.08);font-size:0.8rem;color:#84888D;">No files selected yet — drag PDFs above or click to browse.</div>', unsafe_allow_html=True)
with col1:
    st.markdown('<span class="section-label">Configuration</span>', unsafe_allow_html=True)
    k = st.slider("Keywords to Extract (Knowledge Graph)", min_value=5, max_value=50, value=15, step=5)
    use_gpt4_datasets = display_gpt4_toggle()
    filter_variables = st.checkbox(
        "Filter to Climate Variables Only",
        value=True,
        help="Retain only measurable variables (temperature, salinity, pressure…). Removes organisations, locations, methods."
    )

with col2:
    # Dialog/rerun pattern used throughout this file (Streamlit reruns the
    # whole script top-to-bottom on every widget interaction, so state that
    # must survive across reruns has to live in st.session_state, not a
    # plain local variable): a trigger button sets a `show_X_dialog` flag,
    # Streamlit's automatic rerun then renders the dialog because that flag
    # is now True, and its Confirm button either (a) does the work directly
    # inline and clears the flag — the simpler shape, used by the Q&A dialog
    # right below — or (b) stores an `X_selected` value and calls
    # st.rerun() explicitly, deferring the actual work to a separate block
    # later in the script that checks for that value — used by the
    # Knowledge Graph dialog just below the Q&A one, and throughout Step 04,
    # whenever the processing needs its own dedicated output area (e.g. a
    # progress bar across multiple files) rather than running inline inside
    # the dialog itself. Recognizing this shape once here makes every later
    # `show_*_dialog`/`*_selected` pair in this file self-explanatory.
    if 'show_qa_dialog' not in st.session_state:
        st.session_state.show_qa_dialog = False
    if 'show_kg_dialog' not in st.session_state:
        st.session_state.show_kg_dialog = False

    # ── Q&A Button
    if st.button("📚 Send to Q&A", use_container_width=True, key="send_qa"):
        if uploaded_files and len(uploaded_files) > 0:
            st.session_state.show_qa_dialog = True
            st.session_state.show_kg_dialog = False
        else:
            st.warning("Please upload files first.")

    # ── Q&A inline card
    if st.session_state.show_qa_dialog:
        st.markdown("""
        <div style="
            background:#F0F6FF;
            border:1.5px solid #377FD0;
            border-left:4px solid #377FD0;
            border-radius:6px;
            padding:1.25rem 1.5rem 0.75rem 1.5rem;
            margin-top:0.5rem;
            margin-bottom:0.25rem;
        ">
            <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D3D74;margin-bottom:0.2rem;">
                📚 Send to Q&amp;A
            </div>
            <div style="font-size:0.76rem;color:#84888D;letter-spacing:0.04em;">
                Select the LLM model to use for answering questions
            </div>
        </div>
        """, unsafe_allow_html=True)

        # This fixed model list (repeated at every model-selection dialog in
        # this file) is not auto-discovered from Ollama — it must be kept in
        # sync by hand with whatever models are actually pulled in the local
        # Ollama instance (`ollama pull <name>`). Picking a model here that
        # hasn't been pulled won't fail until generation is actually
        # attempted, not at selection time.
        qa_model = st.selectbox(
            "LLM Model",
            options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
            index=0,
            key="qa_model_dialog",
            help="Ollama model for answering questions."
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ Confirm", use_container_width=True, key="qa_confirm"):
                st.session_state.show_qa_dialog = False
                qa_system.set_model(qa_model)
                st.info(f"Model: **{qa_model}**")
                with st.spinner("Indexing documents…"):
                    added_count = 0
                    for file in uploaded_files:
                        if file.name not in st.session_state.databases:
                            file.seek(0)
                            # "temp_qa_" prefix (vs. "temp_"/"temp_scg_" used
                            # elsewhere in this file) keeps this file's temp
                            # copy from colliding with Step 01's or Step 04's
                            # own temp copy of the same uploaded file, in case
                            # a user triggers more than one processing path
                            # for the same PDF in close succession.
                            temp_path = f"temp_qa_{file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(file.read())
                            if qa_system.add_document(file.name, pdf_path=temp_path):
                                st.session_state.databases.append(file.name)
                                added_count += 1
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    if added_count > 0:
                        st.success(f"✓ {added_count} file(s) indexed.")
                    else:
                        st.info("Files already indexed.")
        with col_cancel:
            if st.button("✕ Cancel", use_container_width=True, key="qa_cancel"):
                st.session_state.show_qa_dialog = False
                st.rerun()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── KG Button
    if st.button("◎ Generate Knowledge Graph", use_container_width=True, key="gen_kg"):
        if uploaded_files and len(uploaded_files) > 0:
            st.session_state.show_kg_dialog = True
            st.session_state.show_qa_dialog = False
        else:
            st.warning("Please upload files first.")

    # ── KG inline card
    if st.session_state.show_kg_dialog:
        st.markdown("""
        <div style="
            background:#F0F6FF;
            border:1.5px solid #377FD0;
            border-left:4px solid #377FD0;
            border-radius:6px;
            padding:1.25rem 1.5rem 0.75rem 1.5rem;
            margin-top:0.5rem;
            margin-bottom:0.25rem;
        ">
            <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D3D74;margin-bottom:0.2rem;">
                ◎ Knowledge Graph Configuration
            </div>
            <div style="font-size:0.76rem;color:#84888D;letter-spacing:0.04em;">
                Choose model and graph type before generating
            </div>
        </div>
        """, unsafe_allow_html=True)

        kg_model = st.selectbox(
            "LLM Model for Relation Extraction",
            options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
            index=0,
            key="kg_model_dialog",
        )
        kg_graph_type = st.selectbox(
            "Graph Visualization Type",
            options=["Full Graph (with Datasets)", "Knowledge Graph Only (without Datasets)"],
            index=0,
            key="kg_graph_type_dialog",
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ Confirm", use_container_width=True, key="kg_confirm"):
                st.session_state.show_kg_dialog = False
                st.session_state.kg_model_selected = kg_model
                st.session_state.kg_graph_type_selected = kg_graph_type
                st.rerun()
        with col_cancel:
            if st.button("✕ Cancel", use_container_width=True, key="kg_cancel"):
                st.session_state.show_kg_dialog = False
                st.rerun()

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════
    #  SECTION 4 ACTION — LLM Enhanced Structural Causal Graph. Only the
    #  trigger + config dialogs + processing live here, next to the KG
    #  button above; results render in the Step 04 section further down
    #  the page, same trigger/result split Knowledge Graph generation uses.
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    if st.button("🧬 Generate Structural Causal Graph", use_container_width=True, key="gen_scg"):
        if not (uploaded_files and len(uploaded_files) > 0):
            st.warning("Please upload files first.")
        else:
            # Always restart the whole KG → CG → dataset → discovery chain from
            # scratch on every click (same as Step 01's KG button) — a repeat
            # click means "regenerate", not "resume where I left off".
            st.session_state.scg_processed_pdfs = {}
            st.session_state.scg_cg_relations = []
            st.session_state.scg_validation_dataset = None
            st.session_state.scg_validation_dataset_name = None
            st.session_state.scg_validation_dataset_file_id = None
            st.session_state.scg_discovery_dialog_handled_for = None
            st.session_state.scg_methods_selected = None
            st.session_state.scg_run_request = None
            st.session_state.scg_results = None
            st.session_state.show_scg_cg_dialog = False
            st.session_state.show_scg_dataset_dialog = False
            st.session_state.show_scg_discovery_dialog = False
            st.session_state.show_scg_kg_dialog = True

    # ── Step A: KG configuration (independent run, own model/graph-type choice)
    if st.session_state.show_scg_kg_dialog:
        st.markdown("""
        <div style="
            background:#F0F6FF;
            border:1.5px solid #377FD0;
            border-left:4px solid #377FD0;
            border-radius:6px;
            padding:1.25rem 1.5rem 0.75rem 1.5rem;
            margin-top:0.5rem;
            margin-bottom:0.25rem;
        ">
            <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D3D74;margin-bottom:0.2rem;">
                ◎ Enhanced SCG — Knowledge Graph Configuration
            </div>
            <div style="font-size:0.76rem;color:#84888D;letter-spacing:0.04em;">
                Independent KG extraction for Step 04 — choose model and graph type
            </div>
        </div>
        """, unsafe_allow_html=True)

        scg_kg_model = st.selectbox(
            "LLM Model for Relation Extraction",
            options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
            index=0,
            key="scg_kg_model_dialog",
        )
        scg_kg_graph_type = st.selectbox(
            "Graph Visualization Type",
            options=["Full Graph (with Datasets)", "Knowledge Graph Only (without Datasets)"],
            index=0,
            key="scg_kg_graph_type_dialog",
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ Confirm", use_container_width=True, key="scg_kg_confirm"):
                st.session_state.show_scg_kg_dialog = False
                st.session_state.scg_kg_model_selected = scg_kg_model
                st.session_state.scg_kg_graph_type_selected = scg_kg_graph_type
                st.rerun()
        with col_cancel:
            if st.button("✕ Cancel", use_container_width=True, key="scg_kg_cancel"):
                st.session_state.show_scg_kg_dialog = False
                st.rerun()

    # ── Step A: KG processing (output rendered in the Step 04 section below)
    if st.session_state.get('scg_kg_model_selected') and uploaded_files and len(uploaded_files) > 0:
        scg_kg_model = st.session_state.scg_kg_model_selected
        scg_kg_graph_type_ui = st.session_state.get('scg_kg_graph_type_selected', "Full Graph (with Datasets)")
        graph_type_map = {
            "Full Graph (with Datasets)": "with_datasets",
            "Knowledge Graph Only (without Datasets)": "without_datasets"
        }
        scg_kg_graph_type = graph_type_map[scg_kg_graph_type_ui]
        st.session_state.scg_kg_model_selected = None
        st.session_state.scg_kg_graph_type_selected = None

        st.info(f"[Enhanced SCG] Model: **{scg_kg_model}** · Graph: **{scg_kg_graph_type_ui}**")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for idx, file in enumerate(uploaded_files):
            progress_text.text(f"Processing {file.name}… ({idx+1}/{total_files})")
            progress_bar.progress((idx + 1) / total_files)
            try:
                file.seek(0)
                file_content = file.read()
                temp_filename = f"temp_scg_{idx}_{file.name.replace(' ', '_')}"
                with open(temp_filename, "wb") as f:
                    f.write(file_content)
                nodes, relations, datasets, keywords_metadata = process_enhanced(
                    temp_filename, k=k, filter_variables=filter_variables,
                    llm_model=scg_kg_model, use_gpt4_datasets=use_gpt4_datasets
                )
                if file.name not in st.session_state.scg_processed_pdfs:
                    st.session_state.scg_processed_pdfs[file.name] = {
                        'nodes': nodes, 'relations': relations, 'datasets': datasets,
                        'graph_type': scg_kg_graph_type
                    }
                else:
                    st.session_state.scg_processed_pdfs[file.name]['nodes'].extend(nodes)
                    st.session_state.scg_processed_pdfs[file.name]['relations'].extend(relations)
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")

        progress_text.empty()
        progress_bar.empty()
        st.success(f"✓ [Enhanced SCG] Knowledge graph generated for {total_files} file(s) — see Step 04 below.")

        # Auto-advance to the CG step — no extra click needed.
        st.session_state.show_scg_cg_dialog = True
        st.rerun()

    # ── Step B: CG configuration (independent run — its output is NOT displayed separately)
    if st.session_state.show_scg_cg_dialog:
        st.markdown("""
        <div style="
            background:#FFF5F0;
            border:1.5px solid #E67E22;
            border-left:4px solid #C0392B;
            border-radius:6px;
            padding:1.25rem 1.5rem 0.75rem 1.5rem;
            margin-top:0.5rem;
            margin-bottom:0.25rem;
        ">
            <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D3D74;margin-bottom:0.2rem;">
                ⟶ Enhanced SCG — Causal Graph Configuration
            </div>
            <div style="font-size:0.76rem;color:#84888D;letter-spacing:0.04em;">
                The LLM will analyse this run's own KG edges to identify causal relationships — not shown separately, it feeds straight into the final graph in Step 04
            </div>
        </div>
        """, unsafe_allow_html=True)

        scg_cg_model = st.selectbox(
            "LLM Model for Causal Extraction",
            options=["llama3", "mistral:7b", "llama3:latest", "gemma3:12b"],
            index=0,
            key="scg_cg_model_dialog",
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✓ Confirm", use_container_width=True, key="scg_cg_confirm"):
                st.session_state.show_scg_cg_dialog = False
                st.session_state.scg_cg_model_selected = scg_cg_model
                st.rerun()
        with col_cancel:
            if st.button("✕ Cancel", use_container_width=True, key="scg_cg_cancel"):
                st.session_state.show_scg_cg_dialog = False
                st.rerun()

    # ── Step B: CG processing
    if st.session_state.get('scg_cg_model_selected'):
        scg_cg_model = st.session_state.scg_cg_model_selected
        st.session_state.scg_cg_model_selected = None

        scg_all_kg_edges = []
        scg_all_nodes = []
        scg_all_datasets = []
        for data in st.session_state.scg_processed_pdfs.values():
            scg_all_kg_edges.extend(data.get('relations', []))
            scg_all_nodes.extend(data.get('nodes', []))
            scg_all_datasets.extend(data.get('datasets', []))

        if scg_all_kg_edges:
            scg_extra_variables = extract_dataset_variables(scg_all_nodes, scg_all_datasets)
            if scg_extra_variables:
                st.info(f"[Enhanced SCG] Extracting causal relationships from {len(scg_all_kg_edges)} KG edge(s) "
                        f"(+ {len(scg_extra_variables)} dataset-extraction variable(s)) using **{scg_cg_model}**…")
            else:
                st.info(f"[Enhanced SCG] Extracting causal relationships from {len(scg_all_kg_edges)} KG edge(s) using **{scg_cg_model}**…")
            with st.spinner("Extracting causal relationships…"):
                try:
                    st.session_state.scg_cg_relations = extract_causal_relations(
                        scg_all_kg_edges, model=scg_cg_model, extra_variables=scg_extra_variables
                    )
                except Exception as e:
                    st.error(f"Causal extraction error: {str(e)}")
            st.rerun()
        else:
            st.warning("No KG edges found for Enhanced SCG. Try generating the KG again above.")

    # ── Step C: dataset upload + variable mapping, restricted to the
    #    variables the LLM Causal Graph (above) actually identified as causal.
    if st.session_state.scg_cg_relations:
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<span class="section-label">Enhanced SCG — Dataset &amp; Variable Mapping</span>', unsafe_allow_html=True)

        scg_dataset_file = st.file_uploader(
            "Upload a real dataset (CSV) to build the Enhanced SCG from data",
            type=["csv"],
            key="scg_dataset_uploader",
            help="A numeric dataset whose columns correspond to the LLM-identified causal variables."
        )
        if scg_dataset_file is not None:
            try:
                st.session_state.scg_validation_dataset = load_dataset(scg_dataset_file)
                st.session_state.scg_validation_dataset_name = scg_dataset_file.name
                st.session_state.scg_validation_dataset_file_id = scg_dataset_file.file_id
            except Exception as e:
                st.error(f"Dataset error: {str(e)}")
                st.session_state.scg_validation_dataset = None

        scg_dataset_ready = st.session_state.scg_validation_dataset is not None
        if scg_dataset_ready:
            st.success(f"✓ Dataset loaded: {st.session_state.scg_validation_dataset_name} ({len(st.session_state.scg_validation_dataset)} rows)")

        # Auto-open Step 1 (dataset + methods) as soon as a (new) dataset is
        # loaded — no extra button click needed, same as the KG step's
        # auto-advance. Tracked by Streamlit's file_id (not filename) — a
        # fresh id is assigned on every upload action, even re-dragging the
        # exact same file/name, so Cancel doesn't reopen on its own but a
        # genuine re-upload always does, regardless of whether the name repeats.
        if scg_dataset_ready and st.session_state.scg_discovery_dialog_handled_for != st.session_state.scg_validation_dataset_file_id:
            st.session_state.show_scg_dataset_dialog = True
            st.session_state.show_scg_discovery_dialog = False
            st.session_state.scg_discovery_dialog_handled_for = st.session_state.scg_validation_dataset_file_id
            # Clear any results from a previous dataset immediately — otherwise
            # a stale graph stays on screen if the new run's mapping ends up
            # under-confirmed, making it look like nothing happened.
            st.session_state.scg_results = None

        # ── Step 1: dataset + method confirmation
        if st.session_state.show_scg_dataset_dialog:
            st.markdown("""
            <div style="
                background:#EAF2FD;
                border:1.5px solid #2ECC71;
                border-left:4px solid #1E8449;
                border-radius:6px;
                padding:1.25rem 1.5rem 0.75rem 1.5rem;
                margin-top:0.5rem;
                margin-bottom:0.25rem;
            ">
                <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D3D74;margin-bottom:0.2rem;">
                    ⟶ Enhanced SCG — Dataset Confirmation
                </div>
                <div style="font-size:0.76rem;color:#84888D;letter-spacing:0.04em;">
                    Choose which discovery method(s) to run on this dataset
                </div>
            </div>
            """, unsafe_allow_html=True)

            scg_ds_df = st.session_state.scg_validation_dataset
            scg_time_col = detect_time_column(scg_ds_df)

            scg_available_methods = ["pc", "lingam", "fci", "cdnod", "daggnn", "ges"]
            if scg_time_col:
                scg_available_methods += ["pcmci", "pcmci_plus", "tcdf", "lpcmci"]
                st.caption(f"Time column detected: **{scg_time_col}** — lagged methods (PCMCI, PCMCI+, TCDF, LPCMCI) are also available.")
            else:
                st.caption("No time column detected — only PC, LiNGAM, FCI, CD-NOD, DAG-GNN, and GES (non-time-series) are available for this dataset.")

            scg_methods = st.multiselect(
                "Causal discovery method(s) to run",
                options=scg_available_methods,
                default=[scg_available_methods[0]],
                key="scg_methods_dialog",
                format_func=_method_display,
            )

            scg_needs_alpha = bool(set(scg_methods) & PC_ALPHA_METHODS)
            scg_needs_tau_max = bool(set(scg_methods) & TAU_MAX_METHODS)

            if scg_needs_alpha:
                scg_pc_alpha = st.number_input(
                    "Significance level (pc_alpha)",
                    min_value=0.001, max_value=0.5, value=0.05, step=0.01,
                    key="scg_pc_alpha_dialog",
                    help="Used by PC, PCMCI, PCMCI+, FCI, CD-NOD, LPCMCI.",
                )
            else:
                scg_pc_alpha = st.session_state.scg_pc_alpha_selected

            if scg_needs_tau_max:
                scg_tau_max = st.number_input(
                    "Maximum time lag (tau_max)",
                    min_value=1, max_value=60, value=21, step=1,
                    key="scg_tau_max_dialog",
                    help="Used by PCMCI, PCMCI+, TCDF, LPCMCI. Default 21 matches "
                         "the lag used in Hossain et al. (PerCom 2025) for Arctic SIE.",
                )
            else:
                scg_tau_max = st.session_state.scg_tau_max_selected

            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✓ Confirm", use_container_width=True, key="scg_dataset_confirm", disabled=not scg_methods):
                    st.session_state.show_scg_dataset_dialog = False
                    st.session_state.scg_methods_selected = scg_methods
                    st.session_state.scg_pc_alpha_selected = scg_pc_alpha
                    st.session_state.scg_tau_max_selected = scg_tau_max
                    st.session_state.show_scg_discovery_dialog = True
                    st.rerun()
            with col_cancel:
                if st.button("✕ Cancel", use_container_width=True, key="scg_dataset_cancel"):
                    st.session_state.show_scg_dataset_dialog = False
                    st.rerun()

        # ── Step 2: variable mapping confirmation
        if st.session_state.show_scg_discovery_dialog:
            st.markdown("""
            <div style="
                background:#EAF2FD;
                border:1.5px solid #2ECC71;
                border-left:4px solid #1E8449;
                border-radius:6px;
                padding:1.25rem 1.5rem 0.75rem 1.5rem;
                margin-top:0.5rem;
                margin-bottom:0.25rem;
            ">
                <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:600;color:#0D3D74;margin-bottom:0.2rem;">
                    ⟶ Enhanced SCG — Variable Mapping Confirmation
                </div>
                <div style="font-size:0.76rem;color:#84888D;letter-spacing:0.04em;">
                    Only variables the LLM Causal Graph identified as causal are offered here
                </div>
            </div>
            """, unsafe_allow_html=True)

            scg_ds_df = st.session_state.scg_validation_dataset
            scg_ds_columns = numeric_columns(scg_ds_df)

            scg_cg_nodes = sorted({
                clean_node_name(n)
                for r in st.session_state.scg_cg_relations
                for n in (r['cause'], r['effect'])
            })

            scg_suggested = _cached_variable_mapping(scg_cg_nodes, scg_ds_columns)
            scg_mapping = {}
            for node in scg_cg_nodes:
                options = ["(none)"] + scg_ds_columns
                default_col = scg_suggested.get(node)
                default_idx = options.index(default_col) if default_col in options else 0
                chosen = st.selectbox(
                    prettify_node_name(node), options=options, index=default_idx,
                    key=f"scg_map_{st.session_state.scg_validation_dataset_file_id}_{node}",
                    format_func=lambda c: c if c == "(none)" else prettify_column_name(c)
                )
                if chosen != "(none)":
                    scg_mapping[node] = chosen

            # ── Custom variables — not identified by the LLM Causal Graph,
            # but the user can still add one by name and map it to a dataset
            # column; it flows into scg_mapping exactly like the nodes above.
            st.markdown('<span class="section-label">Add a custom variable (optional)</span>', unsafe_allow_html=True)
            st.caption("Not identified by the LLM causal graph? Add it here and map it to a dataset column — it will be passed to the causal discovery methods exactly like the variables above.")

            scg_extra_key = f"scg_extra_vars_{st.session_state.scg_validation_dataset_file_id}"
            st.session_state.setdefault(scg_extra_key, [])

            scg_add_col1, scg_add_col2, scg_add_col3 = st.columns([3, 3, 1])
            with scg_add_col1:
                scg_new_var_name = st.text_input(
                    "Variable name", key=f"scg_new_var_name_{st.session_state.scg_validation_dataset_file_id}",
                    label_visibility="collapsed", placeholder="Variable name",
                )
            with scg_add_col2:
                scg_new_var_col = st.selectbox(
                    "Dataset column", options=["(choose column)"] + scg_ds_columns,
                    key=f"scg_new_var_col_{st.session_state.scg_validation_dataset_file_id}",
                    label_visibility="collapsed",
                    format_func=lambda c: c if c == "(choose column)" else prettify_column_name(c),
                )
            with scg_add_col3:
                if st.button("+ Add", key=f"scg_add_var_{st.session_state.scg_validation_dataset_file_id}"):
                    if scg_new_var_name.strip() and scg_new_var_col != "(choose column)":
                        st.session_state[scg_extra_key].append({'name': scg_new_var_name.strip(), 'column': scg_new_var_col})
                        st.rerun()
                    else:
                        st.warning("Enter a variable name and choose a column first.")

            for scg_extra_idx, scg_extra in enumerate(st.session_state[scg_extra_key]):
                scg_extra_row1, scg_extra_row2 = st.columns([9, 1])
                with scg_extra_row1:
                    st.markdown(f"— **{scg_extra['name']}** → `{prettify_column_name(scg_extra['column'])}`")
                with scg_extra_row2:
                    if st.button("✕", key=f"scg_remove_var_{st.session_state.scg_validation_dataset_file_id}_{scg_extra_idx}"):
                        st.session_state[scg_extra_key].pop(scg_extra_idx)
                        st.rerun()

            for scg_extra in st.session_state[scg_extra_key]:
                scg_mapping[scg_extra['name']] = scg_extra['column']

            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✓ Confirm", use_container_width=True, key="scg_discovery_confirm"):
                    st.session_state.show_scg_discovery_dialog = False
                    st.session_state.scg_run_request = {
                        'methods': st.session_state.scg_methods_selected,
                        'mapping': scg_mapping,
                        'pc_alpha': st.session_state.scg_pc_alpha_selected,
                        'tau_max': st.session_state.scg_tau_max_selected,
                    }
                    st.rerun()
            with col_cancel:
                if st.button("✕ Cancel", use_container_width=True, key="scg_discovery_cancel"):
                    st.session_state.show_scg_discovery_dialog = False
                    st.rerun()

        # ── Discovery processing (output rendered in the Step 04 section below)
        if st.session_state.scg_run_request:
            request = st.session_state.scg_run_request
            st.session_state.scg_run_request = None

            mapped_count = len([v for v in request['mapping'].values() if v])
            if mapped_count < 2:
                st.warning("Need at least 2 confirmed variable mappings to run the Enhanced SCG discovery.")
            else:
                st.info(f"Running {', '.join(request['methods'])} on {mapped_count} LLM-vetted variable(s)…")
                with st.spinner("Running statistical causal discovery…"):
                    try:
                        scg_result = run_causal_discovery(
                            st.session_state.scg_validation_dataset,
                            request['mapping'],
                            methods=request['methods'],
                            pc_alpha=request['pc_alpha'],
                            tau_max=request['tau_max'],
                        )
                        st.session_state.scg_results = scg_result
                    except Exception as e:
                        st.error(f"Causal discovery error: {str(e)}")
                st.rerun()

    # ── KG Processing
    if 'kg_model_selected' in st.session_state and st.session_state.kg_model_selected and uploaded_files and len(uploaded_files) > 0:
        kg_model = st.session_state.kg_model_selected
        kg_graph_type_ui = st.session_state.get('kg_graph_type_selected', "Full Graph (with Datasets)")
        graph_type_map = {
            "Full Graph (with Datasets)": "with_datasets",
            "Knowledge Graph Only (without Datasets)": "without_datasets"
        }
        kg_graph_type = graph_type_map[kg_graph_type_ui]
        st.session_state.kg_model_selected = None
        st.session_state.kg_graph_type_selected = None

        st.info(f"Model: **{kg_model}** · Graph: **{kg_graph_type_ui}**")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        all_keywords = []
        all_datasets = []

        for idx, file in enumerate(uploaded_files):
            progress_text.text(f"Processing {file.name}… ({idx+1}/{total_files})")
            progress_bar.progress((idx + 1) / total_files)
            try:
                file.seek(0)
                file_content = file.read()
                temp_filename = f"temp_{idx}_{file.name.replace(' ', '_')}"
                with open(temp_filename, "wb") as f:
                    f.write(file_content)
                nodes, relations, datasets, keywords_metadata = process(
                    temp_filename, k=k, filter_variables=filter_variables,
                    llm_model=kg_model, use_gpt4_datasets=use_gpt4_datasets
                )
                if keywords_metadata and keywords_metadata.get('from_keywords_section'):
                    st.success(f"Keywords section found in {file.name} — {keywords_metadata['total_found']} extracted.")
                elif keywords_metadata:
                    st.info(f"No keywords section in {file.name}. Using {keywords_metadata.get('method', 'algorithmic extraction')}.")

                if keywords_metadata and keywords_metadata.get('filtering_applied'):
                    original_kw = keywords_metadata.get('original_keywords', [])
                    filtered_kw = keywords_metadata.get('filtered_keywords', [])
                    removed_kw = keywords_metadata.get('removed_keywords', [])
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Original Keywords", len(original_kw))
                    with col2: st.metric("Climate Variables", len(filtered_kw), delta=f"{len(filtered_kw)/len(original_kw)*100:.1f}%" if original_kw else "0%")
                    with col3: st.metric("Filtered Out", len(removed_kw), delta=f"-{len(removed_kw)/len(original_kw)*100:.1f}%" if original_kw else "0%")
                    with st.expander(f"Variable Filtering Details — {file.name}"):
                        col_kept, col_removed = st.columns(2)
                        with col_kept:
                            st.markdown("**Variables Kept:**")
                            st.write(", ".join(filtered_kw[:20]) + (f" … +{len(filtered_kw)-20} more" if len(filtered_kw) > 20 else "") if filtered_kw else "None")
                        with col_removed:
                            st.markdown("**Removed:**")
                            st.write(", ".join(removed_kw[:10]) + (f" … +{len(removed_kw)-10} more" if len(removed_kw) > 10 else "") if removed_kw else "None")

                extraction_stats = keywords_metadata.get('extraction_stats', {}) if keywords_metadata else {}
                if file.name not in st.session_state.processed_pdfs:
                    st.session_state.processed_pdfs[file.name] = {
                        'nodes': nodes, 'relations': relations, 'datasets': datasets,
                        'keywords_metadata': keywords_metadata, 'used_gpt4': use_gpt4_datasets,
                        'extraction_cost': extraction_stats.get('total_cost', 0),
                        'graph_type': kg_graph_type
                    }
                else:
                    st.session_state.processed_pdfs[file.name]['nodes'].extend(nodes)
                    st.session_state.processed_pdfs[file.name]['relations'].extend(relations)

                all_keywords.extend(nodes)
                if datasets:
                    for ds in datasets:
                        if ds.get('source') != 'Not specified':
                            all_datasets.append(ds.get('source'))
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

        progress_text.empty()
        progress_bar.empty()
        st.success(f"✓ Knowledge graphs generated for {total_files} file(s).")
        st.info("Tip — click 'Generate Causal Graph (LLM)' below to discover causal relationships from the KG edges.")

        if filter_variables:
            st.markdown('<span class="section-label">Variable Filtering Summary</span>', unsafe_allow_html=True)
            total_original = total_kept = total_removed = 0
            for pdf_name, pdf_data in st.session_state.processed_pdfs.items():
                if pdf_data.get('keywords_metadata', {}).get('filtering_applied'):
                    meta = pdf_data['keywords_metadata']
                    total_original += len(meta.get('original_keywords', []))
                    total_kept += len(meta.get('filtered_keywords', []))
                    total_removed += len(meta.get('removed_keywords', []))
            if total_original > 0:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Total Keywords", total_original)
                with c2: st.metric("Climate Variables", total_kept, delta=f"{total_kept/total_original*100:.1f}%")
                with c3: st.metric("Filtered Out", total_removed, delta=f"-{total_removed/total_original*100:.1f}%")
                with c4: st.metric("Retention Rate", f"{total_kept/total_original*100:.1f}%")

        if all_keywords:
            st.markdown('<span class="section-label">Extracted Climate Variables</span>', unsafe_allow_html=True)
            unique_kw = list(set(all_keywords))
            kw_html = "".join(f'<span class="kw-tag">{kw}</span>' for kw in unique_kw[:30])
            if len(unique_kw) > 30:
                kw_html += f'<span class="kw-tag-light">+{len(unique_kw)-30} more</span>'
            st.markdown(kw_html, unsafe_allow_html=True)

        if all_datasets:
            st.markdown('<span class="section-label">Datasets Identified</span>', unsafe_allow_html=True)
            for ds in set(all_datasets):
                st.markdown(f'<div class="polar-info-row">◎ {ds}</div>', unsafe_allow_html=True)

        st.rerun()  # re-render so CG button reads populated processed_pdfs and becomes enabled


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Q&A
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div id="section-qa"></div>', unsafe_allow_html=True)
st.markdown('<span class="section-label">Step 02</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Document <em>Q&A</em></div>', unsafe_allow_html=True)

if qa_system.list_documents():
    st.success(f"✓ Q&A System Ready — {len(qa_system.list_documents())} document(s) indexed")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("Clear Chat History", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    with col_r2:
        if st.button("Reset Q&A System", use_container_width=True, key="reset_qa"):
            qa_system.reset_and_reload()
            st.session_state.databases = []
            st.session_state.chat_history = []
            st.rerun()
else:
    st.warning("No documents indexed — upload PDFs and click 'Send to Q&A' above.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<span class="section-label">Indexed Documents</span>', unsafe_allow_html=True)
    if st.session_state.databases:
        for db in st.session_state.databases:
            st.markdown(f'<div class="doc-item">{db}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state" style="padding:2rem;"><span class="empty-state-glyph">&#8599;</span><div class="empty-state-text">No documents indexed yet.</div></div>', unsafe_allow_html=True)

with col2:
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-bubble-user"><span class="chat-label chat-label-user">You</span>{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-assistant"><span class="chat-label chat-label-ai">PolarKD</span>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state"><span class="empty-state-glyph">&#8596;</span><div class="empty-state-title">Start a conversation</div><div class="empty-state-text">Try: "What datasets were used?" &middot; "Summarise the findings." &middot; "What methods were employed?" &middot; "What is the study time period?"</div></div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question…",
            placeholder="Type your question here…",
            label_visibility="collapsed"
        )
        col_in, col_send = st.columns([4, 1])
        with col_send:
            submit = st.form_submit_button("Send →", use_container_width=True)
        if submit and user_input:
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            with st.spinner("Thinking…"):
                try:
                    response = qa_system.answer_question(user_input)
                except Exception as e:
                    response = f"Error: {str(e)}. Ensure Ollama is running and accessible."
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div id="section-kg"></div>', unsafe_allow_html=True)
st.markdown('<span class="section-label">Step 03</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Knowledge <em>Graph</em></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="graph-legend">'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#0D3D74;"></div> Entity</div>'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#377FD0;"></div> Relationship</div>'
    '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#377FD0;"></div> Concept</div>'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.processed_pdfs:
    st.markdown('<span class="section-label">Processing Summary</span>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Files Processed", len(st.session_state.processed_pdfs))
    with c2: st.metric("Total Keywords", sum(len(d.get('nodes', [])) for d in st.session_state.processed_pdfs.values()))
    with c3: st.metric("Total Relations", sum(len(d.get('relations', [])) for d in st.session_state.processed_pdfs.values()))
    with c4: st.metric("Files", len(st.session_state.processed_pdfs))

    dataset_filter = display_dataset_filter()
    display_datasets_section(st.session_state.processed_pdfs, dataset_filter)
    display_cost_summary(st.session_state.processed_pdfs)

    all_keywords = []
    keywords_by_file = {}
    for filename, data in st.session_state.processed_pdfs.items():
        fk = data.get('nodes', [])
        all_keywords.extend(fk)
        keywords_by_file[filename] = fk

    if all_keywords:
        st.markdown('<span class="section-label">Extracted Keywords</span>', unsafe_allow_html=True)
        for filename, keywords in keywords_by_file.items():
            st.markdown(f"**{filename}**")
            kw_html = "".join(f'<span class="kw-tag">{kw}</span>' for kw in keywords[:10])
            st.markdown(kw_html, unsafe_allow_html=True)
        st.markdown(f"<small style='color:#84888D;font-size:0.75rem;'>Total unique keywords: {len(set(all_keywords))}</small>", unsafe_allow_html=True)

    try:
        all_nodes, all_relations, all_datasets = [], [], []

        st.markdown(f"<small style='color:#84888D;'>Combining data from {len(st.session_state.processed_pdfs)} files…</small>", unsafe_allow_html=True)
        for filename, data in st.session_state.processed_pdfs.items():
            nodes = data.get('nodes', [])
            relations = data.get('relations', [])
            file_datasets = data.get('datasets', [])
            st.markdown(f"<small style='color:#84888D;'>— {filename}: {len(nodes)} nodes · {len(relations)} relations · {len(file_datasets) if file_datasets else 0} dataset(s)</small>", unsafe_allow_html=True)
            all_nodes.extend(nodes)
            all_relations.extend(relations)
            if file_datasets:
                for ds in file_datasets:
                    if ds.get('source') != 'Not specified':
                        all_datasets.append(ds)

        if all_nodes and all_relations:
            st.markdown(f"<small style='color:#84888D;'>Total: {len(all_nodes)} nodes · {len(all_relations)} relations · {len(all_datasets)} dataset(s)</small>", unsafe_allow_html=True)
            graph_type_to_use = 'with_datasets'
            if st.session_state.processed_pdfs:
                first_file = list(st.session_state.processed_pdfs.keys())[0]
                graph_type_to_use = st.session_state.processed_pdfs[first_file].get('graph_type', 'with_datasets')
            st.info(f"Graph mode: **{graph_type_to_use.replace('_', ' ').title()}**")
            html_string = _cached_kg_graph_html(all_nodes, all_relations, all_datasets, graph_type_to_use)
            st.components.v1.html(html_string, height=500, scrolling=True)

            st.markdown('<span class="section-label">Graph Statistics</span>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Unique Nodes", len(set(all_nodes)))
            with c2: st.metric("Total Relations", len(all_relations))
            with c3: st.metric("Datasets Found", len(all_datasets))
            with c4:
                avg = len(all_relations) // len(st.session_state.processed_pdfs) if st.session_state.processed_pdfs else 0
                st.metric("Avg Relations / File", avg)

    except Exception as e:
        st.error(f"Neo4j error: {str(e)}")
        st.info("Please check your Neo4j credentials.")

else:
    st.markdown(
        '<div class="empty-state">'
        '<svg width="120" height="120" viewBox="0 0 200 200" style="opacity:0.25;margin-bottom:1.5rem;">'
        '<circle cx="100" cy="50" r="14" fill="#0D3D74"/>'
        '<circle cx="50" cy="110" r="14" fill="#0D3D74"/>'
        '<circle cx="150" cy="110" r="14" fill="#0D3D74"/>'
        '<circle cx="75" cy="160" r="14" fill="#0D3D74"/>'
        '<circle cx="125" cy="160" r="14" fill="#0D3D74"/>'
        '<circle cx="100" cy="100" r="18" fill="#377FD0"/>'
        '<line x1="100" y1="100" x2="100" y2="50" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="50" y2="110" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="150" y2="110" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="75" y2="160" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="125" y2="160" stroke="#B8CCE8" stroke-width="1.5"/>'
        '</svg>'
        '<div class="empty-state-title">No graph generated yet</div>'
        '<div class="empty-state-text">Upload PDFs and click "Generate Knowledge Graph" to visualise extracted climate variables and their relationships.</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — LLM ENHANCED STRUCTURAL CAUSAL GRAPH
# ══════════════════════════════════════════════════════════════════════════
# Pure output section — the trigger button, config dialogs, and processing
# logic all live in col2 above (next to the Knowledge Graph button), same
# trigger/result split Step 03 already uses. This block only reads scg_*
# session_state and renders it: first the independently-generated KG (same
# style as Step 03), then the final LLM-enhanced statistical causal graph.

st.markdown("---")
st.markdown('<div id="section-scg"></div>', unsafe_allow_html=True)
st.markdown('<span class="section-label">Step 04</span>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">LLM Enhanced <em>Structural Causal Graph</em></div>', unsafe_allow_html=True)

# ── KG output (same display as Step 03, scoped to this section's own KG run)
if st.session_state.scg_processed_pdfs:
    scg_all_nodes, scg_all_relations, scg_all_datasets = [], [], []
    for data in st.session_state.scg_processed_pdfs.values():
        scg_all_nodes.extend(data.get('nodes', []))
        scg_all_relations.extend(data.get('relations', []))
        for ds in data.get('datasets', []) or []:
            if ds.get('source') != 'Not specified':
                scg_all_datasets.append(ds)

    st.markdown('<span class="section-label">Knowledge Graph</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Files Processed", len(st.session_state.scg_processed_pdfs))
    with c2: st.metric("Unique Nodes", len(set(scg_all_nodes)))
    with c3: st.metric("Total Relations", len(scg_all_relations))

    if scg_all_nodes:
        kw_html = "".join(f'<span class="kw-tag">{kw}</span>' for kw in list(set(scg_all_nodes))[:30])
        st.markdown(kw_html, unsafe_allow_html=True)

    if scg_all_nodes and scg_all_relations:
        try:
            scg_graph_type = list(st.session_state.scg_processed_pdfs.values())[0].get('graph_type', 'with_datasets')
            scg_kg_html = _cached_kg_graph_html(scg_all_nodes, scg_all_relations, scg_all_datasets, scg_graph_type)
            st.components.v1.html(scg_kg_html, height=500, scrolling=True)
        except Exception as e:
            st.error(f"Graph rendering error: {str(e)}")
else:
    st.markdown(
        '<div class="empty-state">'
        '<svg width="120" height="120" viewBox="0 0 200 200" style="opacity:0.25;margin-bottom:1.5rem;">'
        '<circle cx="100" cy="50" r="14" fill="#0D3D74"/>'
        '<circle cx="50" cy="110" r="14" fill="#0D3D74"/>'
        '<circle cx="150" cy="110" r="14" fill="#0D3D74"/>'
        '<circle cx="75" cy="160" r="14" fill="#0D3D74"/>'
        '<circle cx="125" cy="160" r="14" fill="#0D3D74"/>'
        '<circle cx="100" cy="100" r="18" fill="#377FD0"/>'
        '<line x1="100" y1="100" x2="100" y2="50" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="50" y2="110" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="150" y2="110" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="75" y2="160" stroke="#B8CCE8" stroke-width="1.5"/>'
        '<line x1="100" y1="100" x2="125" y2="160" stroke="#B8CCE8" stroke-width="1.5"/>'
        '</svg>'
        '<div class="empty-state-title">No Knowledge Graph yet</div>'
        '<div class="empty-state-text">Click "Generate Structural Causal Graph" above (Step 01 action panel) to start this section\'s independent KG → Causal Graph → discovery pipeline.</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Final output: the Enhanced SCG itself
if st.session_state.scg_results:
    scg_results = st.session_state.scg_results

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="section-label" style="font-weight:700;font-size:0.85rem;">FINAL OUTPUT (CAUSAL GRAPH)</span>', unsafe_allow_html=True)

    st.markdown(
        '<div class="graph-legend">'
        '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#C0392B;"></div> Root Cause</div>'
        '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#8E44AD;"></div> Intermediate</div>'
        '<div class="graph-legend-item"><div class="graph-legend-dot" style="background:#E67E22;"></div> Terminal Effect</div>'
        '<div class="graph-legend-item" style="font-size:0.68rem;color:#84888D !important;">Arrows show causal direction →</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<span class="section-label">Variables Used</span>', unsafe_allow_html=True)
    st.caption(", ".join(scg_results['columns_used']))

    for method, causal_rels in scg_results['kg_edges'].items():
        with st.expander(f"{_method_display(method)} — {len(causal_rels)} causal edge(s)", expanded=True):
            if not causal_rels:
                st.write("No significant edges found at this significance level.")
                continue

            scg_all_causes = list({r['cause'] for r in causal_rels})
            scg_all_effects = list({r['effect'] for r in causal_rels})
            scg_root_causes = [c for c in scg_all_causes if c not in {r['effect'] for r in causal_rels}]
            scg_terminal_effects = [e for e in scg_all_effects if e not in {r['cause'] for r in causal_rels}]
            scg_avg_conf = sum(r['confidence'] for r in causal_rels) / len(causal_rels)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Causal Pairs", len(causal_rels))
            with c2: st.metric("Unique Variables", len(set(scg_all_causes) | set(scg_all_effects)))
            with c3: st.metric("Root Causes", len(scg_root_causes))
            with c4: st.metric("Avg Confidence", f"{scg_avg_conf:.2f}")

            if scg_root_causes:
                st.markdown('<span class="section-label">Root Causes Identified</span>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="causal-cause-tag">{prettify_node_name(rc)}</span>' for rc in scg_root_causes), unsafe_allow_html=True)

            if scg_terminal_effects:
                st.markdown('<span class="section-label">Terminal Effects</span>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="causal-effect-tag">{prettify_node_name(te)}</span>' for te in scg_terminal_effects), unsafe_allow_html=True)

            try:
                scg_html = _cached_causal_graph_html(causal_rels, f"enhanced_scg_{method}.html")
                st.components.v1.html(scg_html, height=500, scrolling=True)
            except Exception as e:
                st.error(f"Graph rendering error: {str(e)}")

            for rel in sorted(causal_rels, key=lambda x: -x['confidence']):
                conf_color = "#C0392B" if rel['confidence'] >= 0.7 else "#E67E22" if rel['confidence'] >= 0.4 else "#F39C12"
                st.markdown(
                    f'<div class="polar-info-row" style="border-left-color:{conf_color};">'
                    f'<b>{prettify_node_name(rel["cause"])}</b> &nbsp;⟶&nbsp; <span style="color:{conf_color};font-weight:600;">{rel["label"]}</span> &nbsp;⟶&nbsp; <b>{prettify_node_name(rel["effect"])}</b>'
                    f'&nbsp;&nbsp;<span style="font-size:0.72rem;color:#84888D;">confidence: {rel["confidence"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        full_graph = scg_results.get('full_graphs', {}).get(method)
        if full_graph:
            with st.expander(f"{_method_display(method)} — Tigramite Causal Graph (all links)", expanded=False):
                st.caption(
                    "Every link Tigramite found — including contemporaneous/ambiguous marks "
                    "('x-x') that don't survive into the resolved KG edges above."
                )
                try:
                    fig, _ax = plot_full_causal_graph(
                        graph=full_graph['graph'],
                        val_matrix=full_graph['val_matrix'],
                        var_names=full_graph['var_names'],
                        title=f"{_method_display(method)} — Full Causal Graph",
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Full graph rendering error: {str(e)}")
elif st.session_state.scg_processed_pdfs:
    # Only show this placeholder once KG (+ CG) has actually started — if
    # nothing has been generated yet, the KG empty-state above already tells
    # the user where to start, so showing a second "click here" would be
    # redundant.
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="empty-state">'
        '<svg width="120" height="120" viewBox="0 0 200 200" style="opacity:0.25;margin-bottom:1.5rem;">'
        '<circle cx="40" cy="100" r="14" fill="#C0392B"/>'
        '<circle cx="100" cy="60" r="14" fill="#8E44AD"/>'
        '<circle cx="100" cy="140" r="14" fill="#8E44AD"/>'
        '<circle cx="160" cy="100" r="14" fill="#E67E22"/>'
        '<line x1="54" y1="100" x2="86" y2="68" stroke="#C0392B" stroke-width="2"/>'
        '<line x1="54" y1="100" x2="86" y2="132" stroke="#C0392B" stroke-width="2"/>'
        '<line x1="114" y1="60" x2="146" y2="92" stroke="#8E44AD" stroke-width="2"/>'
        '<line x1="114" y1="140" x2="146" y2="108" stroke="#8E44AD" stroke-width="2"/>'
        '</svg>'
        '<div class="empty-state-title">No causal graph yet</div>'
        '<div class="empty-state-text">Upload a dataset above and confirm the method + variable mapping to generate the final Enhanced Structural Causal Graph.</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ─── EXPORT ────────────────────────────────────────────────────────────────
if st.session_state.processed_pdfs:
    st.markdown("---")
    st.markdown('<span class="section-label">Export</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.4rem!important;">Download <em>Results</em></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    all_relations = []
    for data in st.session_state.processed_pdfs.values():
        all_relations.extend(data.get('relations', []))

    with col1:
        if all_relations:
            json_data = json.dumps(all_relations, indent=2)
            st.download_button(
                label="Export KG — JSON",
                data=json_data,
                file_name="knowledge_graph.json",
                mime="application/json",
                use_container_width=True
            )
    with col2:
        if all_relations:
            df = pd.DataFrame(all_relations)
            st.download_button(
                label="Export KG — CSV",
                data=df.to_csv(index=False),
                file_name="knowledge_graph.csv",
                mime="text/csv",
                use_container_width=True
            )
    with col3:
        datasets_csv = export_datasets_to_csv(st.session_state.processed_pdfs)
        if datasets_csv:
            st.download_button(
                label="Export Datasets — CSV",
                data=datasets_csv,
                file_name="extracted_datasets.csv",
                mime="text/csv",
                use_container_width=True
            )
    with col4:
        if st.session_state.scg_cg_relations:
            cg_df = pd.DataFrame(st.session_state.scg_cg_relations)
            st.download_button(
                label="Export Causal — CSV",
                data=cg_df.to_csv(index=False),
                file_name="causal_graph.csv",
                mime="text/csv",
                use_container_width=True
            )


# ─── FOOTER ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="polar-footer">'
    '<div class="polar-footer-brand">PolarKD</div>'
    '<div class="polar-footer-links"><a href="#">About</a><a href="#">Documentation</a><a href="#">Contact</a><a href="#">Privacy</a></div>'
    '<div class="polar-footer-copy">AI-powered polar science document intelligence &middot; iHARP &copy; 2024</div>'
    '</div>',
    unsafe_allow_html=True
)
