import streamlit as st
from keywords_extraction import process
from neo4j_storage import Neo4jConnector
from qa_module import qa_system
from frontend_dataset_display import (
    display_gpt4_toggle,
    display_dataset_filter,
    display_datasets_section,
    display_cost_summary,
    export_datasets_to_csv
)
import os
import base64
from io import BytesIO
import json
import pandas as pd

# Page config - Light theme
st.set_page_config(
    page_title="Polar Knowledge Discovery (PolarKD) Toolkit", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Light Theme
st.markdown("""
<style>
    /* Logo inside header */
    .main-header {
        position: relative !important;
    }

    .main-header img.logo {
        position: absolute !important;
        top: 1rem !important;
        right: 1.5rem !important;
        height: 70px !important;
        width: auto !important;
    }

    /* Force light theme throughout */
    .stApp {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* Remove any dark backgrounds from Streamlit components */
    .main .block-container {
        background-color: #ffffff !important;
    }
    
    .stMarkdown, .stText {
        color: #262730 !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 600;
    }
    
    /* Navigation tabs */
    .nav-tabs {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0;
        padding: 1rem;
        background-color: #f7f8fa;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .nav-tab {
        padding: 0.75rem 2rem;
        background: #ffffff;
        border: 2px solid #e1e4e8;
        border-radius: 8px;
        color: #495057;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .nav-tab:hover {
        background-color: #667eea;
        color: #ffffff;
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    /* Section styling */
    h2 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    
    
    /* Info boxes */
    .stAlert {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeaa7 !important;
    }
    
    /* Error messages */
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border: 1px solid #f5c6cb !important;
    }
    
    /* All buttons with consistent purple gradient */
    .stButton > button,
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3) !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: inline-block !important;
    }
    
    .stButton > button:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="baseButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #5a67d8 0%, #6b4999 100%) !important;
    }
    
    /* Ensure buttons in columns are visible */
    [data-testid="column"] .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Keyword tags */
    .keyword-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Climate variable tags (for filtered keywords) */
    .variable-tag {
        display: inline-block;
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Filtered out tags */
    .filtered-tag {
        display: inline-block;
        background: #e9ecef;
        color: #6c757d;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
        text-decoration: line-through;
    }
    
    /* Database items */
    .database-item {
        background: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #495057;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Chat messages */
    .chat-message {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Graph legend */
    .legend-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #495057;
        font-weight: 500;
    }
    
    .legend-circle {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 2px solid #dee2e6;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Style the file uploader to look like a drag-drop area */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        border: 2px dashed #8b92a3 !important;
        border-radius: 15px !important;
        padding: 3rem !important;
        text-align: center !important;
        min-height: 200px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #667eea !important;
        background: linear-gradient(135deg, #e9ecef 0%, #b8c6db 100%) !important;
    }
    
    /* Hide the browse button */
    [data-testid="stFileUploaderDropzone"] button {
        display: none !important;
    }
    
    /* Style the upload text */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #495057 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    /* Add icon before upload text */
    [data-testid="stFileUploaderDropzoneInstructions"]:before {
        content: "📤" !important;
        display: block !important;
        font-size: 3rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background: #f8f9fa;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: #ffffff !important;
        color: #262730 !important;
        border: 2px solid #e1e4e8 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border-radius: 8px !important;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 2px solid #e1e4e8;
        text-align: center;
        color: #6c757d;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .footer-links {
        margin-bottom: 1rem;
    }
    
    .footer-links a {
        margin: 0 1rem;
        color: #6c757d;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer-links a:hover {
        color: #667eea;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Columns divider */
    [data-testid="column"] {
        padding: 0 1rem;
    }
    
    /* Make all text visible on light background */
    p, span, div, label {
        color: #262730 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(40, 167, 69, 0.4);
    }
    
    /* Form submit button with same purple gradient */
    .stFormSubmitButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: inline-block !important;
    }
    
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #5a67d8 0%, #6b4999 100%) !important;
    }
    
    /* Ensure all interactive elements are visible */
    button[kind="primary"], button[kind="secondary"] {
        opacity: 1 !important;
        visibility: visible !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load logo data for header
logo_data = None
logo_type = None
logo_path = os.path.join(os.path.dirname(__file__), "..", "iharp-logo.jpg")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()
    logo_type = "jpeg"
else:
    # Fallback: try PNG name
    logo_path = os.path.join(os.path.dirname(__file__), "..", "iharp_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_type = "png"

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'databases' not in st.session_state:
    st.session_state.databases = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = {}
if 'current_graph' not in st.session_state:
    st.session_state.current_graph = None

# Header with embedded logo
if logo_data and logo_type:
    st.markdown(f"""
<div class="main-header">
    <img src="data:image/{logo_type};base64,{logo_data}" alt="iHARP Logo" class="logo">
    <h1> Polar Knowledge Discovery (PolarKD) Toolkit</h1>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="main-header">
    <h1> Polar Knowledge Discovery (PolarKD) Toolkit</h1>
</div>
""", unsafe_allow_html=True)

# Navigation tabs (visual only)
st.markdown("""
<div class="nav-tabs">
    <span class="nav-tab">🏠 Home</span>
    <span class="nav-tab">📤 Upload PDFs</span>
    <span class="nav-tab">💬 Q&A</span>
    <span class="nav-tab">🔗 Knowledge Graph</span>
</div>
""", unsafe_allow_html=True)

# Upload PDFs Section
st.markdown("## 📤 Upload PDFs")

col1, col2 = st.columns([1, 1])

with col1:
    # Container for the clickable upload area
    with st.container():
        # This creates the actual file uploader that will be styled to look like drag-drop area
        uploaded_files = st.file_uploader(
            "Drag & Drop PDFs here or Click to Upload",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="visible",
            help="Select multiple PDF files"
        )
    
    # Display uploaded files
    if uploaded_files is not None and len(uploaded_files) > 0:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        for i, file in enumerate(uploaded_files, 1):
            st.write(f"{i}. 📄 {file.name} ({file.size // 1024} KB)")
    else:
        st.info("📌 No files selected yet. Click above or drag files to upload.")

with col2:
    st.markdown("### ⚙️ Actions")
    
    st.info("**📚 Send to Q&A**: Prepare documents for question-answering only")
    st.info("**🔗 Generate Knowledge Graph**: Extract keywords and create visualization only")
    
    # Number of keywords for knowledge graph
    k = st.slider("Keywords to Extract (for Knowledge Graph)", min_value=5, max_value=50, value=15, step=5)

    # Toggle for GPT-4 dataset extraction (PLUGGABLE MODULE)
    use_gpt4_datasets = display_gpt4_toggle()

    # Toggle for variable filtering
    filter_variables = st.checkbox(
        "🔬 Filter to Variables Only",
        value=True,
        help="Extract only measurable variables (temperature, pressure, salinity, etc.) and their relationships. Removes organizations, locations, and methods."
    )

    # Initialize dialog states
    if 'show_qa_dialog' not in st.session_state:
        st.session_state.show_qa_dialog = False
    if 'show_kg_dialog' not in st.session_state:
        st.session_state.show_kg_dialog = False

    # Q&A Button - opens dialog
    if st.button("📚 Send to Q&A", use_container_width=True, key="send_qa", help="Load documents for question-answering"):
        if uploaded_files and len(uploaded_files) > 0:
            st.session_state.show_qa_dialog = True
            st.session_state.show_kg_dialog = False
        else:
            st.warning("⚠️ Please upload files first")

    # Q&A Dialog
    if st.session_state.show_qa_dialog:
        with st.container():
            st.markdown("---")
            st.markdown("### 📚 Q&A Configuration")
            qa_model = st.selectbox(
                "Select LLM Model for Q&A",
                options=["mistral:7b", "qwen2.5:7b", "llama3:latest", "gemma3:12b"],
                index=0,
                key="qa_model_dialog",
                help="Select the Ollama model for answering questions."
            )

            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Confirm", use_container_width=True, key="qa_confirm"):
                    st.session_state.show_qa_dialog = False
                    qa_system.set_model(qa_model)
                    st.info(f"Using model: **{qa_model}** for Q&A")
                    with st.spinner("Adding documents to Q&A system..."):
                        added_count = 0
                        for file in uploaded_files:
                            if file.name not in st.session_state.databases:
                                file.seek(0)
                                temp_path = f"temp_qa_{file.name}"
                                with open(temp_path, "wb") as f:
                                    f.write(file.read())
                                if qa_system.add_document(file.name, pdf_path=temp_path):
                                    st.session_state.databases.append(file.name)
                                    added_count += 1
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                        if added_count > 0:
                            st.success(f"✅ {added_count} file(s) added to Q&A system!")
                        else:
                            st.info("Files already in Q&A system")
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True, key="qa_cancel"):
                    st.session_state.show_qa_dialog = False
                    st.rerun()

    # Knowledge Graph Button - opens dialog
    if st.button("🔗 Generate Knowledge Graph", use_container_width=True, key="gen_kg", help="Extract keywords and create graph visualization"):
        if uploaded_files and len(uploaded_files) > 0:
            st.session_state.show_kg_dialog = True
            st.session_state.show_qa_dialog = False
        else:
            st.warning("⚠️ Please upload files first")

    # Knowledge Graph Dialog
    if st.session_state.show_kg_dialog:
        with st.container():
            st.markdown("---")
            st.markdown("### 🔗 Knowledge Graph Configuration")
            kg_model = st.selectbox(
                "Select LLM Model for Relation Extraction",
                options=["mistral:7b", "qwen2.5:7b", "llama3:latest", "gemma3:12b"],
                index=0,
                key="kg_model_dialog",
                help="Select the Ollama model for extracting relationships."
            )

            kg_graph_type = st.selectbox(
                "Graph Visualization Type",
                options=["Full Graph (with Datasets)", "Knowledge Graph Only (without Datasets)"],
                index=0,
                key="kg_graph_type_dialog",
                help="Choose whether to include dataset nodes in the graph visualization."
            )

            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Confirm", use_container_width=True, key="kg_confirm"):
                    st.session_state.show_kg_dialog = False
                    st.session_state.kg_model_selected = kg_model
                    st.session_state.kg_graph_type_selected = kg_graph_type
                    st.rerun()
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True, key="kg_cancel"):
                    st.session_state.show_kg_dialog = False
                    st.rerun()

    # Process Knowledge Graph if model was selected
    if 'kg_model_selected' in st.session_state and st.session_state.kg_model_selected and uploaded_files and len(uploaded_files) > 0:
        kg_model = st.session_state.kg_model_selected
        kg_graph_type_ui = st.session_state.get('kg_graph_type_selected', "Full Graph (with Datasets)")

        # Map UI string to parameter value
        graph_type_map = {
            "Full Graph (with Datasets)": "with_datasets",
            "Knowledge Graph Only (without Datasets)": "without_datasets"
        }
        kg_graph_type = graph_type_map[kg_graph_type_ui]

        # Reset
        st.session_state.kg_model_selected = None
        st.session_state.kg_graph_type_selected = None

        st.info(f"Using model: **{kg_model}** for relation extraction")
        st.info(f"Graph type: **{kg_graph_type_ui}**")
        progress_text = st.empty()
        progress_bar = st.progress(0)

        total_files = len(uploaded_files)
        all_keywords = []
        all_datasets = []

        for idx, file in enumerate(uploaded_files):
            progress_text.text(f"🔄 Processing {file.name}... ({idx+1}/{total_files})")
            progress_bar.progress((idx + 1) / total_files)

            try:
                # Reset file pointer and save file
                file.seek(0)
                file_content = file.read()
                temp_filename = f"temp_{idx}_{file.name.replace(' ', '_')}"
                with open(temp_filename, "wb") as f:
                    f.write(file_content)

                # Process each file with optional variable filtering
                nodes, relations, datasets, keywords_metadata = process(temp_filename, k=k, filter_variables=filter_variables, llm_model=kg_model, use_gpt4_datasets=use_gpt4_datasets)

                # Display Keywords section detection info if available
                if keywords_metadata and keywords_metadata.get('from_keywords_section'):
                    st.success(f"📚 **Keywords section found in {file.name}!** Extracted {keywords_metadata['total_found']} keywords directly from the paper's Keywords section.")
                elif keywords_metadata:
                    st.info(f"ℹ️ No Keywords section found in {file.name}. Using {keywords_metadata.get('method', 'algorithmic extraction')} to extract keywords.")

                # Display Variable Filtering Results if applied
                if keywords_metadata and keywords_metadata.get('filtering_applied'):
                    original_kw = keywords_metadata.get('original_keywords', [])
                    filtered_kw = keywords_metadata.get('filtered_keywords', [])
                    removed_kw = keywords_metadata.get('removed_keywords', [])

                    # Create metrics display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Keywords", len(original_kw))
                    with col2:
                        st.metric("Climate Variables Kept", len(filtered_kw),
                                 delta=f"{len(filtered_kw)/len(original_kw)*100:.1f}%" if original_kw else "0%")
                    with col3:
                        st.metric("Non-Variables Removed", len(removed_kw),
                                 delta=f"-{len(removed_kw)/len(original_kw)*100:.1f}%" if original_kw else "0%")

                    # Expandable section to show actual keywords
                    with st.expander(f"🔬 View Variable Filtering Details for {file.name}"):
                        col_kept, col_removed = st.columns(2)

                        with col_kept:
                            st.markdown("**✅ Climate Variables Kept:**")
                            if filtered_kw:
                                # Display as tags
                                kept_display = ", ".join(filtered_kw[:20])
                                if len(filtered_kw) > 20:
                                    kept_display += f" ... and {len(filtered_kw)-20} more"
                                st.write(kept_display)
                            else:
                                st.write("No variables identified")

                        with col_removed:
                            st.markdown("**❌ Non-Variables Removed:**")
                            if removed_kw:
                                removed_display = ", ".join(removed_kw[:10])
                                if len(removed_kw) > 10:
                                    removed_display += f" ... and {len(removed_kw)-10} more"
                                st.write(removed_display)
                            else:
                                st.write("No keywords filtered out")

                # Store in session state with GPT-4 tracking (PLUGGABLE)
                extraction_stats = keywords_metadata.get('extraction_stats', {})
                if file.name not in st.session_state.processed_pdfs:
                    st.session_state.processed_pdfs[file.name] = {
                        'nodes': nodes,
                        'relations': relations,
                        'datasets': datasets,  # List of datasets
                        'keywords_metadata': keywords_metadata,
                        'used_gpt4': use_gpt4_datasets,
                        'extraction_cost': extraction_stats.get('total_cost', 0),
                        'graph_type': kg_graph_type  # Store selected graph type
                    }
                else:
                    # Merge with existing data
                    st.session_state.processed_pdfs[file.name]['nodes'].extend(nodes)
                    st.session_state.processed_pdfs[file.name]['relations'].extend(relations)

                all_keywords.extend(nodes)  # Collect all keywords
                # NEW: Collect all dataset names from the list
                if datasets:
                    for ds in datasets:
                        if ds.get('source') != 'Not specified':
                            all_datasets.append(ds.get('source'))

                # Clean up temp file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
                import traceback
                st.error(f"Details: {traceback.format_exc()}")

        progress_text.empty()
        progress_bar.empty()

        st.success(f"✅ Knowledge graphs generated for {total_files} file(s)!")
        st.info("💡 Tip: Use 'Send to Q&A' button if you want to ask questions about these documents")

        # Display overall filtering summary if filtering was applied
        if filter_variables:
            st.markdown("### 📊 Variable Filtering Summary")

            # Calculate overall statistics
            total_original = 0
            total_kept = 0
            total_removed = 0

            for pdf_name, pdf_data in st.session_state.processed_pdfs.items():
                if pdf_data.get('keywords_metadata', {}).get('filtering_applied'):
                    metadata = pdf_data['keywords_metadata']
                    total_original += len(metadata.get('original_keywords', []))
                    total_kept += len(metadata.get('filtered_keywords', []))
                    total_removed += len(metadata.get('removed_keywords', []))

            if total_original > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Keywords", total_original)
                with col2:
                    st.metric("Climate Variables", total_kept,
                             delta=f"{total_kept/total_original*100:.1f}%")
                with col3:
                    st.metric("Filtered Out", total_removed,
                             delta=f"-{total_removed/total_original*100:.1f}%")
                with col4:
                    retention_rate = total_kept/total_original*100 if total_original > 0 else 0
                    st.metric("Retention Rate", f"{retention_rate:.1f}%")

        # Display combined keywords
        if all_keywords:
            st.markdown("**🔑 Extracted Climate Variables (from all files):**")
            unique_keywords = list(set(all_keywords))
            keyword_html = ""
            for keyword in unique_keywords[:30]:  # Show top 30
                keyword_html += f'<span class="keyword-tag">{keyword}</span>'
            if len(unique_keywords) > 30:
                keyword_html += f'<span class="keyword-tag">... and {len(unique_keywords)-30} more</span>'
            st.markdown(keyword_html, unsafe_allow_html=True)

        # Display datasets found
        if all_datasets:
            st.markdown("**📊 Datasets Found:**")
            for dataset in set(all_datasets):
                st.info(f"📊 {dataset}")

# Q&A Section
st.markdown("---")
st.markdown("## 💬 Q&A")

# Show Q&A system status
if qa_system.list_documents():
    st.success(f"✅ Q&A System Ready - {len(qa_system.list_documents())} document(s) loaded")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset Q&A System", use_container_width=True, key="reset_qa"):
            qa_system.reset_and_reload()
            st.session_state.databases = []
            st.session_state.chat_history = []
            st.rerun()
else:
    st.warning("⚠️ No documents in Q&A system. Please upload PDFs and click 'Send to Q&A'")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📚 Q&A Documents")
    
    if st.session_state.databases:
        for db in st.session_state.databases:
            st.markdown(f'<div class="database-item">📄 {db}</div>', unsafe_allow_html=True)
    else:
        st.info("No documents in Q&A system yet")

with col2:
    # Chat display area
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""<div class="chat-message">
                        <strong>🧑 You:</strong> {message['content']}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="chat-message">
                        <strong>🤖 Assistant:</strong> {message['content']}
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
                <h4 style="color: #495057;">💡 Try asking:</h4>
                <ul style="color: #6c757d; text-align: left; max-width: 500px; margin: 0 auto;">
                    <li>What datasets were used in the research?</li>
                    <li>What are the main findings?</li>
                    <li>What methods were employed?</li>
                    <li>What is the time period of the study?</li>
                    <li>Summarize the key conclusions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about your documents...",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([5, 1])
        with col2:
            submit = st.form_submit_button("➤ Send", use_container_width=True)
        
        if submit and user_input:
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Generate response using Q&A system
            with st.spinner("🤔 Thinking..."):
                try:
                    response = qa_system.answer_question(user_input)
                except Exception as e:
                    response = f"Error: {str(e)}. Please make sure Ollama is running and accessible."
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            st.rerun()

# Knowledge Graph Section
st.markdown("---")
st.markdown("## 🔗 Knowledge Graph")

# Legend
st.markdown("""
<div class="legend-container">
    <div class="legend-item">
        <div class="legend-circle" style="background: #6c757d;"></div>
        <span>Entity</span>
    </div>
    <div class="legend-item">
        <div class="legend-circle" style="background: #28a745;"></div>
        <span>Relationship</span>
    </div>
    <div class="legend-item">
        <div class="legend-circle" style="background: #007bff;"></div>
        <span>Concept</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Graph visualization
if st.session_state.processed_pdfs:
    # Display processing summary
    st.markdown(f"### 📊 Processing Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Files Processed", len(st.session_state.processed_pdfs))
    with col2:
        total_keywords = sum(len(data.get('nodes', [])) for data in st.session_state.processed_pdfs.values())
        st.metric("🔑 Total Keywords", total_keywords)
    with col3:
        total_relations = sum(len(data.get('relations', [])) for data in st.session_state.processed_pdfs.values())
        st.metric("🔗 Total Relations", total_relations)
    with col4:
        files_list = ", ".join(st.session_state.processed_pdfs.keys())
        st.metric("📄 Files", len(st.session_state.processed_pdfs))
    
    # Display dataset information (PLUGGABLE MODULE)
    dataset_filter = display_dataset_filter()
    display_datasets_section(st.session_state.processed_pdfs, dataset_filter)
    display_cost_summary(st.session_state.processed_pdfs)
    
    # Display keywords from all files
    all_keywords = []
    keywords_by_file = {}
    for filename, data in st.session_state.processed_pdfs.items():
        file_keywords = data.get('nodes', [])
        all_keywords.extend(file_keywords)
        keywords_by_file[filename] = file_keywords  # Store all keywords
    
    if all_keywords:
        st.markdown("### 🔑 Extracted Keywords")
        
        # Show keywords by file
        for filename, keywords in keywords_by_file.items():
            st.markdown(f"**From {filename}:**")
            keyword_html = ""
            for keyword in keywords[:10]:  # Show top 10 per file
                keyword_html += f'<span class="keyword-tag">{keyword}</span>'
            st.markdown(keyword_html, unsafe_allow_html=True)
        
        # Show combined unique keywords
        unique_keywords = list(set(all_keywords))
        st.markdown(f"**Total unique keywords: {len(unique_keywords)}**")
    
    try:
        # Generate and display graph
        neo = Neo4jConnector()
        all_nodes = []
        all_relations = []
        all_datasets = []
        
        # Combine data from ALL processed files
        st.write(f"Combining data from {len(st.session_state.processed_pdfs)} files...")
        for filename, data in st.session_state.processed_pdfs.items():
            nodes = data.get('nodes', [])
            relations = data.get('relations', [])
            file_datasets = data.get('datasets', [])

            dataset_count = len(file_datasets) if file_datasets else 0
            st.write(f"- {filename}: {len(nodes)} nodes, {len(relations)} relations, {dataset_count} dataset(s)")

            all_nodes.extend(nodes)
            all_relations.extend(relations)

            # NEW: Collect all datasets from this file (now a list)
            if file_datasets:
                for ds in file_datasets:
                    if ds.get('source') != 'Not specified':
                        all_datasets.append(ds)

        if all_nodes and all_relations:
            st.write(f"Total: {len(all_nodes)} nodes, {len(all_relations)} relations, {len(all_datasets)} dataset(s)")

            # Get graph type from session state (from the first processed file)
            graph_type_to_use = 'with_datasets'  # Default
            if st.session_state.processed_pdfs:
                first_file = list(st.session_state.processed_pdfs.keys())[0]
                graph_type_to_use = st.session_state.processed_pdfs[first_file].get('graph_type', 'with_datasets')

            st.info(f"Graph visualization: **{graph_type_to_use.replace('_', ' ').title()}**")

            # NEW: Pass entire list of datasets instead of just the first one
            neo.store_keywords_and_relations(all_nodes, all_relations, all_datasets)
            rels = neo.retrieve_relations()
            graph, expansion_js = neo.generate_graph(rels, graph_type=graph_type_to_use)
            graph.save_graph("graph.html")
            
            # Add expansion JavaScript to the HTML
            with open("graph.html", "r") as f:
                html_content = f.read()
            html_content = html_content.replace("</body>", expansion_js + "</body>")
            with open("graph.html", "w") as f:
                f.write(html_content)
            
            with open("graph.html", "r") as f:
                html_string = f.read()
            st.components.v1.html(html_string, height=500, scrolling=True)
            
            neo.close()
            
            # Show statistics
            st.markdown("### 📈 Graph Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Unique Nodes", len(set(all_nodes)))
            with col2:
                st.metric("Total Relations", len(all_relations))
            with col3:
                st.metric("Datasets Found", len(all_datasets))
            with col4:
                avg_relations = len(all_relations) // len(st.session_state.processed_pdfs) if st.session_state.processed_pdfs else 0
                st.metric("Avg Relations/File", avg_relations)
                
    except Exception as e:
        st.error(f"Error with Neo4j: {str(e)}")
        st.info("Please check Neo4j credentials")
else:
    # Placeholder graph
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 4rem; border-radius: 15px; text-align: center;">
        <svg width="200" height="200" viewBox="0 0 200 200" style="opacity: 0.7;">
            <circle cx="100" cy="50" r="15" fill="#667eea"/>
            <circle cx="50" cy="100" r="15" fill="#667eea"/>
            <circle cx="150" cy="100" r="15" fill="#667eea"/>
            <circle cx="75" cy="150" r="15" fill="#667eea"/>
            <circle cx="125" cy="150" r="15" fill="#667eea"/>
            <circle cx="100" cy="100" r="20" fill="#764ba2"/>
            <line x1="100" y1="100" x2="100" y2="50" stroke="#dee2e6" stroke-width="2"/>
            <line x1="100" y1="100" x2="50" y2="100" stroke="#dee2e6" stroke-width="2"/>
            <line x1="100" y1="100" x2="150" y2="100" stroke="#dee2e6" stroke-width="2"/>
            <line x1="100" y1="100" x2="75" y2="150" stroke="#dee2e6" stroke-width="2"/>
            <line x1="100" y1="100" x2="125" y2="150" stroke="#dee2e6" stroke-width="2"/>
        </svg>
        <p style="color: #6c757d; margin-top: 1rem; font-weight: 500;">Upload and process PDFs to generate knowledge graph</p>
    </div>
    """, unsafe_allow_html=True)

# Export buttons (functional)
if st.session_state.processed_pdfs:
    st.markdown("### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        all_relations = []
        for data in st.session_state.processed_pdfs.values():
            all_relations.extend(data.get('relations', []))
        if all_relations:
            json_data = json.dumps(all_relations, indent=2)
            st.download_button(
                label="📄 Export Relations (JSON)",
                data=json_data,
                file_name="knowledge_graph.json",
                mime="application/json",
                use_container_width=True
            )
    with col2:
        if all_relations:
            df = pd.DataFrame(all_relations)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Export Relations (CSV)",
                data=csv,
                file_name="knowledge_graph.csv",
                mime="text/csv",
                use_container_width=True
            )
    with col3:
        # Export datasets (PLUGGABLE MODULE)
        datasets_csv = export_datasets_to_csv(st.session_state.processed_pdfs)
        if datasets_csv:
            st.download_button(
                label="📚 Export Datasets (CSV)",
                data=datasets_csv,
                file_name="extracted_datasets.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-links">
        <a href="#">About</a>
        <a href="#">Documentation</a>
        <a href="#">Contact</a>
        <a href="#">Privacy Policy</a>
    </div>
    <div>
        <small>Built with AI-powered document intelligence • © 2024 PDF Knowledge Explorer</small>
    </div>
</div>
""", unsafe_allow_html=True)