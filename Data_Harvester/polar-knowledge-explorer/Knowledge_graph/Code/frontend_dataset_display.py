"""
Modular dataset display component for frontend (PLUGGABLE).

This module can be easily enabled/disabled or replaced.
"""
import streamlit as st


def display_gpt4_toggle():
    """
    Display GPT-4 toggle control (PLUGGABLE: easy to enable/disable).

    Returns:
        bool: True if GPT-4 should be used, False otherwise
    """
    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            use_gpt4 = st.checkbox(
                "🤖 Use GPT-4 for dataset extraction",
                value=False,
                help="More accurate (95%+ precision, no hallucinations) but costs ~$0.02/paper. "
                     "Uncheck to use free local LLM (lower accuracy)."
            )

        with col2:
            if use_gpt4:
                st.info("💰 GPT-4 Active")
            else:
                st.success("🆓 Free Mode")

        if use_gpt4:
            st.warning("⚠️ GPT-4 will make API calls to OpenAI. Cost: ~$0.015-0.025 per paper.")

        return use_gpt4


def display_dataset_filter():
    """
    Display dataset filter dropdown (PLUGGABLE).

    Returns:
        str: Filter selection ('all', 'primary', or 'cited')
    """
    filter_options = {
        "All datasets": "all",
        "🟢 PRIMARY only (author-created)": "primary",
        "🔵 CITED only (referenced)": "cited"
    }

    selected = st.selectbox(
        "Filter datasets:",
        list(filter_options.keys()),
        help="PRIMARY = datasets the authors created/collected. "
             "CITED = datasets they referenced from others."
    )

    return filter_options[selected]


def filter_datasets(datasets, filter_type):
    """
    Filter datasets by type (PLUGGABLE: easy to add more filter types).

    Args:
        datasets: List of dataset dicts
        filter_type: 'all', 'primary', or 'cited'

    Returns:
        Filtered list of datasets
    """
    if filter_type == 'primary':
        return [d for d in datasets if d.get('dataset_type') == 'primary']
    elif filter_type == 'cited':
        return [d for d in datasets if d.get('dataset_type') == 'cited']
    else:  # 'all'
        return datasets


def get_dataset_badge(dataset_type):
    """
    Get colored badge for dataset type (PLUGGABLE: easy to customize).

    Args:
        dataset_type: 'primary' or 'cited'

    Returns:
        HTML string for badge
    """
    if dataset_type == 'primary':
        return '<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold;">🟢 PRIMARY</span>'
    elif dataset_type == 'cited':
        return '<span style="background-color: #17a2b8; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold;">🔵 CITED</span>'
    else:
        return '<span style="background-color: #6c757d; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold;">❓ UNKNOWN</span>'


def display_dataset_card(dataset_info, idx):
    """
    Display a single dataset as a card (PLUGGABLE: easy to customize layout).

    Args:
        dataset_info: Dict with dataset information
        idx: Dataset index number
    """
    dataset_type = dataset_info.get('dataset_type', 'cited')
    badge_html = get_dataset_badge(dataset_type)

    # Header with badge
    st.markdown(f"**{idx}. {dataset_info.get('source', 'Unknown')}** {badge_html}", unsafe_allow_html=True)

    # Create two-column layout for metadata
    col1, col2 = st.columns(2)

    with col1:
        variables = dataset_info.get('variables', [])
        if variables:
            var_display = ', '.join(variables[:3])
            if len(variables) > 3:
                var_display += f" ... (+{len(variables)-3} more)"
            st.write(f"📈 **Variables:** {var_display}")
        else:
            st.write(f"📈 **Variables:** None specified")

        time_period = dataset_info.get('time_period', 'Not specified')
        st.write(f"📅 **Time Period:** {time_period}")

    with col2:
        location = dataset_info.get('location', 'Not specified')
        st.write(f"🌍 **Location:** {location}")

        confidence = dataset_info.get('confidence', 0.0)
        if confidence > 0:
            # Display as percentage with star rating
            stars = "⭐" * int(confidence * 5)
            st.write(f"⭐ **Confidence:** {confidence:.0%} {stars}")
        else:
            st.write(f"⭐ **Confidence:** Not specified")

    # Usage description (if available)
    usage_description = dataset_info.get('usage_description', '')
    if usage_description and usage_description != '':
        with st.expander("📝 How was this dataset used?"):
            st.write(usage_description)

    # Citation info (if available)
    citation_info = dataset_info.get('citation_info')
    if citation_info:
        with st.expander("📚 Citation"):
            st.write(citation_info)

    # Context (if available)
    context = dataset_info.get('context', '')
    if context and context != '':
        with st.expander("💬 Context from paper"):
            st.write(f'"{context}"')

    st.markdown("---")


def display_datasets_section(processed_pdfs, filter_type='all'):
    """
    Display all datasets with filtering (PLUGGABLE: main display component).

    Args:
        processed_pdfs: Dict mapping filename -> {datasets, nodes, relations, ...}
        filter_type: 'all', 'primary', or 'cited'
    """
    st.markdown("### 📚 Datasets Extracted")

    if not processed_pdfs:
        st.info("No papers processed yet. Upload PDFs above to extract datasets.")
        return

    # Collect all datasets across papers
    total_datasets = 0
    total_primary = 0
    total_cited = 0

    for filename, data in processed_pdfs.items():
        file_datasets = data.get('datasets', [])
        total_datasets += len(file_datasets)
        total_primary += sum(1 for d in file_datasets if d.get('dataset_type') == 'primary')
        total_cited += sum(1 for d in file_datasets if d.get('dataset_type') == 'cited')

    # Display summary
    st.markdown(f"**Total across all papers:** {total_datasets} datasets "
                f"(🟢 {total_primary} PRIMARY, 🔵 {total_cited} CITED)")

    # Display datasets per paper
    for filename, data in processed_pdfs.items():
        file_datasets = data.get('datasets', [])

        if not file_datasets:
            continue

        # Apply filter
        filtered_datasets = filter_datasets(file_datasets, filter_type)

        if not filtered_datasets:
            continue

        # Count by type for this paper
        primary_count = sum(1 for d in filtered_datasets if d.get('dataset_type') == 'primary')
        cited_count = len(filtered_datasets) - primary_count

        # Check if GPT-4 was used
        used_gpt4 = data.get('used_gpt4', False)
        gpt4_badge = "🤖 GPT-4" if used_gpt4 else "🆓 Local"

        # Expandable section per paper
        with st.expander(
            f"📄 **{filename}** - {len(filtered_datasets)} dataset(s) "
            f"(🟢 {primary_count} PRIMARY, 🔵 {cited_count} CITED) [{gpt4_badge}]",
            expanded=True
        ):
            # Display extraction cost if GPT-4 was used
            if used_gpt4:
                extraction_cost = data.get('extraction_cost', 0)
                if extraction_cost > 0:
                    st.caption(f"💰 Extraction cost: ${extraction_cost:.4f}")

            # Display each dataset as a card
            for idx, dataset_info in enumerate(filtered_datasets, 1):
                display_dataset_card(dataset_info, idx)


def display_cost_summary(processed_pdfs):
    """
    Display total GPT-4 cost in sidebar (PLUGGABLE: easy to show/hide).

    Args:
        processed_pdfs: Dict mapping filename -> {datasets, nodes, relations, ...}
    """
    # Check if any papers used GPT-4
    gpt4_papers = [
        (name, data.get('extraction_cost', 0))
        for name, data in processed_pdfs.items()
        if data.get('used_gpt4', False)
    ]

    if not gpt4_papers:
        return  # Don't show cost summary if no GPT-4 usage

    # Calculate total cost
    total_cost = sum(cost for _, cost in gpt4_papers)

    # Display in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💰 GPT-4 Usage")
        st.metric("Papers processed with GPT-4", len(gpt4_papers))
        st.metric("Total cost", f"${total_cost:.4f}")

        # Show breakdown
        with st.expander("Cost breakdown"):
            for name, cost in gpt4_papers:
                st.write(f"• {name}: ${cost:.4f}")

        st.caption("ℹ️ Costs are approximate. Actual OpenAI charges may vary slightly.")


def export_datasets_to_csv(processed_pdfs):
    """
    Export all datasets to CSV (PLUGGABLE: easy to add more export formats).

    Args:
        processed_pdfs: Dict mapping filename -> {datasets, nodes, relations, ...}

    Returns:
        CSV string ready for download
    """
    import pandas as pd

    # Collect all datasets
    all_datasets = []
    for filename, data in processed_pdfs.items():
        for ds in data.get('datasets', []):
            dataset_row = {
                'paper': filename,
                'source': ds.get('source', ''),
                'dataset_type': ds.get('dataset_type', ''),
                'variables': ', '.join(ds.get('variables', [])),
                'time_period': ds.get('time_period', ''),
                'location': ds.get('location', ''),
                'confidence': ds.get('confidence', 0),
                'usage_description': ds.get('usage_description', ''),
                'citation_info': ds.get('citation_info', ''),
                'context': ds.get('context', '')
            }
            all_datasets.append(dataset_row)

    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(all_datasets)
    return df.to_csv(index=False)
