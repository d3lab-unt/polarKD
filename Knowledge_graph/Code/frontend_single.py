import streamlit as st
from keywords_extraction import process
from neo4j_storage import Neo4jConnector
import os

st.set_page_config(page_title="Relationships Graph", layout="wide")
st.title("PDF Knowledge Graph Generator")

# File uploader
uploaded_file = st.file_uploader("Upload a research PDF", type=["pdf"])

# Input for number of keywords
k = st.number_input("Number of Keywords to Extract", min_value=5, max_value=100, value=15, step=1)

if uploaded_file is not None:
    # Save the uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Extract keywords and relations
    with st.spinner("Extracting keywords and relations..."):
        nodes, relations = process("temp.pdf", k=k)

    # Show warning if fewer keywords were extracted
    if len(nodes) < k:
        st.warning(f"⚠️ Only {len(nodes)} unique keywords could be extracted, though you requested {k}. "
                   "The text may be too short or keywords overlapped heavily.")

    # Display extracted content
    st.subheader("Extracted Keywords")
    st.write(nodes)

    st.subheader("Extracted Relations")
    for r in relations:
        st.write(f"({r['source']}) -[{r['relation']}]-> ({r['target']})")

    # Store in Neo4j
    neo = Neo4jConnector()
    with st.spinner("Storing data in Neo4j and visualizing graph..."):
        neo.store_keywords_and_relations(nodes, relations)
        rels = neo.retrieve_relations()
        graph = neo.generate_graph(rels)
        graph.save_graph("graph.html")
        with open("graph.html", "r") as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=600, scrolling=True)

        # Export to CSV and JSON
        csv_df = neo.export_csv("extracted_relations.csv")
        json_data = neo.export_json("extracted_relations.json")

    # File download options
    st.subheader("Download Extracted Data")
    with open("extracted_relations.csv", "rb") as f:
        st.download_button("Download CSV", f, "extracted_relations.csv", "text/csv")
    with open("extracted_relations.json", "rb") as f:
        st.download_button("Download JSON", f, "extracted_relations.json", "application/json")

    neo.close()
    os.remove("temp.pdf")