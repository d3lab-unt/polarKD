# PDF to Knowledge Graph Generator
A local, private, and intelligent system to extract **keywords** and **semantic relationships** from uploaded research PDFs and generate an interactive **Knowledge Graph**.  
Built with Streamlit, Neo4j, Ollama (LLaMA 3), and PyVis.

##  Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [GitHub Setup Instructions](#-github-setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Install and Run Ollama](#4-install-and-run-ollama)
  - [5. Run the Streamlit App](#5-run-the-streamlit-app)
- [Project Folder Structure](#-project-folder-structure)
- [How It Works](#-how-it-works)
- [Example](#-example)
- [Future Enhancements](#-future-enhancements)
- [Example flow](#-Example-Flow)
- [Author](#-author)
- [Final Notes](#-final-notes)

---

##  Overview

This project allows users to upload one or more research papers (PDFs), automatically extract the key concepts, and generate an interactive Knowledge Graph.  
It uses advanced keyword extraction (TF-IDF, YAKE, KeyBERT) and LLM-based relation extraction (using Ollama models) to build meaningful graphs for scientific documents.

---

##  Features

- Upload single or multiple PDFs.
- Intelligent keyword extraction (from *Keywords section* if present, else model-based).
- Relation extraction between keywords using LLM (LLaMA3).
- **Q&A System with RAG**: Ask questions about uploaded documents using Retrieval-Augmented Generation.
- **Dataset Information Extraction**: Automatically identifies and extracts dataset sources, variables, time periods, and locations from research papers.
- **Enhanced Graph Visualization**: Color-coded nodes distinguish between datasets, variables, and regular keywords.
- Visualize the generated Knowledge Graph interactively.
- Download the extracted relations as CSV or JSON.
- 100% Local Processing (Privacy-Preserving).

---

##  Tech Stack

- **Python 3.10+**
- **Streamlit** (Frontend App)
- **pdfplumber** (PDF Text Extraction)
- **NLTK**, **spaCy** (Text Cleaning)
- **TF-IDF**, **YAKE**, **KeyBERT** (Keyword Extraction)
- **Ollama** (LLaMA 3 model for Relation Extraction and Q&A)
- **Neo4j** (Graph Storage with Enhanced Dataset Support)
- **PyVis** (Graph Visualization)
- **SentenceTransformers** (Semantic Scoring and Document Embeddings for RAG)

---

## GitHub Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/pdf-knowledge-graph.git
cd pdf-knowledge-graph
``` 

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Make sure you have neo4j aura instance running
Give the required neo4j ceredentials:
U can find these credentials in the python driver of neo4j see the references.
NEO4j_USER is 'neo4j'
Make sure you have the password while creating account.
```bash
NEO4J_URI='neo4j+s://.......'
NEO4J_USER='neo4j'
NEO4J_PASSWORD='.....'
```

### 5. Install and Run Ollama
Install Ollama and pull the LLaMA3 model:
```bash
ollama run llama3
```

### 6. Run the Streamlit App
```bash
streamlit run frontend.py
```

---

## Project Folder Structure
```bash
pdf-knowledge-graph/
├── frontend_light.py        # Streamlit frontend app with Q&A integration
├── keywords_extraction.py   # Extract keywords, relations, and dataset info
├── neo4j_storage.py         # Enhanced Neo4j with dataset nodes and relationships
├── qa_module.py            # RAG-based Q&A system for document queries
├── storing.py              # Optional CLI mode to process PDFs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## How It Works

### 1. PDF Upload
Upload one or more PDF documents via Streamlit.

### 2. Text Extraction
Extracts text using `pdfplumber` from uploaded PDFs.

### 3. Processing Options

#### Option A: Knowledge Graph Generation
- **Keyword Extraction**: 
  - If a "Keywords" section exists, directly extract those keywords
  - Otherwise, fallback to automatic extraction using TF-IDF, YAKE, and KeyBERT
  - For multiple PDFs, automatically extract roughly `k/n` keywords per PDF (where n = number of PDFs)
- **Dataset Information Extraction**:
  - Automatically identifies dataset sources, variables, time periods, and locations using LLaMA3
  - Creates specialized Dataset nodes with properties (name, time_period, location)
  - Marks dataset variables as dual-labeled nodes (:Keyword:Variable)
  - Establishes relationships: HAS_VARIABLE (dataset→variable) and EXTRACTED_FROM (dataset→keyword)
- **Relation Extraction**: Keyword pairs are passed to LLaMA3 to infer semantic relations
- **Enhanced Graph Construction**: 
  - Creates color-coded nodes: Orange for datasets, Green for variables, Blue for keywords
  - Stores nodes and relationships in Neo4j with dataset context

#### Option B: Q&A System
- **Document Processing**:
  - Splits text into overlapping 800-word chunks for better context retrieval
  - Generates vector embeddings using SentenceTransformer (all-MiniLM-L6-v2)
- **RAG Implementation**:
  - Stores document chunks with their embeddings
  - Uses cosine similarity for semantic search (threshold: 0.15)
  - Retrieves top-k relevant chunks for query answering
- **Answer Generation**:
  - Combines relevant chunks as context
  - Uses LLaMA3 to generate contextual answers
  - Includes source citations from retrieved documents

### 4. Visualization & Interaction
- **Knowledge Graph**: Interactive PyVis network with color-coded nodes and relationships
- **Q&A Interface**: Chat-based interface with conversation history and context-aware responses

### 5. Export Options
Users can download the extracted relations as CSV or JSON files.

---

## Example Workflow

 Upload PDFs → Extract Keywords → Find Relations → Visualize Knowledge Graph

---

## Key Components Explained

### qa_module.py - Q&A System
- **QASystem Class**: Implements RAG-based document Q&A
  - `add_document()`: Processes and stores PDF documents with embeddings
  - `find_relevant_chunks()`: Semantic search using cosine similarity
  - `generate_answer()`: Context-aware answer generation with LLaMA3
  - `answer_question()`: Main interface for Q&A processing

### keywords_extraction.py - Enhanced Extraction
- **extract_dataset_info()**: New function at line 410 that extracts:
  - Dataset sources and names from research papers
  - Measured variables and parameters
  - Time periods and geographic locations
  - Returns structured JSON with dataset metadata
- **process()**: Updated to return dataset information alongside keywords and relations

### neo4j_storage.py - Graph Storage Enhancements  
- **Dataset Node Creation**: Lines 26-43 handle dataset nodes with properties
- **Variable Marking**: Lines 69-70 mark keywords as variables when applicable
- **New Relationships**:
  - `HAS_VARIABLE`: Links datasets to their measured variables (lines 72-79)
  - `EXTRACTED_FROM`: Links datasets to extracted keywords (lines 81-89)
- **Enhanced Visualization**: Lines 127-140 implement color-coded node rendering

### frontend_light.py - Integrated Interface
- **Dual Processing Modes** (lines 428-526):
  - "Send to Q&A": Loads documents into RAG system only
  - "Generate Knowledge Graph": Creates visualization with dataset extraction
- **Q&A Chat Interface** (lines 528-612): Full conversational UI with history
- **Session Management**: Tracks processed PDFs and Q&A documents separately

## Future Enhancements

- OCR support for scanned PDFs (using Tesseract)
- Interactive graph editing (merging/splitting nodes)
- Fine-tune LLaMA3 model for specific domains
- Automatic summarization of graphs
- Cross-document relationship discovery in Q&A
- Export Q&A conversations and insights
- Full-fledged web application with user login support

---
## Example Flow
- Connect to the server (in my case UNT Server) using ssh command.
![image](https://github.com/user-attachments/assets/d35fc7da-dcf3-4ddd-b204-28b428b54fe4)
![image](https://github.com/user-attachments/assets/a6365d23-815a-49a3-86d9-30817a73faf9)
- create venv and install required libraries and go the directory where all .py files are stored.
- Then run ollama serve to pull llama 3.2 from ollama.
  ![image](https://github.com/user-attachments/assets/c785e024-c6dc-41ff-9651-20148d669754)
  ![image](https://github.com/user-attachments/assets/ee44f17d-7834-4a61-be03-6fbb07d297cd)
- Now open another terminal and go to the same directory where code is present and run frontend.py
  ![image](https://github.com/user-attachments/assets/48ac18d4-da82-4fab-afcd-eb1f4871ab09)
- Go to new terminal window and then run the following command to connect server to localhost using ssh tunneling.
  ![image](https://github.com/user-attachments/assets/31563f82-e794-4126-bbc6-c07670d2101c)
- Open the localhost link and upload the files.
  ![image](https://github.com/user-attachments/assets/581e4d2e-fe8f-4737-aeeb-c6f1803b0bbb)
- The below are the results:
  ![image](https://github.com/user-attachments/assets/3159baa9-1823-43dc-a115-987e6fb53564)
  ![image](https://github.com/user-attachments/assets/d24482ac-b8a2-43ca-b6dc-1cc1dc69394f)
  ![image](https://github.com/user-attachments/assets/59527526-a805-4d13-8fda-2bff2307cc7f)
  ![image](https://github.com/user-attachments/assets/ea05dcd9-00a3-4ec5-8474-9cfa5aac2960)
- Neo4j credentials driver info:
  ![image](https://github.com/user-attachments/assets/528c40ec-cf20-49b4-924e-ae5cfb3003de)
  ![image](https://github.com/user-attachments/assets/641c7ec7-812c-4ee6-ade9-2532905c4a02)




##  Author

**Ajith Kumar Dugyala**  
Email: ajithdugyala@gmail.com  
Location: Denton, Texas, USA

---

## Final Notes

This project is fully local and privacy-focused.  
It can be extended for domain-specific research, scientific knowledge graph generation, or intelligent document summarization.

---

