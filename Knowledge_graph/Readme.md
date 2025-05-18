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
- **Ollama** (LLaMA 3 model for Relation Extraction)
- **Neo4j** (Graph Storage)
- **PyVis** (Graph Visualization)
- **SentenceTransformers** (Semantic Scoring)

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

### 4. Install and Run Ollama
Install Ollama and pull the LLaMA3 model:
```bash
ollama run llama3
```

### 5. Run the Streamlit App
```bash
streamlit run frontend.py
```

---

## Project Folder Structure
```bash
pdf-knowledge-graph/
├── frontend.py              # Streamlit frontend app
├── keywords_extraction.py   # Extract keywords and relations
├── neo4j_storage.py         # Neo4j database interaction
├── storing.py               # Optional CLI mode to process PDFs
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## How It Works

### 1. PDF Upload
Upload one or more PDF documents via Streamlit.

###  Text Extraction
Extracts text using `pdfplumber` from uploaded PDFs.

### Keyword Extraction
- If a "Keywords" section exists, directly extract those keywords.
- Otherwise, fallback to automatic extraction using TF-IDF, YAKE, and KeyBERT.
- For multiple PDFs, automatically extract roughly `k/n` keywords per PDF (where n = number of PDFs).

###  Relation Extraction
Keyword pairs are passed to a LLaMA3 model running locally via Ollama to infer semantic relations.

### Knowledge Graph Construction
Nodes (keywords) and edges (relations) are inserted into a Neo4j database.

### Visualization
A PyVis network graph is generated and displayed inside the Streamlit app.

### Export Options
Users can download the extracted relations as CSV or JSON files.

---

## Example Workflow

 Upload PDFs → Extract Keywords → Find Relations → Visualize Knowledge Graph

---

## Future Enhancements

- OCR support for scanned PDFs (using Tesseract)
- Interactive graph editing (merging/splitting nodes)
- Fine-tune LLaMA3 model for specific domains
- Automatic summarization of graphs
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


##  Author

**Ajith Kumar Dugyala**  
Email: ajithdugyala@gmail.com  
Location: Denton, Texas, USA

---

## Final Notes

This project is fully local and privacy-focused.  
It can be extended for domain-specific research, scientific knowledge graph generation, or intelligent document summarization.

---

