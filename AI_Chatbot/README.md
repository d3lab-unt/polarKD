
# Domain Specific AI Chatbot

A local, private, and interactive document-based question answering system using **LangChain**, **FAISS**, and **Ollama** to power **LLaMA 2** models.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Folder Structure](#folder-structure)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

This script allows users to ask questions about the contents of uploaded documents using a **local LLM** served by **Ollama**. It leverages **LangChain** for document loading, chunking, and retrieval, and **FAISS** for fast similarity search on embeddings.

---

## Features

-  Load and chunk local documents
-  Embed text and index using FAISS
-  Retrieve top matches for a user query
-  Use local LLaMA model to generate responses
-  100% local and private

---

##  Tech Stack

- Python 3.10+
- LangChain
- Ollama (LLaMA 2 or compatible model)
- FAISS
- SentenceTransformers (for embedding)
- PyPDF / Unstructured (optional for document parsing)

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/llm-ollama-qa.git
cd llm-ollama-qa
```

### 2. Create a Virtual Environment 
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Run Ollama

Install [Ollama](https://ollama.com/) and run your desired model:
```bash
ollama run llama2
```

### 5. Run the Script
```bash
python qa_main.py
```

---

##  How It Works

###  Document Loader
Loads PDFs or text files and chunks them using LangChain utilities.

###  Embedding & Indexing
Uses SentenceTransformer (e.g., `all-MiniLM-L6-v2`) to convert text chunks into vector space, stored in FAISS.

###  Query Input
User enters a natural language question.

###  Retrieval + LLM
The script retrieves top-k matching document chunks and sends them to the local LLaMA model via Ollama for answer generation.

---

##  Folder Structure

```
llm-ollama-qa/
├── qa_main.py                 # Main script
├── requirements.txt
├── README.md
└── docs/                      # Folder for documents to load
```

---

##  Future Improvements

- Add a web interface (Streamlit or Gradio)
- Support more document formats
- Use async Ollama server for better performance
- Add multi-document session support

---

##  Author

Harini Varanasi 
harinivaranasi.data@gmail.com   