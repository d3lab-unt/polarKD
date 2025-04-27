# PDF to Knowledge Graph Generator
A local, private, and intelligent system to extract **keywords** and **semantic relationships** from uploaded research PDFs and generate an interactive **Knowledge Graph**.  
Built with Streamlit, Neo4j, Ollama (LLaMA 3), and PyVis.

## 📑 Table of Contents

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
- [Author](#-author)
- [Final Notes](#-final-notes)

---

## 🚀 Overview

This project allows users to upload one or more research papers (PDFs), automatically extract the key concepts, and generate an interactive Knowledge Graph.  
It uses advanced keyword extraction (TF-IDF, YAKE, KeyBERT) and LLM-based relation extraction (using Ollama models) to build meaningful graphs for scientific documents.

---

## ✨ Features

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
