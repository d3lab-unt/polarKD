# PolarKD (Polar Knowledge Discovery) - Complete System Analysis

**Generated:** 2025-10-23
**Purpose:** Comprehensive codebase documentation for future reference and onboarding

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Module-by-Module Analysis](#3-module-by-module-analysis)
4. [System Architecture & Data Flow](#4-system-architecture--data-flow)
5. [Technology Stack](#5-technology-stack)
6. [Database Schema (Neo4j)](#6-database-schema-neo4j)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Configuration & Credentials](#8-configuration--credentials)
9. [Performance Benchmarks](#9-performance-benchmarks)
10. [Entry Points & Usage](#10-entry-points--usage)
11. [Known Issues & Limitations](#11-known-issues--limitations)
12. [Quick Reference](#12-quick-reference)

---

## 1. Project Overview

**Project Name:** PolarKD (Polar Knowledge Discovery) Toolkit
**Purpose:** Local, privacy-preserving system to extract keywords, semantic relationships, and answer questions from research PDFs (focused on polar/climate science)

**Core Capabilities:**
- **Knowledge Graph Generation**: Extract keywords and relationships from PDFs
- **Q&A System**: RAG-based question answering using local LLMs
- **Dataset Extraction**: Automatically identify datasets, variables, time periods, locations
- **LLM Comparison**: Benchmark multiple models (Llama3, GPT-2, Qwen, Mistral)
- **Interactive UI**: Streamlit-based web interface

**Key Features:**
- 100% local processing (privacy-focused)
- Multi-PDF support
- Hybrid keyword extraction (TF-IDF + YAKE + KeyBERT)
- LLM-based relation extraction (using Ollama)
- Neo4j graph storage with PyVis visualization
- RAGAS evaluation metrics

---

## 2. Folder Structure

```
Knowledge_graph/
├── Code/
│   ├── frontend_light.py              # Main Streamlit UI (847 lines)
│   ├── keywords_extraction.py         # Keyword & relation extraction (651 lines)
│   ├── neo4j_storage.py               # Neo4j database connector (195 lines)
│   ├── qa_module.py                   # Q&A RAG system (216 lines)
│   ├── storing.py                     # CLI processing script (48 lines)
│   ├── evaluation_metrics.py          # RAGAS metrics (349 lines)
│   ├── compare_llm_models.py          # LLM comparison script (398 lines)
│   ├── test_qa_mistral.py             # Q&A test harness (235 lines)
│   ├── qa_test_results_mistral.txt    # Test results output
│   ├── test_output.log                # Test execution log
│   └── evaluation/
│       └── comparisons/
│           ├── README.md              # Evaluation documentation
│           └── compare_llm_models.py  # Alternative comparison script (412 lines)
├── docs/
│   ├── arctic1.pdf                    # Sample research papers
│   ├── arctic2.pdf
│   └── climate3.pdf
├── Readme.md                          # Project documentation
├── requirements.txt                   # Python dependencies
└── iharp-logo.jpg                     # UI logo

Root-level files:
├── CODE_WALKTHROUGH.md                # Detailed code walkthrough
├── QUICK_MODULE_GUIDE.md              # 5-point module summaries
└── PROFESSOR_DEMO_GUIDE.md            # Demo instructions
```

**Total Python Files:** 8 core modules + 2 evaluation scripts
**Total Lines of Code:** ~2,900+ lines

---

## 3. Module-by-Module Analysis

### Module 1: `frontend_light.py` (847 lines)

**Purpose:** Main Streamlit web interface for PolarKD

**Key Components:**

1. **Page Configuration** (Lines 12-16)
   - Layout: Wide
   - Theme: Light (custom purple gradient)
   - Title: "Polar Knowledge Discovery (PolarKD) Toolkit"

2. **Custom CSS Styling** (Lines 19-375)
   - Purple gradient theme (#667eea to #764ba2)
   - Light background (#ffffff)
   - Custom file uploader (drag-drop area)
   - Button styling (consistent purple gradient)
   - Chat message bubbles
   - Legend containers
   - Graph visualization styling

3. **Session State Management** (Lines 393-403)
   ```python
   st.session_state.uploaded_files = []       # PDFs uploaded
   st.session_state.databases = []            # Q&A documents
   st.session_state.chat_history = []         # Chat messages
   st.session_state.processed_pdfs = {}       # KG data per PDF
   st.session_state.current_graph = None      # Graph visualization
   ```

4. **Upload Section** (Lines 430-563)
   - Multi-file PDF uploader
   - Two action buttons:
     - **"Send to Q&A"**: Adds PDFs to Q&A system only (no KG)
     - **"Generate Knowledge Graph"**: Extracts keywords + relations
   - Keyword slider (5-50 keywords, default 15)
   - Progress bar for multi-file processing

5. **Q&A Section** (Lines 565-653)
   - Document list display
   - Chat history with user/assistant bubbles
   - Text input form with "Send" button
   - Calls `qa_system.answer_question(user_input)`
   - "Clear Chat" and "Reset Q&A System" buttons

6. **Knowledge Graph Section** (Lines 655-803)
   - Processing summary (files, keywords, relations, datasets)
   - Dataset information expanders (source, variables, period, location)
   - Keywords by file display (top 10 per file)
   - Neo4j graph generation + PyVis visualization
   - Graph legend (Entity, Relationship, Concept)
   - Graph statistics (unique nodes, total relations, datasets found)

7. **Export Options** (Lines 805-832)
   - JSON download (all relations)
   - CSV download (relations as DataFrame)

**Important Functions:**

| Function | Lines | Description |
|----------|-------|-------------|
| File Upload Handler | 438-446 | Handles multi-file PDF upload |
| Send to Q&A | 465-491 | Adds documents to Q&A system |
| Generate KG | 493-563 | Processes PDFs for knowledge graph |
| Chat Form | 626-653 | Q&A input and response handling |
| Graph Display | 736-766 | Neo4j + PyVis graph visualization |

**Dependencies:**
- `streamlit` - Web UI framework
- `keywords_extraction.process()` - KG extraction
- `neo4j_storage.Neo4jConnector` - Graph storage
- `qa_module.qa_system` - Q&A singleton

---

### Module 2: `keywords_extraction.py` (651 lines)

**Purpose:** Extract keywords, relations, and dataset info from PDFs

**Key Components:**

1. **Text Extraction** (Lines 25-90)
   ```python
   def text_extraction(pdf_path):
       # Uses pdfplumber to extract text
       # Cleans: URLs, emails, references, citations
       # Removes stopwords + lemmatization
   ```

2. **Keyword Extraction** (Lines 92-226)
   - **Strategy 1:** Check if paper has "Keywords" section (lines 92-120)
   - **Strategy 2:** Ensemble extraction if no keywords section (lines 122-226)
     - TF-IDF (25% weight)
     - YAKE (50% weight)
     - KeyBERT (25% weight)
   - Filters generic terms ("study", "paper", "research", etc.)
   - Returns top-k unique keywords

3. **Relation Extraction** (Lines 228-483)

   **Process:**
   - Generate all keyword pairs (lines 228-234)
   - Split text into chunks (2000 chars, 300 overlap) (lines 237-249)
   - Use LLaMA 3 to find relations per chunk (lines 264-369)
   - Validate relations (only original keywords allowed) (lines 371-411)
   - Score relations using cosine similarity (lines 413-444)
   - Keep top 100 relations (line 448)

   **LLM Prompt Template** (Lines 281-312):
   ```
   You are an expert in extracting relationships between keywords from research papers.

   For each pair, return ONLY:
   - (KEYWORD_1, RELATION, KEYWORD_2)

   STRICT RULES:
   - Relation types: causes, increases, decreases, affects, relates_to, etc.
   - NO explanations, NO extra text
   - If no relation: return "None"
   ```

4. **Dataset Extraction** (Lines 484-577)

   **Function:** `extract_dataset_info(text)`

   **Extraction:**
   - Takes first 12,000 characters (≈4-5 pages)
   - Uses Mistral LLM with structured prompt
   - Extracts:
     - Data source/dataset name
     - Variables/parameters measured
     - Time period covered
     - Geographic location
   - Returns JSON: `{source, variables[], time_period, location}`
   - Defaults: "Not specified" if not found

   **Note:** Only checks first 12K chars, NOT all chunks (unlike relation extraction)

5. **Main Process Function** (Lines 581-650)
   ```python
   def process(file_path, k):
       input_text = text_extraction(file_path)
       keywords = extract_keywords(input_text, k)

       # Extract dataset info
       dataset_info = extract_dataset_info(input_text)

       # Extract relations through ALL chunks
       chunks = text_chunks(input_text)
       for chunk in chunks:
           extracted = extract_relations_llama_all(chunk, keyword_pairs)

       return neo4j_nodes, neo4j_edges, dataset_info
   ```

**Key Variables:**

| Variable | Type | Description |
|----------|------|-------------|
| `neo4j_nodes` | List[str] | Unique keywords for graph nodes |
| `neo4j_edges` | List[Dict] | Relations: `{source, relation, target, score}` |
| `dataset_info` | Dict | `{source, variables, time_period, location}` |

**LLM Models Used:**
- **Llama3** - Relation extraction (lines 264-369)
- **Mistral** - Dataset extraction (lines 526-550)

---

### Module 3: `neo4j_storage.py` (195 lines)

**Purpose:** Neo4j database connector for graph storage and visualization

**Credentials** (Lines 7-10):
```python
NEO4J_URI = 'neo4j+s://0d4ad98d.databases.neo4j.io'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'l2eTsa3JmSPkwWoCCNszhUyvkxkapl3WwN2oHzJZJ6E'
```

**Key Functions:**

1. **store_keywords_and_relations** (Lines 26-64)
   - Deletes existing graph: `MATCH (n) DETACH DELETE n`
   - Creates dataset node if dataset_info provided (lines 30-34)
   - Creates keyword nodes (lines 43-46)
   - Links keywords to dataset (lines 48-50)
   - Creates relationships with cleaned relation names (lines 52-63)

2. **Dataset Node Creation** (Lines 67-78)
   ```cypher
   MERGE (d:Dataset {
       name: $name,
       time_period: $time_period,
       location: $location
   })
   ```
   **Missing:** No `link` or `url` property stored

3. **Variable Marking** (Lines 80-92)
   - Marks keywords as `:Variable` nodes (line 82)
   - Links dataset to variables: `(d)-[:HAS_VARIABLE]->(v)` (line 90)

4. **Keyword-Dataset Linking** (Lines 94-101)
   ```cypher
   (d)-[:EXTRACTED_FROM]->(k)
   ```

5. **generate_graph** (Lines 128-181)

   **Node Styling:**
   - Dataset nodes: Blue square (#4285f4), size 25 (lines 148-151)
   - Variable nodes: Yellow circle (#fbbc04) (lines 152-154)
   - Regular keywords: Default gray circle (lines 156-157)

   **Edge Styling:**
   - Dataset edges: Blue dashed lines (lines 175-176)
   - Regular edges: Solid lines (line 178)

   **Hover Tooltips:**
   - `title=f"Dataset: {src}"` creates hover text (line 151)
   - NOT clickable popups (no JavaScript click handlers)

6. **Export Functions** (Lines 183-195)
   - `export_csv()`: Saves to CSV using pandas
   - `export_json()`: Saves to JSON using json.dump

**Important Notes:**

- Graph is CLEARED on each store operation (line 28)
- Relation names must match regex `^[A-Z_][A-Z0-9_]*$` (line 59)
- PyVis repulsion settings: `node_distance=200, spring_length=300` (line 180)

---

### Module 4: `qa_module.py` (216 lines)

**Purpose:** RAG-based Q&A system using Ollama LLMs

**Class:** `QASystem`

**Initialization** (Lines 17-23):
```python
def __init__(self, model_name="llama3"):
    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
    self.llm_model = model_name  # Default: llama3
    self.documents = {}  # {filename: full_text}
    self.embeddings = {}  # {filename: chunk_embeddings}
    self.chunks = {}  # {filename: [chunks]}
```

**Key Functions:**

1. **add_document** (Lines 50-75)
   - Extracts text from PDF using pdfplumber
   - Cleans text (line 60)
   - Creates chunks (800 words, 200 overlap) (line 66)
   - Generates embeddings (line 71)
   - Prints: `"Added {filename} with {len(chunks)} chunks"`

2. **find_relevant_chunks** (Lines 86-114)
   - Encodes query (line 92)
   - Calculates cosine similarity (line 99)
   - Filters chunks with score > 0.15 (line 105)
   - Returns top_k chunks (default: 5) (line 114)

3. **generate_answer** (Lines 116-157)

   **Prompt Structure** (Lines 128-137):
   ```
   You are a helpful assistant analyzing research documents.
   Based on the following context from the uploaded PDFs, answer the question.
   If the answer is not in the context, say so clearly.

   Context from documents:
   {context}

   User Question: {query}

   Please provide a clear, concise answer based on the information provided.
   ```

   **LLM Call** (Lines 141-144):
   ```python
   response = ollama.chat(
       model=self.llm_model,  # "llama3" by default
       messages=[{"role": "user", "content": prompt}]
   )
   ```

   **Source Citations** (Lines 148-151):
   ```python
   sources = list(set([chunk['filename'] for chunk in relevant_chunks]))
   answer += f"\n\n📚 Sources: {', '.join(sources)}"
   ```

4. **answer_question** (Lines 159-174)
   - Main entry point
   - Calls `find_relevant_chunks(query)` (line 165)
   - Calls `generate_answer(query, relevant_chunks)` (line 173)

**Singleton Instance:**
```python
qa_system = QASystem()  # Line 216
```

**Performance:**
- Average response time: 42 seconds (Llama3 7B)
- Chunk size: 800 words
- Overlap: 200 words
- Top-k chunks: 5 (default)

---

### Module 5: `storing.py` (48 lines)

**Purpose:** CLI script for processing single PDFs

**Usage:**
```bash
python storing.py path/to/paper.pdf
```

**Process Flow:**
1. Takes PDF path from `sys.argv[1]` (default: "temp.pdf")
2. Extracts keywords and relations: `process(pdf_path, k=15)`
3. Prints keywords and relations
4. Creates relation dictionary (multiple relations per pair)
5. Stores in Neo4j
6. Exports to CSV and JSON
7. Generates and displays graph: `graph.show("graph.html")`

**Output Files:**
- `extracted_relations.csv`
- `extracted_relations.json`
- `graph.html`

**Key Difference from Frontend:**
- **CLI:** Processes one PDF at a time, auto-opens graph.html
- **Frontend:** Processes multiple PDFs, displays graph in Streamlit

---

### Module 6: `evaluation_metrics.py` (349 lines)

**Purpose:** RAGAS-style evaluation for RAG systems

**Classes:**

#### 6.1 **RAGASEvaluator** (Lines 13-220)

**Initialization:**
```python
def __init__(self, qa_system, model_name: str = "llama3"):
    self.qa_system = qa_system
    self.model_name = model_name
    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
```

**Metric 1: Faithfulness** (Lines 31-95)

**Purpose:** Measure how much of answer is grounded in context (hallucination detection)

**Algorithm:**
1. Clean answer (remove source citations) (lines 48-49)
2. Split into sentences (line 52)
3. Filter short sentences (<20 chars) (line 53)
4. Encode sentences and context (lines 66-67)
5. Calculate cosine similarity per sentence (line 77)
6. Count sentences with similarity > 0.35 (threshold) (line 82)
7. Faithfulness = verified_count / total_sentences (line 87)
8. Fallback: If strict threshold gives 0, use partial credit (lines 89-93)

**Score Range:** 0.0 to 1.0 (higher = less hallucination)

**Metric 2: Relevancy** (Lines 97-131)

**Purpose:** Measure how well answer addresses the question

**Algorithm:**
1. Clean answer (line 114)
2. Encode query and answer (lines 122-123)
3. Calculate cosine similarity (line 126)
4. Return similarity as relevancy score

**Score Range:** 0.0 to 1.0 (higher = better relevance)

**Combined Score:**
```python
combined_score = (faithfulness + relevancy) / 2
```

**evaluate_test_set** (Lines 164-220):
- Runs multiple queries
- Calculates aggregate metrics:
  - `avg_faithfulness`
  - `avg_relevancy`
  - `combined_score`
  - `std_faithfulness`
  - `std_relevancy`

#### 6.2 **RetrievalEvaluator** (Lines 223-349)

**Purpose:** Evaluate retrieval quality (separate from answer quality)

**Metrics:**

1. **MRR (Mean Reciprocal Rank)** (Lines 233-247)
   ```python
   for i, doc in enumerate(retrieved_docs, 1):
       if doc['filename'] in relevant_docs:
           return 1.0 / i  # Reciprocal of rank
   ```

2. **NDCG@K (Normalized Discounted Cumulative Gain)** (Lines 249-274)
   - DCG: Weighted by position (discount factor: 1/log2(i+1))
   - IDCG: Ideal DCG (all relevant docs at top)
   - NDCG = DCG / IDCG

3. **Recall@K** (Lines 276-294)
   ```python
   num_relevant_retrieved / len(relevant_docs)
   ```

4. **Hit Rate** (Lines 296-308)
   - Binary: 1.0 if any relevant doc found, else 0.0

5. **Diversity** (Lines 310-348)
   - Calculates pairwise similarity between retrieved docs
   - Diversity = 1 - average_similarity
   - Range: 0.0 (all identical) to 1.0 (all unique)

---

### Module 7: `compare_llm_models.py` (398 lines)

**Purpose:** Compare LLM models for RAG answer quality

**Class:** `GPT2QASystem` (extends `QASystem`)

**Supported Models:**
1. **Mistral 7B** (via Ollama) - Default inference model
2. **GPT-2** (via Transformers) - Lightweight baseline
3. **Qwen 2.5 7B** (via Ollama) - Reasoning specialist

**GPT-2 Integration** (Lines 24-106):

**Initialization:**
```python
if model_name == "gpt2":
    self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
```

**Generation Strategy:**
- Shortened context: 800 chars (vs 2000+ for Ollama models)
- Top 3 chunks only (vs 5 for others)
- Max new tokens: 150
- Temperature: 0.7
- Prompt truncation: 512 tokens (line 73)

**Test Queries** (Lines 108-121):
```python
[
    "What datasets are mentioned in the documents?",
    "What are the main findings and results?",
    "What methodologies are used?",
    # ... 10 total queries
]
```

**evaluate_llm_model** (Lines 124-239):

**Process:**
1. Initialize QA system with specified LLM (line 144)
2. Load PDFs from folder (lines 147-159)
3. For each query:
   - Retrieve chunks (line 170)
   - Generate answer (line 173)
   - Store detailed results (lines 176-193)
4. Run RAGAS evaluation (lines 206-209)
5. Merge RAGAS scores into detailed results (lines 212-216)

**Output Structure:**
```python
{
    'model_name': 'mistral',
    'num_pdfs': 5,
    'num_queries': 10,
    'ragas_metrics': {
        'avg_faithfulness': 0.33,
        'avg_relevancy': 0.45,
        'combined_score': 0.39
    },
    'detailed_query_results': [
        {
            'query_id': 1,
            'query': "What datasets...",
            'num_chunks_retrieved': 5,
            'retrieved_chunks': [...],
            'answer': "...",
            'faithfulness_score': 0.35,
            'relevancy_score': 0.42
        },
        # ... more results
    ]
}
```

**compare_llm_models** (Lines 242-378):

**Output:**
- JSON file: `llm_comparison_{timestamp}.json`
- Comparison table (faithfulness, relevancy, combined)
- Key findings (best faithfulness, best relevancy, best overall)
- Analysis (% better/worse than baseline)

**Main Function** (Lines 381-397):
```python
PDF_FOLDER = "/home/ad1457@students.ad.unt.edu/Downloads/pdf_folder"
OUTPUT_DIR = "llm_comparison"

compare_llm_models(PDF_FOLDER, OUTPUT_DIR)
```

---

### Module 8: `test_qa_mistral.py` (235 lines)

**Purpose:** Test harness for Q&A system

**Test Categories:**

1. Dataset Identification (4 questions)
2. Variable/Feature Extraction (3 questions)
3. Geographic Scope (3 questions)
4. Temporal Coverage (3 questions)
5. Contextual Metadata (3 questions)
6. Sample Use-Case Questions (4 questions)

**Total:** 20 test questions

**Test Flow:**

1. **Setup** (Lines 72-87)
   - Check PDF exists
   - Initialize QA system with "llama3" model
   - Print header with model info

2. **Load PDF** (Lines 89-108)
   - Add document to QA system
   - Measure load time
   - Print number of chunks created

3. **Run Questions** (Lines 110-165)
   - Loop through categories and questions
   - Measure answer time per question
   - Store results
   - 0.5s pause between questions

4. **Print Summary** (Lines 167-196)
   - Total questions, successful, failed
   - Total time, average time per question
   - Category breakdown

5. **Save Results** (Lines 198-220)
   - Output file: `qa_test_results_mistral.txt`
   - Format: Category, Question, Answer, Time

**Sample Output:**
```
====================================================================================================
Q&A TEST RESULTS - LLAMA 3
====================================================================================================

PDF: 2405.08174v2 (2).pdf
Model: LLaMA 3
Date: 2025-10-12 13:03:52

----------------------------------------------------------------------------------------------------
Category: Dataset Identification
Question: Which datasets are referenced in this paper?

Answer:
According to the uploaded PDFs, the following datasets are referenced in this paper:

1. Synthetic data (generated for experimentation)
2. Real-world Arctic data (specifically, Pan-Arctic sea ice concentrations and atmospheric processes)

📚 Sources: 2405.08174v2 (2).pdf

Time: 47.00s
----------------------------------------------------------------------------------------------------
```

---

### Module 9: `evaluation/comparisons/compare_llm_models.py` (412 lines)

**Purpose:** Alternative LLM comparison script with standalone implementation

**Key Differences from `Code/compare_llm_models.py`:**

1. **Self-contained RAG system** (no dependency on `qa_module.py`)
2. **Direct PDF loading** (Lines 96-114)
3. **Command-line arguments** (Lines 392-407)
4. **More lenient faithfulness threshold** (0.3 vs 0.35) (Line 255)

**Class:** `LLMEvaluator`

**Initialization** (Lines 32-65):
```python
def __init__(self, pdf_folder: str):
    self.pdf_folder = pdf_folder
    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load GPT-2
    self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    self.gpt2_pipeline = pipeline("text-generation", ...)
```

**Test Queries** (Lines 54-65):
- Same 10 queries as `test_qa_mistral.py`

**GPT-2 Generation** (Lines 139-177):
- Context limit: 500 chars (vs 800 in other script)
- Top 3 chunks
- Max new tokens: 150
- Uses `pipeline` instead of manual generation

**Ollama Generation** (Lines 179-214):
- Supports: llama3, qwen2.5:7b, mistral
- Full context (no truncation)
- Top 5 chunks

**Faithfulness Calculation** (Lines 216-258):
- Checks each sentence against ALL context chunks (not just combined)
- Takes max similarity across chunks (line 250-252)
- Threshold: 0.3 (more lenient than 0.35)

**Command-line Usage:**
```bash
python compare_llm_models.py \
  --pdf_folder ~/Downloads/pdf_folder \
  --models gpt2 llama3 qwen2.5:7b \
  --output llm_comparison_results.json
```

---

### Module 10: `evaluation/comparisons/README.md`

**Purpose:** Documentation for LLM comparison evaluation

**Key Information:**

1. **Supported Models:**
   - GPT-2 (Hugging Face Transformers)
   - Llama3 (Ollama)
   - Qwen 2.5 7B (Ollama)
   - Mistral 7B (Ollama, optional)

2. **Installation:**
   - Python dependencies: transformers, torch, sentence-transformers
   - Ollama installation + model pulling

3. **Metrics:**
   - Faithfulness (0.0-1.0) - hallucination detection
   - Relevancy (0.0-1.0) - question answering quality
   - Combined Score - average of both

4. **GPT-2 Limitations:**
   - Smaller context window
   - Only top 3 chunks (vs 5 for others)
   - Generation limited to 150 tokens

5. **Memory Requirements:**
   - GPT-2: ~500MB RAM (CPU) or ~1GB GPU
   - Llama3 7B: ~8GB RAM
   - Qwen 2.5 7B: ~8GB RAM
   - Mistral 7B: ~8GB RAM

6. **Troubleshooting:**
   - "Ollama not available" - GPT-2 will still work
   - Out of memory - reduce PDFs or use CPU
   - Slow generation - use GPU

---

## 4. System Architecture & Data Flow

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit)                      │
│  - PDF Upload                                                │
│  - Q&A Chat Interface                                        │
│  - Knowledge Graph Visualization                             │
│  - Export Options (CSV/JSON)                                 │
└────────────┬─────────────────────────────────┬──────────────┘
             │                                 │
             │                                 │
   ┌─────────▼──────────┐          ┌──────────▼─────────────┐
   │  Q&A PIPELINE      │          │  KG GENERATION PIPELINE│
   │                    │          │                         │
   │  1. PDF → Text     │          │  1. PDF → Text          │
   │  2. Text → Chunks  │          │  2. Text → Keywords     │
   │  3. Chunks → FAISS │          │  3. Keywords → Pairs    │
   │  4. Query → Chunks │          │  4. Pairs → Relations   │
   │  5. Chunks+Query → │          │  5. Relations → Neo4j   │
   │     LLM → Answer   │          │  6. Neo4j → PyVis       │
   └────────────────────┘          └─────────────────────────┘
             │                                 │
             │                                 │
   ┌─────────▼──────────┐          ┌──────────▼─────────────┐
   │   OLLAMA (LLM)     │          │   NEO4J (GRAPH DB)     │
   │  - Llama3 (8B)     │          │  - Keyword Nodes       │
   │  - Mistral (7B)    │          │  - Dataset Nodes       │
   │  - Qwen 2.5 (7B)   │          │  - Variable Nodes      │
   │  - GPT-2 (1.5B)    │          │  - Relationship Edges  │
   └────────────────────┘          └────────────────────────┘
```

### 4.2 Knowledge Graph Generation Flow

```
┌──────────┐
│   PDF    │
└────┬─────┘
     │
     ▼
┌──────────────────────┐
│ Text Extraction      │
│ (pdfplumber)         │
│ - Extract all pages  │
│ - Clean text         │
│ - Remove stopwords   │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Keyword Extraction   │
│ (TF-IDF+YAKE+KeyBERT)│
│ - Check for "Keywords"│
│ - Ensemble scoring   │
│ - Filter generic     │
│ - Return top-k       │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Relation Extraction  │
│ (LLaMA 3)            │
│ - Generate pairs     │
│ - Split into chunks  │
│ - LLM per chunk      │
│ - Validate relations │
│ - Score + rank       │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Dataset Extraction   │
│ (Mistral)            │
│ - First 12K chars    │
│ - Extract: source,   │
│   variables, period, │
│   location           │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Neo4j Storage        │
│ - Create nodes       │
│ - Create edges       │
│ - Link datasets      │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ PyVis Visualization  │
│ - Style nodes        │
│ - Style edges        │
│ - Generate HTML      │
└──────────────────────┘
```

### 4.3 Q&A System Flow

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌──────────────────────┐
│ Encode Query         │
│ (all-MiniLM-L6-v2)   │
│ - 384-dim embedding  │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Semantic Search      │
│ (FAISS Cosine Sim)   │
│ - Compare with chunks│
│ - Filter score>0.15  │
│ - Return top-5       │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Build Prompt         │
│ - Add context chunks │
│ - Add query          │
│ - Add instructions   │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ LLM Generation       │
│ (Ollama/GPT-2)       │
│ - Generate answer    │
│ - Add source cites   │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Return Answer        │
│ - Display in chat    │
│ - Show sources       │
└──────────────────────┘
```

### 4.4 Evaluation Pipeline

```
┌──────────────────────┐
│  Test Queries        │
│  (10 questions)      │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│  For Each Model:     │
│  - GPT-2             │
│  - Llama3            │
│  - Qwen 2.5          │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│  Generate Answers    │
│  - Retrieve chunks   │
│  - Generate via LLM  │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│  Calculate RAGAS     │
│  - Faithfulness      │
│    (sentence-context)│
│  - Relevancy         │
│    (query-answer)    │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│  Aggregate Results   │
│  - Avg faithfulness  │
│  - Avg relevancy     │
│  - Combined score    │
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│  Generate Report     │
│  - Comparison table  │
│  - Best model        │
│  - Save to JSON      │
└──────────────────────┘
```

---

## 5. Technology Stack

### 5.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **streamlit** | 1.32.2 | Web UI framework |
| **pdfplumber** | 0.10.3 | PDF text extraction |
| **nltk** | 3.8.1 | Text preprocessing (stopwords, lemmatization) |
| **spacy** | 3.7.4 | NLP text processing |
| **scikit-learn** | 1.4.2 | TF-IDF, cosine similarity |
| **keybert** | 0.8.3 | BERT-based keyword extraction |
| **yake** | 0.4.8 | Unsupervised keyword extraction |
| **sentence-transformers** | 2.6.1 | Semantic embeddings (all-MiniLM-L6-v2) |
| **torch** | ≥1.13.0 | Deep learning backend |
| **pyvis** | 0.3.2 | Interactive network visualization |
| **neo4j** | 5.19.0 | Graph database driver |
| **ollama** | 0.1.7 | Local LLM serving |
| **pandas** | 2.2.2 | Data manipulation |
| **numpy** | 1.26.4 | Numerical operations |
| **networkx** | 3.2.1 | Graph algorithms |
| **transformers** | ≥4.30.0 | Hugging Face models (GPT-2) |
| **ragas** | ≥0.1.0 | RAG evaluation metrics |
| **fuzzywuzzy** | 0.18.0 | String matching |
| **python-Levenshtein** | 0.23.0 | Edit distance calculations |

### 5.2 External Services

1. **Ollama** (Local LLM Server)
   - Models: llama3, mistral, qwen2.5:7b
   - Default port: 11434
   - Context window: ~8K tokens

2. **Neo4j Aura** (Cloud Graph Database)
   - URI: `neo4j+s://0d4ad98d.databases.neo4j.io`
   - User: `neo4j`
   - Password: (stored in `neo4j_storage.py`)

### 5.3 Embedding Models

1. **all-MiniLM-L6-v2** (sentence-transformers)
   - Dimensions: 384
   - Max sequence length: 256 tokens
   - Used for: Query encoding, chunk encoding, similarity scoring
   - Speed: ~3000 sentences/sec (GPU)

### 5.4 LLM Models

| Model | Parameters | Context | Use Case | Speed |
|-------|-----------|---------|----------|-------|
| **Llama3** | 8B | 8K | Q&A generation, relation extraction | ~2 tokens/sec |
| **Mistral 7B** | 7B | 8K | Dataset extraction | ~2 tokens/sec |
| **Qwen 2.5 7B** | 7B | 8K | Comparison, Q&A | ~2 tokens/sec |
| **GPT-2** | 1.5B | 1K | Baseline comparison | ~10 tokens/sec |

---

## 6. Database Schema (Neo4j)

### 6.1 Node Types

#### 6.1.1 Keyword Node
```cypher
(:Keyword {
    name: String  // e.g., "sea_ice_concentration"
})
```

#### 6.1.2 Variable Node (extends Keyword)
```cypher
(:Keyword:Variable {
    name: String  // e.g., "temperature"
})
```

#### 6.1.3 Dataset Node
```cypher
(:Dataset {
    name: String,           // e.g., "NSIDC Sea Ice Index"
    time_period: String,    // e.g., "1979-2023"
    location: String        // e.g., "Arctic Ocean"
})
```
**Missing:** `link: String` (URL/DOI) - not currently stored

### 6.2 Relationship Types

#### 6.2.1 Keyword Relationships
```cypher
(:Keyword)-[:CAUSES]->(:Keyword)
(:Keyword)-[:INCREASES]->(:Keyword)
(:Keyword)-[:DECREASES]->(:Keyword)
(:Keyword)-[:AFFECTS]->(:Keyword)
(:Keyword)-[:RELATES_TO]->(:Keyword)
(:Keyword)-[:CORRELATES_WITH]->(:Keyword)
```
*Note:* Relation names are dynamically created from LLM output

#### 6.2.2 Dataset Relationships
```cypher
(:Dataset)-[:HAS_VARIABLE]->(:Variable)
(:Dataset)-[:EXTRACTED_FROM]->(:Keyword)
```

### 6.3 Cypher Queries

**Clear Database:**
```cypher
MATCH (n) DETACH DELETE n
```

**Get All Relations:**
```cypher
MATCH (a:Keyword)-[r]->(b:Keyword)
RETURN a.name AS source, type(r) AS relation, b.name AS target
```

**Get Dataset Info:**
```cypher
MATCH (d:Dataset)
RETURN d.name, d.time_period, d.location
```

**Get Variables for Dataset:**
```cypher
MATCH (d:Dataset)-[:HAS_VARIABLE]->(v:Variable)
WHERE d.name = $dataset_name
RETURN v.name
```

---

## 7. Evaluation Framework

### 7.1 RAGAS Metrics

#### Faithfulness

**Definition:** Proportion of answer sentences that can be verified from context

**Formula:**
```
Faithfulness = (Verified Sentences) / (Total Sentences)

Where:
  Verified = cosine_similarity(sentence_emb, context_emb) > 0.35
```

**Threshold:**
- **Standard:** 0.35 (used in `evaluation_metrics.py`)
- **Lenient:** 0.30 (used in `evaluation/comparisons/compare_llm_models.py`)

**Interpretation:**
- **1.0:** All sentences grounded in context (no hallucination)
- **0.5:** Half of sentences verified
- **0.0:** No sentences verified (high hallucination)

**Fallback:** If strict threshold gives 0, uses partial credit: `avg_similarity * 0.5`

#### Relevancy

**Definition:** Semantic similarity between query and answer

**Formula:**
```
Relevancy = cosine_similarity(query_emb, answer_emb)
```

**Range:** 0.0 to 1.0

**Interpretation:**
- **>0.7:** Highly relevant answer
- **0.4-0.7:** Moderately relevant
- **<0.4:** Poorly addresses question

#### Combined Score

```
Combined = (Faithfulness + Relevancy) / 2
```

### 7.2 Retrieval Metrics

#### MRR (Mean Reciprocal Rank)

**Formula:**
```
MRR = 1 / (Rank of first relevant document)
```

**Example:**
- Relevant doc at position 1: MRR = 1.0
- Relevant doc at position 3: MRR = 0.33

#### NDCG@K (Normalized Discounted Cumulative Gain)

**Formula:**
```
DCG@K = Σ(relevance_i / log2(i + 1)) for i in 1..K

NDCG@K = DCG@K / IDCG@K
```

**Range:** 0.0 to 1.0

#### Recall@K

**Formula:**
```
Recall@K = (Relevant docs in top-K) / (Total relevant docs)
```

#### Hit Rate

**Formula:**
```
Hit Rate = 1 if any relevant doc in retrieved, else 0
```

#### Diversity

**Formula:**
```
Diversity = 1 - (Average pairwise similarity)
```

**Range:** 0.0 (all identical) to 1.0 (all unique)

### 7.3 Benchmark Results

From `qa_test_results_mistral.txt` (20 queries, Llama3):

| Metric | Value |
|--------|-------|
| **Total Questions** | 20 |
| **Successful Answers** | 20 (100%) |
| **Failed** | 0 |
| **Total Time** | 864.20s |
| **Avg Time/Question** | 43.21s |

**Category Breakdown:**

| Category | Questions | Success Rate | Avg Time |
|----------|-----------|--------------|----------|
| Dataset Identification | 4 | 4/4 (100%) | 34.38s |
| Variable/Feature Extraction | 3 | 3/3 (100%) | 37.78s |
| Geographic Scope | 3 | 3/3 (100%) | 37.73s |
| Temporal Coverage | 3 | 3/3 (100%) | 45.61s |
| Contextual Metadata | 3 | 3/3 (100%) | 55.90s |
| Sample Use-Case | 4 | 4/4 (100%) | 48.91s |

From `QUICK_MODULE_GUIDE.md`:

**LLM Comparison Results:**

| Model | Faithfulness | Relevancy | Combined | Ranking |
|-------|--------------|-----------|----------|---------|
| **Qwen 2.5 7B** | 0.33 | 0.45 | **0.39** | 🥇 Best |
| **Llama3 8B** | 0.25 | 0.50 | **0.38** | 🥈 2nd |
| **GPT-2 1.5B** | 0.18 | 0.32 | **0.25** | 🥉 3rd |

**Retrieval Comparison (Hybrid vs Single Method):**

| Method | MRR | NDCG@10 | Improvement |
|--------|-----|---------|-------------|
| **Hybrid (FAISS+BM25)** | **0.82** | **0.76** | Baseline |
| FAISS only | 0.71 | 0.68 | -13% / -11% |
| BM25 only | 0.65 | 0.61 | -21% / -20% |

**Improvement:** Hybrid retrieval gives +15% MRR, +11% NDCG over best single method

---

## 8. Configuration & Credentials

### 8.1 Neo4j Credentials

**Location:** `Code/neo4j_storage.py` (Lines 7-10)

```python
NEO4J_URI = 'neo4j+s://0d4ad98d.databases.neo4j.io'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'l2eTsa3JmSPkwWoCCNszhUyvkxkapl3WwN2oHzJZJ6E'
```

**Security Note:** Credentials are hardcoded (not best practice for production)

**Recommended Fix:** Use environment variables
```python
import os
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
```

### 8.2 Ollama Configuration

**Default Port:** 11434
**Models Required:**
- `ollama pull llama3`
- `ollama pull mistral`
- `ollama pull qwen2.5:7b`

**Check Status:**
```bash
ollama list
ollama serve
```

### 8.3 Streamlit Configuration

**Page Config** (frontend_light.py, Lines 12-16):
```python
st.set_page_config(
    page_title="Polar Knowledge Discovery (PolarKD) Toolkit",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

**Run Command:**
```bash
streamlit run frontend_light.py
```

**Default URL:** http://localhost:8501

### 8.4 Embedding Model

**Model:** `all-MiniLM-L6-v2`
**Auto-download:** Yes (via sentence-transformers)
**Cache Location:** `~/.cache/torch/sentence_transformers/`

### 8.5 Keyword Extraction Parameters

**Default k:** 15 keywords (adjustable in UI: 5-50)

**Ensemble Weights:**
- TF-IDF: 25%
- YAKE: 50%
- KeyBERT: 25%

**Chunking:**
- Chunk size: 2000 characters
- Overlap: 300 characters

### 8.6 Q&A System Parameters

**Chunking:**
- Chunk size: 800 words
- Overlap: 200 words

**Retrieval:**
- Top-k chunks: 5
- Similarity threshold: 0.15

**LLM:**
- Default model: llama3
- Temperature: Not specified (uses Ollama defaults)

---

## 9. Performance Benchmarks

### 9.1 Processing Speed

| Operation | Time | Notes |
|-----------|------|-------|
| **PDF Text Extraction** | ~1-2s | per 10-page PDF |
| **Keyword Extraction** | ~3-5s | for 15 keywords |
| **Relation Extraction** | ~30-60s | depends on chunk count |
| **Dataset Extraction** | ~10-15s | Mistral LLM call |
| **Neo4j Storage** | ~1-2s | for 15 nodes, 20 edges |
| **Graph Visualization** | ~1s | PyVis generation |
| **Q&A Answer Generation** | ~40-45s | Llama3 (per question) |

**Total Time (Single PDF → Knowledge Graph):** ~60-90 seconds

### 9.2 Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| **Streamlit App** | ~300MB | Base UI |
| **Embedding Model** | ~100MB | all-MiniLM-L6-v2 |
| **PDF Processing** | ~50MB | per PDF in memory |
| **Neo4j Driver** | ~50MB | Connection overhead |
| **Ollama (Llama3 8B)** | ~8GB | LLM serving |
| **GPT-2 (if used)** | ~500MB (CPU) | 1GB (GPU) |

**Total System:** ~9GB (with Ollama)

### 9.3 Token Limits

| Model | Max Tokens | Actual Usage |
|-------|-----------|--------------|
| **Llama3** | 8,192 | ~2,000-3,000 (Q&A) |
| **Mistral** | 8,192 | ~1,500 (dataset extraction) |
| **Qwen 2.5** | 8,192 | ~2,000-3,000 (Q&A) |
| **GPT-2** | 1,024 | ~512 (truncated) |

### 9.4 Scalability

**Multi-PDF Processing:**

| PDFs | Keywords | Relations | Time | Memory |
|------|----------|-----------|------|--------|
| 1 | 15 | ~20 | ~90s | ~9GB |
| 5 | 75 | ~100 | ~450s | ~9.5GB |
| 10 | 150 | ~200 | ~900s | ~10GB |

**Q&A Scalability:**

| Documents | Chunks | Load Time | Query Time |
|-----------|--------|-----------|------------|
| 1 | 10 | ~1.5s | ~42s |
| 5 | 50 | ~7s | ~45s |
| 10 | 100 | ~15s | ~50s |

**Note:** Query time increases slightly with more documents due to larger semantic search space.

---

## 10. Entry Points & Usage

### 10.1 Main UI (Streamlit)

**Start:**
```bash
cd Knowledge_graph/Code
streamlit run frontend_light.py
```

**Access:** http://localhost:8501

**Features:**
1. Upload PDFs (drag-drop or click)
2. Send to Q&A (for questions only)
3. Generate Knowledge Graph (keywords + relations)
4. Ask questions in chat
5. View graph visualization
6. Export CSV/JSON

### 10.2 CLI Processing

**Single PDF:**
```bash
cd Knowledge_graph/Code
python storing.py path/to/paper.pdf
```

**Output:**
- `extracted_relations.csv`
- `extracted_relations.json`
- `graph.html` (auto-opens)

### 10.3 Q&A Testing

**Run Test Suite:**
```bash
cd Knowledge_graph/Code
python test_qa_mistral.py
```

**Requirements:**
- PDF at: `/Users/ajithkumardugyala/Downloads/2405.08174v2 (2).pdf`
- Ollama running with llama3

**Output:**
- `qa_test_results_mistral.txt`

### 10.4 LLM Comparison

**Run Comparison:**
```bash
cd Knowledge_graph/Code
python compare_llm_models.py
```

**Or with evaluation script:**
```bash
cd Knowledge_graph/Code/evaluation/comparisons
python compare_llm_models.py \
  --pdf_folder /path/to/pdfs \
  --models gpt2 llama3 qwen2.5:7b \
  --output results.json
```

**Output:**
- JSON file with detailed results
- Console comparison table

### 10.5 Direct Module Usage

**Keyword Extraction:**
```python
from keywords_extraction import process

nodes, relations, dataset_info = process("paper.pdf", k=15)
print(f"Keywords: {nodes}")
print(f"Relations: {relations}")
print(f"Dataset: {dataset_info}")
```

**Q&A System:**
```python
from qa_module import qa_system

# Add document
qa_system.add_document("paper.pdf", pdf_path="path/to/paper.pdf")

# Ask question
answer = qa_system.answer_question("What datasets are used?")
print(answer)
```

**Neo4j Storage:**
```python
from neo4j_storage import Neo4jConnector

neo = Neo4jConnector()
neo.store_keywords_and_relations(nodes, relations, dataset_info)
rels = neo.retrieve_relations()
graph = neo.generate_graph(rels)
graph.show("graph.html")
neo.close()
```

---

## 11. Known Issues & Limitations

### 11.1 Dataset Extraction Limitations

**Issue:** Only checks first 12,000 characters (≈4-5 pages)

**Location:** `keywords_extraction.py` Line 491
```python
text_sample = text[:12000] if len(text) > 12000 else text
```

**Impact:**
- Datasets mentioned later in paper may be missed
- Works well for papers with datasets in Introduction/Methods sections

**Recommended Fix:**
- Loop through ALL chunks (like relation extraction does)
- Aggregate dataset info from multiple chunks

### 11.2 Neo4j Credentials Hardcoded

**Issue:** Database credentials stored in source code

**Location:** `neo4j_storage.py` Lines 7-10

**Security Risk:** High (credentials exposed in version control)

**Recommended Fix:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
```

### 11.3 Graph Cleared on Each Store

**Issue:** Existing graph is deleted before storing new data

**Location:** `neo4j_storage.py` Line 28
```python
session.run("MATCH (n) DETACH DELETE n")
```

**Impact:**
- Cannot incrementally add PDFs to existing graph
- Multi-PDF processing creates combined graph, but re-processing loses history

**Recommended Fix:**
- Add parameter: `clear_existing=True` (default False)
- Use MERGE instead of CREATE for idempotent operations

### 11.4 Dataset Nodes Missing URL/Link Field

**Issue:** Dataset nodes don't store URLs or DOIs

**Location:** `neo4j_storage.py` Lines 67-78

**Current Schema:**
```cypher
(:Dataset {name, time_period, location})
```

**Recommended Schema:**
```cypher
(:Dataset {name, time_period, location, link, doi})
```

**Required Changes:**
1. Update `extract_dataset_info()` to extract links
2. Update `_create_dataset()` to store link property
3. Update graph visualization to show clickable links

### 11.5 No Click Events on Graph Nodes

**Issue:** Graph nodes only have hover tooltips, not click popups

**Location:** `neo4j_storage.py` Lines 148-157

**Current Implementation:**
```python
net.add_node(src, label=src, color='#4285f4',
             title=f"Dataset: {src}")  # Only hover tooltip
```

**Recommended Fix:**
- Inject JavaScript click handlers into generated HTML
- Show popup with dataset info (link, variables, period, location)

**Implementation Complexity:** Medium (~4 hours)

### 11.6 GPT-2 Context Window Too Small

**Issue:** GPT-2 limited to 1024 tokens, requires aggressive truncation

**Location:** `compare_llm_models.py` Lines 59-65

**Workaround:**
- Only top 3 chunks (vs 5)
- Context truncated to 500-800 chars
- Max generation: 150 tokens

**Impact:**
- GPT-2 scores are artificially lower due to less context
- Not a fair comparison with 8K context models

**Recommendation:**
- Use GPT-2 as baseline only
- Do not expect production-quality results

### 11.7 Relation Names Must Be Valid Cypher

**Issue:** LLM-generated relation names must match `^[A-Z_][A-Z0-9_]*$`

**Location:** `neo4j_storage.py` Lines 58-61

**Example:**
- ✅ Valid: "CAUSES", "INCREASES", "RELATES_TO"
- ❌ Invalid: "causes" (lowercase), "relates to" (space), "affects-negatively" (hyphen)

**Impact:**
- Some relations are skipped if LLM returns invalid format
- No feedback to user about skipped relations

**Recommended Fix:**
- Add relation name normalization:
  ```python
  rel = rel.upper().replace(" ", "_").replace("-", "_")
  rel = re.sub(r'[^A-Z0-9_]', '', rel)  # Remove invalid chars
  ```

### 11.8 No Hybrid Retrieval in Q&A

**Issue:** Q&A system uses only semantic search (FAISS), not hybrid

**Location:** `qa_module.py` Lines 86-114

**Current:**
- FAISS (dense retrieval) only
- Misses exact keyword matches

**Recommended:**
- Add BM25 (sparse retrieval)
- Combine scores: 60% FAISS + 40% BM25
- Add MMR re-ranking for diversity

**Expected Improvement:**
- +15% MRR
- +11% NDCG@10
(Based on QUICK_MODULE_GUIDE.md benchmarks)

### 11.9 Single LLM Model Per Session

**Issue:** Cannot switch LLM models in Streamlit UI

**Location:** `frontend_light.py`

**Current:**
- Uses global `qa_system = QASystem()` with hardcoded model

**Recommended:**
- Add model selector dropdown in UI
- Reinitialize `qa_system` when model changes

### 11.10 No Error Handling for Ollama Downtime

**Issue:** If Ollama service stops, app crashes with unclear error

**Location:** Multiple files (ollama.chat() calls)

**Current Error:**
```
Error generating answer: Connection refused
```

**Recommended Fix:**
```python
try:
    response = ollama.chat(...)
except Exception as e:
    return "⚠️ Ollama service unavailable. Please ensure Ollama is running (ollama serve)."
```

---

## 12. Quick Reference

### 12.1 Common Commands

```bash
# Start Ollama
ollama serve

# Pull models
ollama pull llama3
ollama pull mistral
ollama pull qwen2.5:7b

# Start Streamlit UI
cd Knowledge_graph/Code
streamlit run frontend_light.py

# Process single PDF (CLI)
python storing.py paper.pdf

# Run Q&A tests
python test_qa_mistral.py

# Run LLM comparison
python compare_llm_models.py

# Run evaluation comparison (with args)
cd evaluation/comparisons
python compare_llm_models.py --pdf_folder ~/pdfs --models gpt2 llama3 qwen2.5:7b
```

### 12.2 Important File Paths

| File | Purpose |
|------|---------|
| `frontend_light.py` | Main Streamlit UI |
| `keywords_extraction.py` | KG extraction pipeline |
| `neo4j_storage.py` | Graph database operations |
| `qa_module.py` | Q&A RAG system |
| `evaluation_metrics.py` | RAGAS metrics |
| `compare_llm_models.py` | LLM comparison |
| `requirements.txt` | Python dependencies |
| `Readme.md` | Project documentation |

### 12.3 Key Functions by Task

**Extract Keywords:**
```python
from keywords_extraction import extract_keywords, text_extraction
text = text_extraction("paper.pdf")
keywords = extract_keywords(text, k=15)
```

**Generate Knowledge Graph:**
```python
from keywords_extraction import process
from neo4j_storage import Neo4jConnector

nodes, relations, dataset_info = process("paper.pdf", k=15)
neo = Neo4jConnector()
neo.store_keywords_and_relations(nodes, relations, dataset_info)
graph = neo.generate_graph(neo.retrieve_relations())
graph.show("graph.html")
```

**Ask Questions:**
```python
from qa_module import qa_system

qa_system.add_document("paper.pdf", pdf_path="path/to/paper.pdf")
answer = qa_system.answer_question("What datasets are used?")
print(answer)
```

**Evaluate Answer Quality:**
```python
from evaluation_metrics import RAGASEvaluator

evaluator = RAGASEvaluator(qa_system, model_name="llama3")
result = evaluator.evaluate_single_query("What are the main findings?")
print(f"Faithfulness: {result['faithfulness']}")
print(f"Relevancy: {result['relevancy']}")
```

### 12.4 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Ollama not available" | Run `ollama serve` in separate terminal |
| "Neo4j connection refused" | Check credentials in `neo4j_storage.py` |
| "Out of memory" | Reduce number of PDFs, close other apps |
| "Model not found" | Pull model: `ollama pull llama3` |
| "Streamlit port in use" | Change port: `streamlit run frontend_light.py --server.port 8502` |
| Slow Q&A responses | Normal (40-45s), consider using GPT-4o API for speed |
| Graph not displaying | Check Neo4j connection, verify graph.html generated |
| Keywords extraction empty | Check if PDF has "Keywords" section or readable text |

### 12.5 Performance Tips

1. **Faster Q&A:**
   - Use fewer chunks (top_k=3 instead of 5)
   - Switch to lighter model (GPT-2 instead of Llama3)
   - Use OpenAI API (GPT-4o-mini ~2-3s response)

2. **Better Accuracy:**
   - Increase keywords (k=30 instead of 15)
   - Use hybrid retrieval (add BM25)
   - Use larger model (Qwen 2.5 > Llama3 > GPT-2)

3. **Lower Memory:**
   - Process PDFs one at a time
   - Use GPT-2 instead of Llama3
   - Clear session state between operations

---

## 13. Future Enhancement Opportunities

Based on identified limitations, here are recommended enhancements:

### 13.1 High Priority

1. **Dataset URL Extraction**
   - Modify `extract_dataset_info()` to extract URLs/DOIs
   - Store in Neo4j Dataset nodes
   - Display as clickable links in graph

2. **Hybrid Retrieval for Q&A**
   - Add BM25 (keyword search)
   - Combine with FAISS (semantic search)
   - Expected: +15% accuracy improvement

3. **Environment Variable Configuration**
   - Move Neo4j credentials to `.env` file
   - Add `.env.example` template
   - Update documentation

### 13.2 Medium Priority

4. **Incremental Graph Updates**
   - Add option to append to existing graph
   - Avoid clearing graph on each store

5. **Full-Document Dataset Extraction**
   - Loop through all chunks (not just first 12K chars)
   - Aggregate dataset info from entire paper

6. **LLM Model Selector in UI**
   - Dropdown to choose model (Llama3, Mistral, Qwen)
   - Dynamically reinitialize Q&A system

### 13.3 Low Priority

7. **Click Events on Graph Nodes**
   - Inject JavaScript for click handlers
   - Show popup with dataset details

8. **Better Error Handling**
   - Graceful Ollama failure messages
   - Retry logic for LLM calls
   - User-friendly error displays

9. **Multi-user Support**
   - Session-based graph storage
   - User authentication
   - Private document libraries

---

**Document Version:** 1.0
**Last Updated:** 2025-10-23
**Maintained By:** PolarKD Development Team
