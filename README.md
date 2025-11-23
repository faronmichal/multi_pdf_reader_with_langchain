```markdown
# Multi-PDF Research Assistant

## Overview
Multi-PDF Research Assistant is a lightweight Retrieval-Augmented Generation (RAG) system that allows users to ask natural-language questions across multiple PDF documents.  
It automates PDF ingestion, text chunking, embedding, vector indexing with FAISS, and provides grounded answers using an LLM.  
Includes a Streamlit UI, CLI access, and a fully mocked test suite for safe, offline development.

---

## Features
- Upload and ingest multiple PDFs automatically
- Configurable text chunking and overlapping
- Generate embeddings for vector search
- Fast FAISS similarity search across all documents
- Grounded RAG answers with citations
- Interactive Streamlit interface and command-line usage
- Mocked pytest tests for CI/CD without API calls

---

## Project Structure


├── configs/                # Configuration files for models and pipelines
├── data/
│   ├── indexes/            # FAISS vector index files
│   └── raw_pdfs/           # Uploaded PDF documents
├── src/
│   ├── ingest/             # PDF loading, extraction, and chunking
│   ├── query/              # Retrieval and RAG answer generation
│   ├── ui/                 # Streamlit app
├── tests/
│   ├── test_ingest.py
│   ├── test_index.py
│   └── test_query.py
└── requirements.txt

````

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/faronmichal/multi_pdf_reader_with_langchain
cd multi_pdf_reader_with_langchain
````

### Create and Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Required only for real LLM calls. Tests run without this key.

---

## Usage

### Streamlit Web Interface

```bash
streamlit run src/ui/app.py
```

### Command-Line Interface

```bash
python src/query/query.py
```

---

## Testing

Run the full test suite:

```bash
pytest tests/
```

Tests cover:

* PDF ingestion and chunking (`test_ingest.py`)
* Embedding and FAISS indexing (`test_index.py`)
* Retrieval and QA pipeline (`test_query.py`)

---

## Workflow

1. **Upload PDFs** → stored in `data/raw_pdfs/`
2. **Ingest** → extract text and create chunks
3. **Embed** → generate vector embeddings for chunks
4. **Index** → store embeddings in FAISS under `data/indexes/`
5. **Query** → retrieve top-k chunks via similarity search
6. **Answer** → LLM generates grounded answers with citations

---

