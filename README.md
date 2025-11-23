Jasne. Oto **naprawiony, jednolity, czysty README**, cały w **jednym** kod-blocku, bez dzielenia, bez łamania drabinki, z idealnym monospace, który **wygląda dobrze na GitHubie**.

```markdown
# Multi-PDF Research Assistant

## Overview
The **Multi-PDF Research Assistant** is a compact Retrieval-Augmented Generation (RAG) system that lets you ask natural-language questions across multiple PDF documents.  
It automates PDF ingestion, text chunking, embeddings, FAISS vector indexing, and grounded LLM-based question answering.  
Includes a Streamlit UI and fully mocked tests (no API calls required).

---

## Key Features
- **Multi-PDF ingestion** with automated extraction.
- **Configurable chunking** (size & overlap).
- **Fast FAISS vector search** stored locally.
- **RAG Q&A pipeline** with strict grounding + citations.
- **Streamlit UI** for real-time usage.
- **CLI mode** for terminal workflows.
- **Mocked pytest test suite** — safe for CI.

---

## Project Structure

```

├── configs/                # YAML configs for models, chunking, pipeline
├── data/
│   ├── indexes/            # FAISS index files
│   └── raw_pdfs/           # Uploaded source PDFs
├── src/
│   ├── ingest/             # PDF loading, cleaning, text extraction, chunking
│   ├── query/              # Retrieval + RAG answer generation
│   ├── ui/                 # Streamlit interface
│   └── utils/              # Embedding utils, file ops, common helpers
├── tests/                  # Fully mocked unit tests
│   ├── test_ingest.py
│   ├── test_index.py
│   └── test_query.py
└── requirements.txt

````

---

## Installation

### Clone the repository
```bash
git clone https://github.com/faronmichal/multi_pdf_reader_with_langchain
cd multi_pdf_reader_with_langchain
````

### Create & activate virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` at the project root:

```
OPENAI_API_KEY=sk-proj-...
```

Required only for **real** LLM calls — tests run without it.

---

## Running the Application

### Streamlit Web Interface

```bash
streamlit run src/ui/app.py
```

### CLI Query Mode

```bash
python src/query/query.py
```

---

## Testing

All tests use mocks (no FAISS building, no external APIs).

Run the suite:

```bash
pytest tests/
```

Includes:

* PDF ingestion & chunking (`test_ingest.py`)
* Embedding + FAISS index logic (`test_index.py`)
* Retrieval + QA pipeline (`test_query.py`)

---

## How It Works (Full Pipeline)

1. **Upload PDFs** → stored in `data/raw_pdfs/`
2. **Ingestion** → text extracted, cleaned, chunked
3. **Embedding** → chunks → embedding vectors
4. **Indexing** → FAISS index saved in `data/indexes/`
5. **Query** → similarity search retrieves top-k chunks
6. **LLM Answering** → grounded answer with citations

---
