# Multi-PDF Research Assistant

## Overview
The Multi-PDF Research Assistant is a compact Retrieval-Augmented Generation (RAG) system enabling natural-language queries across multiple PDF documents.  
The pipeline handles PDF ingestion, text chunking, embeddings, vector indexing with FAISS, and LLM-based question answering grounded strictly in retrieved document content.  
Includes a Streamlit UI for interactive use and fully mocked tests for safe, free CI execution.

---

## Key Features
- **Multi-PDF ingestion** with automatic text extraction.
- **Chunking pipeline** with adjustable chunk size & overlap.
- **Vector search using FAISS**, stored locally for speed and reproducibility.
- **RAG QA pipeline** with strict grounding + citations from matched chunks.
- **Streamlit-based UI** for real-time document uploads and chat.
- **CLI mode** for terminal workflows or scripting.
- **Mocked pytest suite** (no OpenAI API calls required).

---

## Project Structure

├── configs/                # Model, chunking, and pipeline configs
├── data/
│   ├── indexes/            # FAISS index files
│   └── raw_pdfs/           # Uploaded source PDFs
├── src/
│   ├── ingest/             # PDF loading + text extraction + chunking
│   ├── query/              # Retrieval logic + RAG answer generation
│   ├── ui/                 # Streamlit web interface
│   └── utils/              # Helper utilities (embedding, file ops, etc.)
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

### Create and activate virtual environment

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

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-proj-...
```

This is only required for real LLM calls.
All tests run without this key.

---

## Running the Application

### Streamlit Web Interface

Launch the interactive app:

```bash
streamlit run src/ui/app.py
```

### CLI Interface

Run a query directly from the terminal:

```bash
python src/query/query.py
```

---

## Testing

All tests are mocked, require no external services, and are safe for CI/CD.

Run the full test suite:

```bash
pytest tests/
```

Test coverage includes:

* PDF ingestion & chunking (`test_ingest.py`)
* Embedding + FAISS index logic (`test_index.py`)
* Retrieval + QA pipeline (`test_query.py`)

---

## How the System Works

1. **Upload PDFs** → Files saved to `data/raw_pdfs/`.
2. **Ingestion** → PDFs parsed, cleaned, split into text chunks.
3. **Embedding** → Text chunks embedded using configurable embedding models.
4. **Indexing** → FAISS index created and saved to `data/indexes/`.
5. **Querying** → User queries → top-k chunks retrieved via similarity search.
6. **LLM Answering** → Model generates grounded answer with citations from chunks.

