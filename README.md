# Multi-Document RAG Assistant

This project lets you upload multiple PDFs, index them with embeddings, and ask questions across all documents.  
Answers are grounded in the source text and include citations.

## Project Structure

data/
raw_pdfs/ # store PDFs here
indexes/ # FAISS vectorstore saved here
src/
ingest/ # PDF loading + splitting
index/ # embeddings + FAISS persistence
query/ # retrieval logic + QA chain
ui/ # streamlit UI 
utils/ # config + helper scripts
configs/ # model + chunk settings