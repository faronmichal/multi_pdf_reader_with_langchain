import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# get project root folder automatically
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw_pdfs"
INDEX_DIR = ROOT_DIR / "data" / "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

def load_and_split_pdfs():
    documents = []

    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, file_name)
            print(f"Loading: {file_name}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    print(f" Loaded {len(documents)} pages total.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f" Split into {len(chunks)} chunks.")
    return chunks

def build_vectorstore(chunks):
    print("Creating embeddings")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Vector index saved to: {INDEX_DIR}")

if __name__ == "__main__":
    chunks = load_and_split_pdfs()
    build_vectorstore(chunks)
    print("Ingestion complete.")