import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# path
ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT_DIR / "data" / "indexes"

# load index function
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# add new pdf file
def add_pdf_to_index(uploaded_file):
    # temporary pdf file save
    temp_pdf_path = Path("temp_uploaded.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(str(temp_pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    new_store = FAISS.from_documents(chunks, embeddings)

    try:
        existing_store = load_vectorstore()
        existing_store.merge_from(new_store)
        existing_store.save_local(INDEX_DIR)
    except:
        # make new index if it doesn't exist
        new_store.save_local(INDEX_DIR)

    temp_pdf_path.unlink()  # delete temporary file

# loading index to qa
def build_qa():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    template = """
    You are a helpful assistant answering questions **only** using information from the provided text context.
    If the answer is not clearly stated in the context, reply exactly with:
    "I don't know based on the documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# ui start
st.set_page_config(page_title="PDF research assistant", page_icon="📄")
st.title("PDF research assistant")

# upload
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    add_pdf_to_index(uploaded)
    st.success("PDF was added to the data")

qa = build_qa()

# chat
query = st.text_input("Ask a question:")
if query:
    response = qa.invoke(query)
    answer = response["result"]
    sources = response["source_documents"]

    st.write("Answer:")
    st.write(answer)

    if answer.strip() != "I don't know based on the documents.":
        st.write("Sources:")
        for s in sources:
            name = s.metadata.get("source", "nieznany plik")
            page = s.metadata.get("page", "?")
            st.write(f"- **{name}** (strona {page})")
    else:
        st.write("No data in the documents.")