import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# path
ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT_DIR / "data" / "indexes"


# load index function
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


# chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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


# loading index to qa (with memory)
def build_qa():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # add memory to keep conversation context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    template = """
    You are a helpful assistant answering questions only using information from the provided text context.
    If the answer is not clearly stated in the context, reply exactly with:
    "I don't know based on the documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa


# ui start
st.set_page_config(page_title="PDF research assistant", page_icon="📄")
st.title("PDF research assistant")

if "qa" not in st.session_state:
    try:
        st.session_state.qa = build_qa()
    except Exception:
        st.warning("No FAISS index found. Please upload a PDF first.")


# upload
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    add_pdf_to_index(uploaded)
    st.success("PDF was added to the data")
    st.session_state.qa = build_qa()


# reset conversation
if st.button("Reset conversation"):
    st.session_state.chat_history = []
    st.session_state.qa = build_qa()
    st.experimental_rerun()


# chat
query = st.text_input("Ask a question:")
if query and "qa" in st.session_state:
    st.session_state.chat_history.append({"role": "user", "content": query})

    # model with memory, uses conversation context
    response = st.session_state.qa({"question": query})
    answer = response["answer"]
    sources = response["source_documents"]

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.write("Conversation history:")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(f"You: {msg['content']}")
        else:
            st.write(f"AI: {msg['content']}")

    st.write("---")

    if answer.strip() != "I don't know based on the documents":
        st.write("Sources:")
        for s in sources:
            name = s.metadata.get("source", "unknown file")
            page = s.metadata.get("page", "?")
            st.write(f"- {name} (page {page})")
    else:
        st.write("No relevant sources found.")