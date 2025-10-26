import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT_DIR / "data" / "indexes"

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)
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

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

def answer_question(q):
    response = qa.invoke(q)
    answer = response["result"]
    sources = response["source_documents"]
    return answer, sources

st.set_page_config(page_title="Multi-PDF Research Assistant", page_icon="📚")
st.title("Document Research Assistant")

query = st.text_input("Ask a question:")

if query:
    answer, sources = answer_question(query)
    st.write("Answer:")
    st.write(answer)

    if answer.strip() != "I don't know based on the documents.":
        st.write("Sources:")
        for s in sources:
            source_name = s.metadata.get("source", "unknown file")
            page = s.metadata.get("page", "?")
            st.write(f"- **{source_name}** (page {page})")
    else:
        st.write("No sources found in the documents.")