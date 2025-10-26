import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Locate index folder no matter where you run this script
ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT_DIR / "data" / "indexes"

# Load vectorstore from disk
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Model setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Anti-hallucination prompt
template = """
You are a helpful assistant answering questions **only** using information from the provided text context.
If the answer is not clearly stated in the context, reply exactly with:
"I don't know based on the documents."

Context:
{context}

Question:
{question}

Answer (use neutral tone, and cite the documents when possible):
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

# Build QA chain with sources output enabled
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

def ask(question: str):
    response = qa.invoke(question)
    answer = response["result"]
    sources = response["source_documents"]
    return answer, sources


if __name__ == "__main__":
    print("\n multi-PDF research assistant ready.")
    print("Type your question or type 'exit' to quit.\n")

    while True:
        q = input(" You: ")
        if q.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        answer, sources = ask(q)

        print(f"\n Answer:\n{answer}\n")

        # Only show sources if answer is real
        if answer.strip() != "I don't know based on the documents.":
            print("Sources:")
            for s in sources:
                source_name = s.metadata.get("source", "unknown file")
                page = s.metadata.get("page", "?")
                print(f" - {source_name} (page {page})")
        else:
            print("No sources because the answer was not found in the documents.")

        print("\n" + "-"*60 + "\n")