import pytest
from unittest.mock import patch
from langchain_core.documents import Document

@pytest.fixture
def mock_documents():
    # Return the example list of documents
    return [
        Document(page_content="Tekst strony 1", metadata={"source": "test1.pdf", "page": 1}),
        Document(page_content="Tekst strony 2", metadata={"source": "test1.pdf", "page": 2})
    ]

@pytest.fixture(autouse=True)
def mock_env_dependencies():
    # Block connection to OPENAI and vectorbase
    with patch('langchain_openai.OpenAIEmbeddings'), \
         patch('langchain_openai.ChatOpenAI'), \
         patch('langchain_community.vectorstores.FAISS'):
        yield