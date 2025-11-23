import pytest
from unittest.mock import patch, MagicMock
import sys

# We need to clear the import cache before the test
# so that Python reloads src.query.query with our new mocks
@pytest.fixture(autouse=True)
def clean_sys_modules():
    if 'src.query.query' in sys.modules:
        del sys.modules['src.query.query']

def test_ask_function_returns_answer():
    # Tests the ask() function from src/query/query.py.
    
    # Prepare the fake response
    mock_response = {
        "result": "This is a test answer.",
        "source_documents": [
            MagicMock(metadata={"source": "report.pdf", "page": 5})
        ]
    }

    with patch('langchain_community.vectorstores.FAISS'), \
         patch('langchain_openai.OpenAIEmbeddings'), \
         patch('langchain_openai.ChatOpenAI'), \
         patch('langchain.chains.RetrievalQA') as MockQA:

        # Configure the QA chain mock to return our fake response
        # When .invoke() is called on the chain, return mock_response
        qa_instance = MockQA.from_chain_type.return_value
        qa_instance.invoke.return_value = mock_response
        
        # Now we import the module. It will pick up the mocks defined above.
        from src.query.query import ask

        # Run the function
        answer, sources = ask("What is the question?")

        # Assertions
        assert answer == "This is a test answer."
        assert len(sources) == 1
        assert sources[0].metadata["source"] == "report.pdf"