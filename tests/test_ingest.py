import pytest
from unittest.mock import patch, MagicMock

# Import functions from the file: src/ingest/ingest_pdfs.py
from src.ingest.ingest_pdfs import load_and_split_pdfs, build_vectorstore

def test_load_and_split_pdfs_logic(mock_documents):
    # Mock os.listdir, so it pretends to see PDF
    with patch('os.listdir', return_value=['dokument.pdf']):
        # Mock PyPDFLoader
        with patch('src.ingest.ingest_pdfs.PyPDFLoader') as MockLoader:
            instance = MockLoader.return_value
            instance.load.return_value = mock_documents

            chunks = load_and_split_pdfs()

            assert len(chunks) > 0
            # See if RecursiveCharacterTextSplitter worked (returned document objects)
            assert hasattr(chunks[0], 'page_content')

def test_build_vectorstore_calls_save():
    mock_chunks = [MagicMock()]
    
    # Because of ingest_pdfs.py import FAISS and OpenAIEmbeddings,
    # we must mock them here
    with patch('src.ingest.ingest_pdfs.FAISS') as MockFAISS, \
         patch('src.ingest.ingest_pdfs.OpenAIEmbeddings'):
        
        build_vectorstore(mock_chunks)
        
        # Check if save_local was run
        MockFAISS.from_documents.return_value.save_local.assert_called_once()