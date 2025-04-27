import os
from dotenv import load_dotenv
from langchain import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Get Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

def load_pdf_file(data):
    """
    Load PDF files from a directory.
    Args:
        data (str): Path to the directory containing PDF files
    Returns:
        list: List of loaded documents
    """
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """
    Split documents into text chunks.
    Args:
        extracted_data (list): List of documents to split
    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_embeddings():
    """
    Initialize and return the Google Generative AI embeddings model.
    Returns:
        GoogleGenerativeAIEmbeddings: The embeddings model
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return embeddings
