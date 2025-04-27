import os
import glob
from dotenv import load_dotenv
from pypdf import PdfReader
from typing import Callable, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Pinecone

# Load environment variables
load_dotenv()

# Get Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")


def load_pdf_file(path: str) -> List[Document]:
    """
    Load PDF files from a directory or a single PDF file and return a list of Documents.
    """
    # Determine PDF paths
    if os.path.isdir(path):
        pdf_paths = glob.glob(os.path.join(path, "*.pdf"))
    elif os.path.isfile(path) and path.lower().endswith('.pdf'):
        pdf_paths = [path]
    else:
        raise ValueError(f"No PDF file or directory found at {path}")

    documents: List[Document] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": i}
                )
            )
    return documents


def text_split(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    """
    Split documents into text chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize and return the Google Generative AI embeddings model.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )


def download_embeddings(
    path: str,
    embedding_fn: Callable,
    index_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 20
) -> None:
    """
    Load PDFs, split into chunks, and upsert embeddings into Pinecone.
    """
    # 1) Load documents
    docs = load_pdf_file(path)

    # 2) Split into chunks
    chunks = text_split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 3) Create Pinecone vector store
    Pinecone.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        index_name=index_name
    )
