"""
Vector Database Initialization Script

This script loads PDF documents, processes them into vector embeddings,
and stores them in Pinecone vector database for retrieval.

Usage:
    python store_index.py

Author: [Your Name]
Created: 2025
"""

import os
import glob
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import Optional

from src.helper import load_pdf_file, text_split, get_embedding_model
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv()

# Get API keys and environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west1-gcp"

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
    raise ValueError("Missing required environment variables. Check your .env file.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


def create_vector_store(
    data_path: str = "Data/",
    index_name: str = "chatbotai",
    dimension: int = 1536,  # OpenAI embeddings dimension
    metric: str = "cosine",
    chunk_size: int = 500,
    chunk_overlap: int = 20
) -> Pinecone:
    """
    Load PDFs, split into chunks, initialize index if needed, and upsert to Pinecone.
    Returns:
        Pinecone: The LangChain Pinecone vector store instance.
    """
    # 1) Load and parse documents
    docs = load_pdf_file(data_path)

    # 2) Split into text chunks
    chunks = text_split(
        documents=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 3) Get embedding model
    embeddings = get_embedding_model()

    # 4) Since we already created the index, we don't need to check/create it again
    # The index already exists from our previous script

    # 5) Build and return the vector store
    store = LangchainPinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    return store


if __name__ == "__main__":
    vs = create_vector_store()
    print(f"Vector store created successfully!")
