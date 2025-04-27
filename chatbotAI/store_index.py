```python
# store_index.py
import os
import glob
from dotenv import load_dotenv
import pinecone
from typing import Optional

from src.helper import load_pdf_file, text_split, get_embedding_model
from langchain.vectorstores import Pinecone

# Load environment variables
load_dotenv()

# Get API keys and environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west1-gcp"

if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
    raise ValueError("Missing required environment variables. Check your .env file.")

# Initialize Pinecone client
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)


def create_vector_store(
    data_path: str = "Data/",
    index_name: str = "chatbotai",
    dimension: int = 768,
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

    # 4) Create index if it doesn't exist
    existing_indexes = pinecone.list_indexes()
    if index_name not in existing_indexes:
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric
        )

    # 5) Build and return the vector store
    store = Pinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    return store


if __name__ == "__main__":
    vs = create_vector_store()
    print(f"Vector store '{vs.index_name}' created successfully with {len(vs._index.describe_index_stats()['namespaces'])} namespaces.")
```
