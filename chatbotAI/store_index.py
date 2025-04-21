import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_embeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Get API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not all([GOOGLE_API_KEY, PINECONE_API_KEY]):
    raise ValueError("Missing required API keys. Check your .env file.")

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# Load and process PDFs
def create_vector_store():
    """
    Create a vector store from PDF documents using Pinecone.
    Returns:
        PineconeVectorStore: The created vector store
    """
    # Load PDFs
    extracted_data = load_pdf_file(data='Data/')
    
    # Split into chunks
    text_chunks = text_split(extracted_data)
    
    # Get embeddings model
    embeddings = download_embeddings()
    
    # Create or get index
    index_name = "chatbotai"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # Dimension for Gemini embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    
    # Create vector store
    docsearch = Pinecone.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    return docsearch

if __name__ == "__main__":
    # Create the vector store
    vector_store = create_vector_store()
    print("Vector store created successfully!") 