import os
from dotenv import load_dotenv
import pinecone

from langchain_google_genai import ChatGoogleGenerativeAI
from src.helper import load_pdf_file, text_split, get_embedding_model
from langchain_community.vectorstores import Pinecone

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

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True
)

# Initialize embeddings model
embeddings = get_embedding_model()

# Index name
INDEX_NAME = "chatbotai"

# Create Pinecone index if it doesn't exist
existing_indexes = pinecone.list_indexes()
if INDEX_NAME not in existing_indexes:
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine"
    )

# Load, split, and index documents
docs = load_pdf_file("Data/")
chunks = text_split(
    documents=docs,
    chunk_size=500,
    chunk_overlap=20
)
vector_store = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

# Example usage: simple retrieval + generation
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=False
)

def ask_question(question: str) -> str:
    """
    Ask a question using the RetrievalQA chain.
    """
    result = qa_chain.run(question)
    return result

if __name__ == "__main__":
    query = "What is the capital of France?"
    answer = ask_question(query)
    print(f"Q: {query}")
    print(f"A: {answer}")
