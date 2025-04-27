import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from pinecone import Pinecone, ServerlessSpec

from src.helper import load_pdf_file, text_split, get_embedding_model

# Load environment variables
load_dotenv()

# Get API keys and environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not all([GOOGLE_API_KEY, PINECONE_API_KEY]):
    raise ValueError("Missing required API keys. Check your .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

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
vector_store = LangChainPinecone.from_documents(
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
