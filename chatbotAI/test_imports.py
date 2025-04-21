from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import pinecone

# Load environment variables
load_dotenv()

# Get the API key from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Debugging: Print the API key (first few characters for safety)
if PINECONE_API_KEY:
    print(f"API Key found (first 4 chars): {PINECONE_API_KEY[:4]}...")
else:
    raise ValueError("PINECONE_API_KEY not found. Check your .env file.")

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

print("Pinecone initialized successfully!")

# Test the imports
print("All imports successful!")
print(f"PINECONE_API_KEY exists: {'PINECONE_API_KEY' in os.environ}") 