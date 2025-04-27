import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

# Load environment variables
load_dotenv()

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Load and process PDFs
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbotai"

# Create or connect to Pinecone index
if index_name not in pc.list_indexes().names:
    pc.create_index(
        name=index_name,
        dimension=768,  # Gemini embeddings are 768-dimensional
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Example usage
def ask_question(question):
    # Generate response using Gemini
    response = llm.invoke(question)
    return response.content

def format_response(response_dict):
    """
    Format the response dictionary to show only the answer.
    Args:
        response_dict (dict): Dictionary containing the response data
    Returns:
        str: The answer only
    """
    # If response is already a string, return it directly
    if isinstance(response_dict, str):
        return response_dict

    # If there's an answer field, return just that
    if 'answer' in response_dict:
        return response_dict['answer']
    
    # If there's only page content, return that
    if 'page_content' in response_dict:
        return response_dict['page_content']
    
    return "No answer found in the response"

# Test the chatbot
if __name__ == "__main__":
    question = "What is the capital of France?"
    answer = ask_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}") 
