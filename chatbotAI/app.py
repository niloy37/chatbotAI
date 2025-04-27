import os
import sys
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI
from src.helper import get_embedding_model
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Get API keys and environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west1-gcp"
if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
    raise ValueError("Missing required environment variables. Check your .env file.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name
INDEX_NAME = "chatbotai"

# (Optional) ensure index exists; comment out if index already created
# if INDEX_NAME not in [idx.name for idx in pc.list_indexes().names()]:
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region=PINECONE_ENV
#         )
#     )

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    convert_system_message_to_human=True
)

# Initialize embeddings model
embeddings = get_embedding_model()

# Connect to existing Pinecone index in LangChain vector store
vector_store = LangChainPinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# Build retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Define system prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Don't make up an answer. "
    "Answer in a concise and friendly manner.\n\n"
    "Context: {context}\n"
    "Question: {query}\n"
    "Answer: "
)
prompt = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "query"],
)

# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    input_key="query",
    output_key="result",
    chain_type_kwargs={"prompt": prompt},
)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'answer': 'Please provide a question.'})
    try:
        result = qa_chain.invoke({"query": question})
        answer = result.get('result', '')
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /ask: {e}", file=sys.stderr)
        if app.debug:
            return jsonify({'answer': 'Sorry, I encountered an error.', 'error': str(e)})
        return jsonify({'answer': 'Sorry, I encountered an error. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)
