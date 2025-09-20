"""
Medical Chatbot with RAG (Retrieval-Augmented Generation)

A Flask-based web application that provides intelligent medical Q&A using:
- OpenAI's GPT-3.5-turbo for natural language generation
- Pinecone vector database for document retrieval
- LangChain for orchestrating the RAG pipeline

Author: [Your Name]
Created: 2025
"""

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()

# Get API keys and environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
    raise ValueError("Missing required environment variables. Check your .env file.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize OpenAI LLM (GPT-3.5-turbo is the cheapest model)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.3
)

# Initialize embeddings model
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

# Connect to existing Pinecone index
INDEX_NAME = "chatbotai"
vector_store = LangchainPinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# Build retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Custom system prompt
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Don't make up an answer.
Answer in a concise and friendly manner.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create the chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'answer': 'Please provide a question.'})
    try:
        response = chain.invoke(question)
        return jsonify({'answer': response.content})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'answer': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
