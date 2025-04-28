# app.py

import os
from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import pinecone

# Load environment variables (make sure you set these in Render settings too)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY)

# Connect to Pinecone index
index_name = "your-index-name"  # <-- Replace with your real index name
vectorstore = PineconeVectorStore(index_name=index_name, embedding=OpenAIEmbeddings())

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Initialize Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Basic Response Flow
    try:
        # You can add retrieval-augmented generation here with your vectorstore
        response = llm.invoke(user_input)
        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
