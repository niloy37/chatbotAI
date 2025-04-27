from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Pinecone
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Get embeddings
embeddings = download_embeddings()

# Get existing Pinecone index
index_name = "chatbotai"

# Initialize vector store
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Create system prompt
system_prompt = (" You are an assistant for question-answering tasks."
                 "Use the following pieces of retrieved context to answer the question."
                 "the question If you don't know the answer, just say that you don't know. Don't make up an answer."
                 "Answer in a concise and friendly manner."
                 "\n\n"
                 "{context}"
                 )

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create chains
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'answer': 'Please provide a question.'})
    
    try:
        # Get response using the RAG chain
        response = rag_chain.invoke({"input": question})
        
        return jsonify({'answer': response["answer"]})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'answer': 'Sorry, I encountered an error. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)
