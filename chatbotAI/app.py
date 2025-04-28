import os
from flask import Flask, request, jsonify, send_from_directory
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import pinecone

# Initialize Flask app
app = Flask(__name__)

# Configuration - get environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = PINECONE_INDEX
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name=index_name, dimension=1536, metric="cosine")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Check if the index is empty
index_obj = pinecone.Index(index_name)
stats = index_obj.describe_index_stats()
is_empty = True
if stats and stats["namespaces"]:
    ns_stats = stats["namespaces"].get("", {})
    if ns_stats.get("vector_count", 0) > 0:
        is_empty = False

# Load PDFs and populate Pinecone if needed
if is_empty:
    docs = []
    data_folder = "Data"
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_folder, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            for page in pages:
                split_docs = text_splitter.split_documents([page])
                docs.extend(split_docs)
    if docs:
        vectorstore = Pinecone.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name,
        )
else:
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# Create retriever and QA chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-turbo", google_api_key=GOOGLE_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type="stuff",
    input_key="query",
    output_key="answer"
)

@app.route("/")
def index():
    return send_from_directory(directory='.', filename='index.html')

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    user_input = data.get("message") or data.get("question") or ""
    if not user_input:
        return jsonify({"answer": ""})
    result = qa_chain({"query": user_input})
    answer = result["answer"] if isinstance(result, dict) else result
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
