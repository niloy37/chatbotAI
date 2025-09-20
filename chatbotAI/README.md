# Medical Chatbot with RAG (Retrieval-Augmented Generation)

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0+-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-orange.svg)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector--DB-purple.svg)

A sophisticated AI-powered medical chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses based on medical documents.

## 🎯 **What This Project Does**

1. **Document Processing**: Converts medical PDF documents into vector embeddings using OpenAI's embedding model
2. **Vector Storage**: Stores document vectors in Pinecone vector database for efficient similarity search
3. **Intelligent Retrieval**: When users ask questions, retrieves the most relevant document chunks
4. **Context-Aware Generation**: Sends retrieved context to OpenAI's GPT-3.5-turbo for accurate, grounded responses
5. **Interactive Interface**: Provides a seamless chatbot experience via Flask web application

## 🛠️ **Technology Stack**

- **Backend**: Python, Flask
- **LLM Integration**: LangChain, OpenAI API (GPT-3.5-turbo)
- **Vector Database**: Pinecone
- **Document Processing**: PyPDF, LangChain Text Splitters
- **Environment Management**: python-dotenv, conda

## 🚀 **Getting Started**

### Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone API key
- Conda (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd medical-chatbot-rag
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n chatbotAI python=3.10
   conda activate chatbotAI
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENV=your_pinecone_environment_here
   ```

5. **Initialize vector database**
   ```bash
   python store_index.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the chatbot**
   Open your browser and go to `http://localhost:5000`

## 📁 **Project Structure**

```
├── app.py                 # Main Flask application
├── gemini_chatbot.py     # Chatbot logic (legacy name, now uses OpenAI)
├── store_index.py        # Vector database initialization
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── src/
│   ├── helper.py        # Utility functions for document processing
│   └── prompt.py        # Custom prompts
├── templates/
│   └── index.html       # Frontend interface
├── static/
│   └── style.css        # Styling
└── Data/
    └── *.pdf           # Medical documents (add your PDFs here)
```

## 🔧 **Configuration**

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key ([Get it here](https://platform.openai.com/api-keys))
- `PINECONE_API_KEY`: Your Pinecone API key ([Get it here](https://www.pinecone.io/))
- `PINECONE_ENV`: Your Pinecone environment (e.g., "us-east-1-aws")

### Model Configuration

- **LLM**: GPT-3.5-turbo (cost-optimized for high performance)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Dimension**: 1536 (OpenAI standard)
- **Similarity Metric**: Cosine similarity

## 🔮 **Future Enhancements**

### CacheBlend Implementation

Planning to implement **CacheBlend**, an advanced optimization technique that will significantly improve response times:

- **KV Cache Reusing**: Reuse precomputed key-value caches for text chunks
- **Selective Recomputation**: Target only ~15% of tokens that matter most for cross-chunk interactions
- **Pipelined I/O**: Overlap computation with cache fetching for zero-latency storage access

This will enable faster responses and support for longer conversations while reducing computational costs.

## 🔒 **Security & Best Practices**

- ✅ API keys stored in environment variables (never committed to Git)
- ✅ Comprehensive `.gitignore` for sensitive files
- ✅ Error handling and input validation
- ✅ Modular code structure for maintainability
- ✅ Type hints and documentation

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 **Contact**

- **GitHub**: [your-github-username]
- **LinkedIn**: [your-linkedin-profile]
- **Email**: [your-email]

---

⭐ **Star this repository if it helped you!**

## Security Note

Never commit your `.env` file to version control. The `.env.example` file is provided as a template.

```bash
Project repo:
```
### STEP 01- Create a conda evironment after opening the repsoitory

```bash
conda create -n chatbotAI python=3.10 -y
```

```bash
conda activate llmapp
```
### Step 02- install the requirements

```bash
pip install -r requirements.txt
```


