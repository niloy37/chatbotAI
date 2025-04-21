# Medical Chatbot

A Flask-based medical chatbot using LangChain, Google's Gemini, and Pinecone for vector storage.

## Setup Instructions

1. Clone the repository:
```bash
git clone <your-repo-url>
cd chatbotAI
```

2. Create a virtual environment:
```bash
conda create -n chatbotAI python=3.10
conda activate chatbotAI
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` and add your actual API keys:
     - Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
     - Get a Pinecone API key from [Pinecone](https://www.pinecone.io/)

5. Run the application:
```bash
python app.py
```

## Environment Variables

- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Name of your Pinecone index (default: "chatbotai")

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


