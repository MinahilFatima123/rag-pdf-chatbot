# RAG PDF Chatbot

A chatbot that answers questions about any PDF you upload. Built with LangChain, ChromaDB, Groq, and FastAPI. Monitored with LangSmith and deployed on Railway.

Live API: https://rag-pdf-chatbot-production-e417.up.railway.app/docs

## How it works

1. Upload any PDF via the '/upload' endpoint
2. Ask questions about it via the '/ask' endpoint
3. The chatbot finds the most relevant parts of your PDF and answers using Llama 3

## How to run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a .env file:

```
GROQ_API_KEY=your_key
HUGGINGFACEHUB_API_TOKEN=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=rag-pdf-chatbot
```

Run the API:

```bash
python -m uvicorn api:app --reload
```

Open `http://localhost:8000/docs` to test it.

## API

**POST /upload** — upload your PDF

**POST /ask** — ask a question about it

```json
{ "question": "What is market segmentation?" }
```

```json
{ "question": "What is market segmentation?", "answer": "..." }
```

## Stack

LangChain, ChromaDB, Groq (Llama 3), HuggingFace Embeddings, FastAPI, LangSmith, Docker, Railway
