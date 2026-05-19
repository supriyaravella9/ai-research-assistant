# AI-Powered Research Assistant

A full stack AI web application that searches the web on any topic
and answers questions about what it finds using RAG and Claude API.

## What It Does
- Search any topic on the web using Tavily Search API
- Extracts and stores content in ChromaDB vector database
- Ask questions in plain English about the research
- Get accurate answers with source citations using Claude API

## Tech Stack
- Python
- FastAPI (backend)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- Anthropic Claude API
- Tavily Search API
- HTML and JavaScript (frontend)

## How to Run

### 1. Install dependencies
pip install fastapi uvicorn tavily-python chromadb sentence-transformers anthropic

### 2. Add your API keys in backend/main.py
ANTHROPIC_API_KEY = "your-key-here"
TAVILY_API_KEY = "your-key-here"

### 3. Start the backend server
cd backend
uvicorn main:app --reload

### 4. Open the frontend
Open frontend/index.html in your browser

## How It Works
1. User enters a research topic
2. Tavily searches the web and returns top 5 results
3. Content is chunked, embedded and stored in ChromaDB
4. User asks questions about the research
5. Relevant chunks are retrieved using semantic search
6. Claude generates accurate answers with source references# ai-research-assistant
 AI web app that searches the web and answers questions using RAG and Claude API
