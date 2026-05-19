from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
import uuid
import os

# ── SETUP ──────────────────────────────────────────────────────

app = FastAPI()

# This allows your frontend webpage to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your API keys
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
TAVILY_API_KEY = "your-tavily-api-key-here"

# Initialize all clients
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

# Initialize embedding model
print("Loading embedding model... please wait...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")


# ── DATA MODELS ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str

class QuestionRequest(BaseModel):
    question: str
    collection_name: str


# ── ROUTE 1: HEALTH CHECK ────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "Research Assistant API is running"}


# ── ROUTE 2: RESEARCH A TOPIC ───────────────────────────────────

@app.post("/research")
async def research_topic(request: ResearchRequest):
    topic = request.topic
    print(f"\nResearching topic: {topic}")

    # Step 1 — Search the web using Tavily
    print("Searching the web...")
    search_results = tavily.search(
        query=topic,
        search_depth="advanced",
        max_results=5
    )

    # Step 2 — Extract content from search results
    documents = []
    sources = []

    for result in search_results["results"]:
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")

        if content:
            full_text = f"Title: {title}\nSource: {url}\n\n{content}"
            documents.append(full_text)
            sources.append({"title": title, "url": url})

    print(f"Found {len(documents)} sources")

    # Step 3 — Create a unique collection for this session
    collection_name = f"research_{uuid.uuid4().hex[:8]}"
    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    # Step 4 — Embed and store all documents
    print("Storing content in vector database...")
    embeddings = embedding_model.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    print("Content stored successfully!")

    # Step 5 — Generate a brief summary
    combined_content = "\n\n".join([
        f"Source {i+1}: {doc[:500]}"
        for i, doc in enumerate(documents)
    ])

    summary_response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": f"""Based on these search results about '{topic}',
            write a 3 sentence summary of what was found:

            {combined_content}"""
        }]
    )

    summary = summary_response.content[0].text

    return {
        "collection_name": collection_name,
        "sources": sources,
        "summary": summary,
        "total_sources": len(documents)
    }


# ── ROUTE 3: ASK A QUESTION ─────────────────────────────────────

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    collection_name = request.collection_name

    print(f"\nQuestion: {question}")

    # Step 1 — Get the collection for this session
    collection = chroma_client.get_collection(name=collection_name)

    # Step 2 — Search for relevant chunks
    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_chunks = results["documents"][0]

    # Step 3 — Generate answer using Claude
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a research assistant. Answer the question
based only on the provided research content below.
Always mention which source the information came from.
If the answer is not in the content say so clearly.

Research Content:
{context}

Question: {question}

Provide a clear detailed answer with source references."""

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.content[0].text

    return {
        "answer": answer,
        "chunks_used": len(retrieved_chunks)
    }