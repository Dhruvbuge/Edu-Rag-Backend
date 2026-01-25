# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from fastapi.middleware.cors import CORSMiddleware

from utils.rag_utils import generate_answer

# --------------------------------------------------
# Load ENV variables
# --------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_rag_collection")

if not OPENAI_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Missing required environment variables")

# --------------------------------------------------
# Initialize clients (LIGHTWEIGHT)
# --------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0,
)

EMBEDDING_MODEL = "text-embedding-3-small"

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title="Educational RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Request schema
# --------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    image_base64: Optional[str] = None

# --------------------------------------------------
# Embedding helper (OpenAI)
# --------------------------------------------------
def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding

# --------------------------------------------------
# Multimodal Answer
# --------------------------------------------------
def generate_multimodal_answer(question: str, context: str, image_base64: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
You are a helpful educational assistant.

Use the following retrieved context to answer the question.

Context:
{context}

Question:
{question}
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}",
                    },
                ],
            }
        ],
        max_output_tokens=500,
    )

    return response.output_text

# --------------------------------------------------
# API Endpoint
# --------------------------------------------------
@app.post("/query")
def ask_question(req: QueryRequest):
    try:
        context = ""

        if req.question.strip():
            q_emb = embed_text(req.question)

            result = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=q_emb,
                limit=req.top_k,
                with_payload=True,
            )

            hits = result.points
            context = "\n---\n".join(
                [p.payload["text"] for p in hits]
            ) if hits else ""

        if req.image_base64:
            answer = generate_multimodal_answer(
                req.question,
                context,
                req.image_base64,
            )
        else:
            answer = generate_answer(client, context, req.question)

        return {"answer": answer}

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        raise

# --------------------------------------------------
# Health Check (Render requirement)
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}
