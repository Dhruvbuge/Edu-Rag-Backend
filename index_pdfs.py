# index_pdfs.py

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from utils.pdf_utils import extract_text_from_folder
from utils.rag_utils import legal_chunking
from utils.qdrant_utils import setup_qdrant, upsert_chunks

# --- STEP 1: Load Environment Variables ---
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_rag_collection")
PDF_FOLDER = os.getenv("PDF_FOLDER", "data")  # folder path containing multiple PDFs

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("❌ Missing QDRANT_URL or QDRANT_API_KEY in .env file")

# --- STEP 2: Initialize Embedder ---
print("🚀 Initializing embedding model...")
embedder = SentenceTransformer("BAAI/bge-large-en")

# --- STEP 3: Extract Text from PDFs ---
print(f"📂 Extracting text from all PDFs in: {PDF_FOLDER}")
pdf_texts = extract_text_from_folder(PDF_FOLDER)

if not pdf_texts:
    raise ValueError(f"❌ No PDFs found or no text extracted from folder: {PDF_FOLDER}")

combined_text = "\n".join(pdf_texts.values())

# --- STEP 4: Chunk the Text ---
chunks = legal_chunking(combined_text)
print(f"🧩 Created {len(chunks)} chunks.")

# --- STEP 5: Embed and Store in Qdrant ---
print("🔢 Generating embeddings...")
embeddings = embedder.encode(chunks, show_progress_bar=True)
vector_size = embedder.get_sentence_embedding_dimension()

print("💾 Setting up Qdrant collection...")
qdrant_client = setup_qdrant(COLLECTION_NAME, vector_size, QDRANT_API_KEY, QDRANT_URL)

upsert_chunks(qdrant_client, COLLECTION_NAME, chunks, embeddings)
print(f"✅ Stored {len(chunks)} chunks in Qdrant!")
print("🎉 Indexing complete. You can now run main.py to ask questions.")
