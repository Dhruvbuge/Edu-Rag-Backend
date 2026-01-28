🎓 Edu-RAG Backend

A FastAPI-based Retrieval-Augmented Generation (RAG) backend designed for educational use cases.
This service retrieves relevant context from a vector database (Qdrant) and generates AI-powered answers using OpenAI models.

🚀 Live API

🔗 Backend URL (Render):
https://edu-rag-backend.onrender.com

📌 Features

🔍 Semantic search using vector embeddings

🧠 Retrieval-Augmented Generation (RAG)

📄 PDF-based knowledge ingestion

⚡ FastAPI REST API

☁️ Cloud-hosted on Render

🔐 Secure API keys via environment variables

🌐 CORS-enabled for frontend integration

🧱 Tech Stack
Backend

Python 3.10+

FastAPI

OpenAI API (Embeddings + Generation)

Qdrant Cloud (Vector Database)

Uvicorn (ASGI Server)

ML / NLP

Sentence Transformers

OpenAI Embeddings

Chunk-based document indexing

🗂️ Project Structure
.
├── main.py                 # FastAPI app & /query endpoint
├── index_pdfs.py           # PDF ingestion & indexing script
├── requirements.txt        # Python dependencies
├── utils/
│   ├── pdf_utils.py        # PDF text extraction
│   ├── rag_utils.py        # Chunking & answer generation
│   └── qdrant_utils.py     # Qdrant setup & upserts
├── .gitignore
└── README.md

🔄 How the RAG Pipeline Works
1️⃣ Indexing (Offline Step)

PDFs are read from a folder

Text is chunked into smaller sections

Each chunk is converted into embeddings

Embeddings are stored in Qdrant

2️⃣ Querying (Runtime)

User sends a question to /query

Question is embedded

Qdrant retrieves top-K relevant chunks

Retrieved context is passed to an AI model

Final answer is returned to the client

📡 API Endpoints
Health Check
GET /


Response

{
  "status": "ok"
}

Ask a Question (RAG)
POST /query


Request Body

{
  "question": "What is machine learning?",
  "top_k": 5
}


Response

{
  "answer": "Machine learning is a subset of artificial intelligence..."
}

⚙️ Environment Variables

Create these in Render or a .env file locally:

OPENAI_API_KEY=your_openai_key
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key
COLLECTION_NAME=chatbot_with_qdrant
PDF_FOLDER=data


⚠️ Never commit API keys to GitHub

🛠️ Running Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Index PDFs (one-time step)
python index_pdfs.py

3️⃣ Start the server
uvicorn main:app --reload


Server will run at:

http://127.0.0.1:8000

🌍 Deployment

Platform: Render

Start Command

uvicorn main:app --host 0.0.0.0 --port $PORT


Auto-deploy: Enabled via GitHub

Free tier: Spins down after inactivity (cold start delay expected)

🌐 CORS Configuration

Backend allows requests from:

http://localhost:3000

https://eduragai.netlify.app

This enables secure frontend–backend communication.

⚠️ Notes & Limitations

Free OpenAI tier may hit rate limits

Cold starts on Render may cause first request delay

AI responses may be inaccurate — verify critical info

🧠 Use Cases

Educational chatbots

AI tutoring systems

Legal / academic document Q&A

RAG experimentation projects

👨‍💻 Author

Dhruv Buge
Computer Science (AI & ML) Undergraduate
GitHub: https://github.com/Dhruvbuge

📄 License

This project is intended for educational and learning purposes.
