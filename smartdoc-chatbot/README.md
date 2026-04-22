# SmartDoc Chatbot 📚

A production-ready **Retrieval Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask questions about it — answers are grounded entirely in your document.

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Gemini embedding-001 |
| Vector DB | Qdrant Cloud |
| RAG Framework | LangChain |
| Chat UI | Chainlit |
| REST API | FastAPI |

---

## Project Structure

```
smartdoc-chatbot/
├── app.py              # Chainlit chat UI (main entry point)
├── api.py              # FastAPI REST backend
├── rag/
│   ├── __init__.py
│   ├── loader.py       # PDF loading + RecursiveCharacterTextSplitter
│   ├── embedder.py     # Gemini embeddings → Qdrant storage
│   └── retriever.py    # Semantic search + Gemini LLM answer generation
├── .env.example        # Environment variable template
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build file
└── README.md
```

---

## Quick Start

### Step 1 — Get API Keys

#### A. Google Gemini API Key (Free)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key — looks like `AIzaSy...`

> **Free tier:** 15 requests/minute, 1 million tokens/day — more than enough for development.

#### B. Qdrant Cloud (Free)

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Click **"Sign Up"** (GitHub or email)
3. Click **"Create cluster"** → choose **Free tier** (1 GB, no credit card needed)
4. Wait ~30 seconds for the cluster to start
5. From the cluster dashboard copy:
   - **Cluster URL** — e.g. `https://abc123.us-east4-0.gcp.cloud.qdrant.io:6333`
   - **API Key** — click **"API Keys"** tab → **"Create"** → copy the key

---

### Step 2 — Set Up the Project

```bash
# Clone / navigate to the project folder
cd smartdoc-chatbot

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 3 — Configure Environment Variables

```bash
# Copy the template
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
GOOGLE_API_KEY=AIzaSy...your_key_here
QDRANT_URL=https://your-cluster.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=smartdoc
```

---

### Step 4 — Run the Applications

#### Chat UI (Chainlit)

```bash
chainlit run app.py --watch
```

Opens at **http://localhost:8000** — you'll see the chat interface in your browser.

#### FastAPI Backend (separate terminal)

```bash
uvicorn api:app --reload --port 8000
```

Opens at:
- **http://localhost:8000/docs** — interactive Swagger UI
- **http://localhost:8000/redoc** — ReDoc documentation

> Run both simultaneously by opening two terminal windows.

---

## API Reference

### `GET /status`

Health check and collection stats.

```bash
curl http://localhost:8000/status
```

```json
{
  "status": "healthy",
  "collection": {
    "exists": true,
    "collection_name": "smartdoc",
    "vectors_count": 142,
    "points_count": 142
  },
  "all_collections": ["smartdoc"]
}
```

---

### `POST /ingest`

Upload and process a PDF.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/your/document.pdf"
```

```json
{
  "status": "success",
  "doc_id": "my_document",
  "filename": "my_document.pdf",
  "chunks_processed": 87,
  "collection": "smartdoc"
}
```

---

### `POST /chat`

Ask a question (streaming by default).

```bash
# Streaming (default)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "doc_id": "my_document"}'

# Non-streaming (JSON response)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the key findings", "doc_id": "my_document", "stream": false}'
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | The question to answer |
| `doc_id` | string | null | Scope to a specific document |
| `stream` | bool | true | Stream tokens or return JSON |
| `top_k` | int | 5 | Number of chunks to retrieve (1–20) |

---

## Docker Deployment

```bash
# Build the image
docker build -t smartdoc-chatbot .

# Run Chainlit UI
docker run -p 8080:8080 --env-file .env smartdoc-chatbot

# Run FastAPI backend
docker run -p 8000:8000 --env-file .env smartdoc-chatbot \
  uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## How It Works

```
User uploads PDF
      │
      ▼
┌─────────────────────┐
│   loader.py         │  PyPDFLoader → RecursiveCharacterTextSplitter
│   chunk_size=1000   │  → List[Document] (with page metadata)
│   overlap=200       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   embedder.py       │  Gemini embedding-001 (768-dim vectors)
│                     │  → Qdrant Cloud (cosine similarity)
└─────────┬───────────┘
          │
User asks question
          │
          ▼
┌─────────────────────┐
│   retriever.py      │  Query → embed → top-5 Qdrant chunks
│                     │  → Gemini 2.5 Flash → streamed answer
└─────────────────────┘
```

---

## Configuration

| Variable | Description | Example |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini API key | `AIzaSy...` |
| `QDRANT_URL` | Qdrant Cloud cluster URL | `https://abc.qdrant.io:6333` |
| `QDRANT_API_KEY` | Qdrant API key | `eyJ...` |
| `QDRANT_COLLECTION_NAME` | Collection name | `smartdoc` |

---

## Troubleshooting

**`GOOGLE_API_KEY` not set** → Copy `.env.example` to `.env` and add your key.

**Qdrant connection error** → Check your `QDRANT_URL` includes the port (`:6333`) and the cluster is running in Qdrant Cloud dashboard.

**PDF read error** → Make sure the PDF is not password-protected and is a valid PDF file.

**`No relevant information found`** → The document may not contain an answer to your question, or the PDF text extraction failed (scanned/image PDFs are not supported).

**Slow embedding** → Normal for large PDFs. Gemini embedding-001 processes ~50 chunks/min on the free tier.
