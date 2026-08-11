# PDFQuery — RAG Assistant

A FastAPI backend for uploading PDF documents and querying them in natural language. Uses Retrieval-Augmented Generation (RAG): documents are chunked, embedded, and stored in ChromaDB; queries retrieve the most relevant chunks and pass them as context to Qwen2.5 0.5B running via Ollama.

---

## Tech Stack

| Layer | Tool |
|---|---|
| API | FastAPI |
| LLM | Qwen2.5 0.5B via Ollama |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF Parsing | pypdf |
| Container | Docker |

---

## Architecture

```
Upload:  PDF → Parse → Chunk → Embed → ChromaDB
Query:   Question → Embed → ChromaDB similarity search → Context → Qwen2.5 0.5B → Answer
```

---

## Project Structure

```
app/
  main.py              # FastAPI app, registers routers
  routes/
    upload.py          # POST /upload — PDF ingestion pipeline
    query.py           # POST /query — question answering
  services/
    rag.py             # Orchestrates retrieve → prompt → generate
    embed.py           # sentence-transformers wrapper
    llm.py             # Ollama HTTP API calls
  utils/
    parser.py          # PDF text extraction
    chunking.py        # 500-char chunks with 100-char overlap
  db/
    chroma.py          # ChromaDB client — add and query documents
docker-compose.yml
requirements.txt
```

---

## Getting Started

### 1. Start Ollama

```bash
docker compose up -d
```

### 2. Pull the Qwen2.5 0.5B model (first time only)

```bash
docker exec -it <container_id> ollama pull qwen2.5:0.5b
```

> A small model is used here since it runs reliably on machines with limited RAM (~8GB). Swap for a larger model like `llama3` in [app/services/llm.py](app/services/llm.py) if your machine has more headroom (llama3:8B needs ~5-6GB free RAM/VRAM just to load).

Get the container ID from `docker ps`.

### 3. Configure environment

Create a `.env` file in the project root:

```
OLLAMA_URL=http://localhost:11434
```

> Change this to `http://ollama:11434` if the app is containerised and on the same Docker network as Ollama.

### 4. Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Upload a PDF (`multipart/form-data`, field: `file`) |
| POST | `/query` | Ask a question (`{"query": "..."}`) — returns `answer` and `sources` |

---

## Features

- PDF document ingestion and semantic indexing
- Semantic search using vector embeddings
- Context-aware answers grounded in the uploaded document
- Fully local — no external APIs or cloud services

---

## Future Improvements

- Chat UI
- Streaming responses
- Multi-document support
- Persistent document management
