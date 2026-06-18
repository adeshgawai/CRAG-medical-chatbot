# 🩺 MediAssist — Corrective RAG Medical Chatbot

A medical question-answering chatbot built on **Corrective RAG (CRAG)** — an agentic retrieval pipeline that evaluates retrieved documents, rewrites queries, and falls back to live web search when local knowledge is insufficient. Grounded in **The Merck Manual of Diagnosis and Therapy** (19th Ed.) via a domain-specific FAISS index.

> Live Demo: [Hugging Face Spaces](#) <!-- replace with your HF link -->

---

## 🧠 Why CRAG, Not Naive RAG?

Standard RAG blindly injects retrieved chunks into the prompt regardless of relevance — leading to hallucinations when the retrieval is off. **CRAG adds a correction loop**:

1. Retrieve from local FAISS index
2. **Evaluate each chunk** using an LLM-powered listwise scorer (Pydantic structured output)
3. Route based on verdict:
   - `CORRECT` → refine context → generate answer
   - `INCORRECT` → rewrite query → Tavily web search → refine → generate
   - `AMBIGUOUS` → merge local + web docs → refine → generate
4. **Context refinement** strips noise (disclaimers, HTML, irrelevant sentences) before generation

---

## 🏗️ Architecture

```
User Question
      │
      ▼
 ┌─────────────┐
 │ Route Query │  ──── general_chat ──────────────────────────┐
 └─────────────┘                                               │
      │ medical_rag                                            │
      ▼                                                        │
 ┌──────────┐     ┌───────────────┐                           │
 │ Retrieve │────▶│ Eval Each Doc │                           │
 └──────────┘     └───────────────┘                           │
                        │                                      │
           ┌────────────┼────────────┐                        │
        CORRECT      INCORRECT    AMBIGUOUS                    │
           │             │             │                       │
           │      ┌──────▼──────┐      │                      │
           │      │Rewrite Query│      │                      │
           │      └──────┬──────┘      │                      │
           │      ┌──────▼──────┐      │                      │
           │      │ Web Search  │      │                      │
           │      └──────┬──────┘      │                      │
           └──────────▶  ▼  ◀──────────┘                      │
                   ┌────────┐                                  │
                   │ Refine │                                  │
                   └────┬───┘                                  │
                        │                                      │
                        ▼                                      │
                  ┌──────────┐  ◀────────────────────────────┘
                  │ Generate │
                  └──────────┘
                        │
                     Answer
```

**Built with LangGraph's `StateGraph`** — each box is a node, each arrow is a conditional or direct edge.

---

## ✨ Features

- **CRAG pipeline** with LangGraph — retrieve, evaluate, correct, generate
- **Listwise document scoring** using Pydantic structured output (`AllDocsEval`) — scores all retrieved chunks in one LLM call
- **Query rewriting** for optimized web fallback search
- **Tavily web search** integration for real-time fallback
- **Context refinement node** — strips irrelevant sentences before generation
- **PubMedBERT embeddings** (`NeuML/pubmedbert-base-embeddings`) — domain-tuned for biomedical text
- **Knowledge base**: The Merck Manual of Diagnosis and Therapy (FAISS index, 1000-token chunks, 200 overlap)
- **User authentication** — Flask-Login + bcrypt (scrypt) password hashing
- **Dual database support** — SQLite (local dev) and Supabase PostgreSQL (production)
- **Dockerized** and deployed on Hugging Face Spaces

---

## 📂 Project Structure

```
hf_deployment/
│
├── src/
│   ├── graph.py        # LangGraph StateGraph definition
│   ├── nodes.py        # All graph nodes (retrieve, eval, refine, generate, route, web_search)
│   ├── state.py        # TypedDict state schema
│   ├── retriever.py    # Lazy-loaded FAISS retriever singleton
│   ├── llm.py          # Groq LLM loader
│   ├── prompt.py       # Prompt templates
│   ├── helper.py       # PDF loader, text splitter, embedding model
│   └── models.py       # SQLAlchemy User model
│
├── faiss_index/        # Pre-built FAISS index (Merck Manual)
├── templates/          # Flask HTML templates (landing, chat)
├── static/             # CSS stylesheets
├── app.py              # Flask app — routes, auth, chat endpoint
├── Dockerfile          # Docker config for HF Spaces (port 7860)
└── pyproject.toml      # Dependencies (uv)
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))
- Tavily API key (free tier at [app.tavily.com](https://app.tavily.com))

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/crag-based-chatbot.git
cd crag-based-chatbot/hf_deployment

uv venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # Linux/Mac

uv pip install -r pyproject.toml
```

### 2. Configure environment variables

Create a `.env` file in the root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
SECRET_KEY=your_flask_secret_key

# Optional: Supabase PostgreSQL (defaults to SQLite if not set)
DATABASE_URL=postgresql://user:password@host:port/dbname
```

### 3. Build the FAISS index (if not included)

If you don't have the pre-built `faiss_index/` folder, place your PDF(s) in a `data/` directory and run:

```bash
python build_index.py
```

> The current index is built from **The Merck Manual of Diagnosis and Therapy (19th Ed.)** using `NeuML/pubmedbert-base-embeddings` with 1000-token chunks and 200-token overlap.

### 4. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

---

## 🐳 Docker

```bash
docker build -t mediassist .
docker run -p 7860:7860 --env-file .env mediassist
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (`StateGraph`) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | `NeuML/pubmedbert-base-embeddings` |
| Vector Store | FAISS (local) |
| Web Search | Tavily |
| Structured Output | Pydantic + `with_structured_output` |
| Backend | Flask, Flask-Login, Flask-SQLAlchemy |
| Database | SQLite / Supabase PostgreSQL |
| Deployment | Docker, Hugging Face Spaces |
| Dependency Mgmt | `uv` |

---

## 🔑 Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq inference API key |
| `TAVILY_API_KEY` | ✅ | Tavily web search API key |
| `SECRET_KEY` | ✅ | Flask session secret key |
| `DATABASE_URL` | ❌ | PostgreSQL URL (defaults to SQLite) |

---

## 📌 Notes

- The FAISS index (`faiss_index/`) is committed to this repo for convenience on HF Spaces. For large-scale deployments, host it separately.
- The chatbot auto-routes greetings and general conversation to the LLM directly, skipping retrieval.
- Medical responses always include a disclaimer to consult a licensed physician for personal health decisions.

---

## 📄 License

MIT
