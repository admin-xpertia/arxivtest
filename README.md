# Arxiv Research Agent

LangChain agent that searches Arxiv papers and answers questions, powered by Mistral.

## Setup

```bash
pip install -r requirements.txt
export MISTRAL_API_KEY="your-api-key-here"
```

## Run

**Backend** (from project root):
```bash
uvicorn main:app --reload --port 8000
```

**Frontend** (in a second terminal, from project root):
```bash
streamlit run frontend/app.py
```

The frontend opens at `http://localhost:8501` and connects to the backend at `http://localhost:8000`.

## Project Structure

```
arxiv-agent/
├── domain/          # Pydantic models (input/output)
├── application/     # LangChain agent logic
├── infrastructure/  # Mistral LLM + Arxiv tool config
├── api/             # FastAPI routes (SSE streaming)
├── frontend/        # Streamlit chat UI
├── main.py          # FastAPI app entrypoint
└── requirements.txt
```
