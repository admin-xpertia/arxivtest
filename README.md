# Arxiv Research Agent

LangChain agent that searches Arxiv papers and answers questions, powered by Mistral.

## Local Setup

```bash
pip install -r requirements.txt
```

Add your API key in `.streamlit/secrets.toml`:

```toml
MISTRAL_API_KEY = "your-api-key-here"
```

## Run Locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set `app.py` as the main file.
4. In **Settings > Secrets**, add:
   ```toml
   MISTRAL_API_KEY = "your-api-key-here"
   ```
5. Deploy.

## Project Structure

```
arxiv-agent/
├── app.py               # Single entry point (agent + UI)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml     # API keys (local dev, git-ignored)
└── .gitignore
```
