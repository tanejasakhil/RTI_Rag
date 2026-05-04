# RTI Document RAG System

A Retrieval-Augmented Generation system for querying PDF documents crawled from `rti.dopt.gov.in`.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenRouter API key
echo "OPENROUTER_API_KEY=your-key-here" > .env

# 3. Index your PDFs (run once — supports incremental re-runs)
python ingest.py

# 4. Launch the app
streamlit run app.py

# 5. (Optional) Run evaluation
python eval.py
```

## Architecture

- **PDF Parsing**: Granite-Docling VLM (258M) — structure-aware, handles tables
- **Embeddings**: nomic-embed-text-v2 (MoE, local, free)
- **Vector DB**: Qdrant (local disk)
- **Reranker**: BAAI/bge-reranker-v2-m3 (local, free)
- **LLM**: Gemma 4 31B via OpenRouter (free tier)
- **Frontend**: Streamlit chat UI
- **Validation**: Pydantic v2 at every layer
