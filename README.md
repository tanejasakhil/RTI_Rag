# RTI Document RAG System

A Retrieval-Augmented Generation system for querying PDF documents crawled from `rti.dopt.gov.in`, covering the central RTI Act and state-specific RTI rules across 20+ Indian states.

## Quick Start

```bash
# 1. Install dependencies (requires Python 3.10+)
pip install -r requirements.txt

# 2. Create .env with your API keys
cp .env.example .env   # then fill in values

# 3. Parse, translate, and index all PDFs (run once)
python ingest.py

# 4. Launch the chat UI
python -m streamlit run app.py

# 5. (Optional) Run evaluation
python eval.py
```

## Architecture

| Component | Model / Tool | Notes |
|---|---|---|
| **PDF Parsing** | `pymupdf4llm` | CPU-only, markdown-preserving extraction |
| **OCR Fallback** | `EasyOCR` (en + hi) | Auto-detected for scanned PDFs |
| **Translation** | `Helsinki-NLP/opus-mt-hi-en` | Hindi → English, ~300MB, GPU |
| **Embeddings** | `BAAI/bge-base-en-v1.5` | 109M params, ~440MB VRAM |
| **Vector DB** | Qdrant (local disk) | Persisted at `./qdrant_db` |
| **Reranker** | `BAAI/bge-reranker-base` | 109M params, ~440MB VRAM |
| **LLM** | `gemma-4-31b-it` via Google AI Studio | Free tier, OpenAI-compatible endpoint |
| **Frontend** | Streamlit chat UI | Streaming responses + source citations |
| **Validation** | Pydantic v2 | Input, chunk, and response validation |

## Ingestion Pipeline

```
RTI_Docs/ (PDFs)
    │
    ▼
pymupdf4llm  ──── scanned? ──→  EasyOCR (en+hi)
    │
    ▼
parsed_docs.json  (all languages, cached)
    │
    ▼
Helsinki opus-mt-hi-en  →  translate Hindi → English
Drop other non-English languages
    │
    ▼
BAAI/bge-base-en-v1.5  →  embed chunks
    │
    ▼
Qdrant (./qdrant_db)
```

Re-running `ingest.py` skips already-parsed files (cached in `parsed_docs.json`). Delete that file to force a full re-parse.

## Query Pipeline

```
User question
    │
    ▼
bge-base-en-v1.5  →  vector search (top 15)
    │
    ▼
SimilarityPostprocessor  →  filter score < 0.4
    │
    ▼
bge-reranker-base  →  rerank, keep top 5
    │
    ▼
LongContextReorder  →  best chunks at edges
    │
    ▼
gemma-4-31b-it (Google AI Studio)  →  answer with citations
```

## Environment Variables

Create a `.env` file in the project root:

```
AISTUDIO_API_KEY=your-google-ai-studio-key
HF_TOKEN=your-huggingface-token        # required for gated models
```

Get your key at [aistudio.google.com](https://aistudio.google.com) → API keys.

## Hardware Requirements

Tested on NVIDIA GTX 1650 (4GB VRAM), torch 2.5.1+cu121.

| Phase | VRAM Used |
|---|---|
| OCR (ingest) | ~1GB (EasyOCR, freed after) |
| Translation (ingest) | ~300MB (MarianMT, freed after) |
| Embedding (ingest) | ~440MB |
| Query time (app) | ~880MB (embed + reranker) |

## Evaluation

Questions are defined in `eval_set.json` (15 questions — 6 central RTI + 9 state-specific). Results are saved to `eval_results.json` with per-category recall breakdown.

```bash
python eval.py
```
