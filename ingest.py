from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models import DocumentChunk
import qdrant_client
import hashlib
import json
import os
import gc
import logging

import torch

from config import config

# Import the Granite-Docling loader
from loader import load_pdfs_with_granite_docling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Incremental Ingestion ──

def get_file_hash(filepath: str) -> str:
    """SHA256 hash for change detection."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_new_or_changed_files(pdf_dir: str) -> list:
    """Compare current PDFs against stored hashes to find new/changed files."""
    existing = json.load(open(config.hash_file)) if os.path.exists(config.hash_file) else {}
    new_files = []
    current_hashes = {}

    for f in os.listdir(pdf_dir):
        if not f.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, f)
        h = get_file_hash(path)
        current_hashes[f] = h
        if existing.get(f) != h:
            new_files.append(f)

    # Save updated hashes
    json.dump(current_hashes, open(config.hash_file, "w"), indent=2)
    return new_files


def free_gpu_memory():
    """Force-free all GPU memory so the next model can use it."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(
            f"GPU memory freed. "
            f"Allocated: {torch.cuda.memory_allocated() / 1e6:.0f}MB, "
            f"Reserved: {torch.cuda.memory_reserved() / 1e6:.0f}MB"
        )


def ingest():
    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Parse PDFs with Granite-Docling VLM (uses full GPU)
    # ═══════════════════════════════════════════════════════════════
    logger.info(f"Loading PDFs from: {config.pdf_dir}")
    documents = load_pdfs_with_granite_docling(config.pdf_dir)
    logger.info(f"Loaded {len(documents)} documents")

    # ── Free GPU from Granite-Docling before loading embedding model ──
    logger.info("Freeing GPU memory from Granite-Docling VLM...")
    free_gpu_memory()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Embed & Index (embedding model now gets full GPU)
    # ═══════════════════════════════════════════════════════════════

    # ── Embedding model (runs locally, free) ──
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="nomic-ai/nomic-embed-text-v2-moe",   # MoE arch, better on long docs
        trust_remote_code=True
    )

    # ── Chunking strategy ──
    # Since Granite-Docling outputs markdown, use MarkdownNodeParser to split
    # on headings — this keeps sections/articles as coherent units.
    # Falls back to SentenceSplitter for content without markdown headings.
    Settings.node_parser = MarkdownNodeParser()

    # ── Qdrant local setup ──
    client = qdrant_client.QdrantClient(path=config.qdrant_path)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Validate chunks with Pydantic before indexing ──
    valid_docs = []
    skipped = 0
    for doc in documents:
        try:
            DocumentChunk(
                text=doc.text,
                source_file=doc.metadata.get("file_name", "unknown"),
            )
            valid_docs.append(doc)
        except ValueError as e:
            logger.warning(f"Skipped invalid chunk: {e}")
            skipped += 1

    logger.info(f"Valid documents: {len(valid_docs)} | Skipped: {skipped}")

    # ── Index ──
    VectorStoreIndex.from_documents(
        valid_docs,
        storage_context=storage_context,
        show_progress=True
    )

    logger.info("✅ Indexing complete! Qdrant DB saved to ./qdrant_db")

if __name__ == "__main__":
    ingest()
