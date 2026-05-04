"""
Import pre-parsed documents from Kaggle JSON into Qdrant.

Usage:
  1. Run kaggle_parse.py on Kaggle (T4 GPU) to get parsed_documents.json
  2. Place parsed_documents.json in this directory
  3. Run: uv run python ingest_from_json.py
"""

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models import DocumentChunk
import qdrant_client
import json
import os
import logging

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PARSED_JSON = "parsed_documents.json"


def ingest_from_json():
    if not os.path.exists(PARSED_JSON):
        logger.error(
            f"{PARSED_JSON} not found! "
            "Run kaggle_parse.py on Kaggle first and download the output."
        )
        return

    # ── Load parsed documents from JSON ──
    with open(PARSED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        f"Loaded {data['total_parsed']} parsed documents "
        f"({data['total_failed']} failed during Kaggle parsing)"
    )

    if data["failed_files"]:
        logger.warning(f"Failed files: {[f['file'] for f in data['failed_files']]}")

    # Convert JSON back to LlamaIndex Document objects
    documents = []
    for doc_data in data["documents"]:
        doc = Document(
            text=doc_data["text"],
            metadata=doc_data["metadata"],
        )
        documents.append(doc)

    # ── Embedding model (runs on GPU) ──
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="nomic-ai/nomic-embed-text-v2-moe",
        trust_remote_code=True,
    )

    # ── Chunking — splits on markdown headings ──
    Settings.node_parser = MarkdownNodeParser()

    # ── Qdrant local setup ──
    client = qdrant_client.QdrantClient(path=config.qdrant_path)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Validate chunks with Pydantic ──
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

    # ── Index (embedding happens here on GPU) ──
    VectorStoreIndex.from_documents(
        valid_docs,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("✅ Indexing complete! Qdrant DB saved to ./qdrant_db")


if __name__ == "__main__":
    ingest_from_json()
