import gc
import logging

import torch
import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import config
from loader import parse_pdfs_to_json, translate_and_filter, load_documents_from_json
from models import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest():
    # Parse all PDFs → JSON  (PyMuPDF, EasyOCR)
    logger.info(f"Parsing PDFs from: {config.pdf_dir}")
    parse_pdfs_to_json(config.pdf_dir, config.parsed_json)

    # Translate Hindi → English, drop other languages
    logger.info("Translating Hindi docs and filtering non-English...")
    translate_and_filter(config.parsed_json)

    # Load JSON as LlamaIndex Documents 
    llama_docs = load_documents_from_json(config.parsed_json)
    logger.info(f"Loaded {len(llama_docs)} English documents for indexing")

    valid_docs, skipped = [], 0
    for doc in llama_docs:
        try:
            DocumentChunk(
                text=doc.text,
                source_file=doc.metadata.get("file_name", "unknown"),
            )
            valid_docs.append(doc)
        except ValueError as e:
            logger.warning(f"Skipped invalid chunk: {e}")
            skipped += 1
    logger.info(f"Valid: {len(valid_docs)} | Skipped: {skipped}")

    # Embed & Index  (GPU, bge-base-en-v1.5)
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.embed_model)
    Settings.node_parser = MarkdownNodeParser()

    client = qdrant_client.QdrantClient(path=config.qdrant_path)
    vector_store = QdrantVectorStore(client=client, collection_name=config.collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        valid_docs,
        storage_context=storage_context,
        show_progress=True,
    )
    logger.info("Indexing complete — Qdrant DB saved to ./qdrant_db")


if __name__ == "__main__":
    ingest()
