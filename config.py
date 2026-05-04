from pydantic import field_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    openrouter_api_key: str
    model_name: str = "google/gemma-4-31b-it:free"

    # Retrieval
    top_k_retrieve: int = 15        # Cast wide net first
    top_n_rerank: int = 5           # Keep best 5 after reranking
    similarity_cutoff: float = 0.4  # Discard weak matches

    # Chunking
    chunk_size: int = 512           # Smaller chunks for precision
    chunk_overlap: int = 50         # Minimal overlap to reduce waste

    # Model
    max_tokens: int = 1024
    temperature: float = 0.1        # Low = less hallucination

    # Rate limiting
    max_requests_per_minute: int = 18  # Buffer below 20 limit

    # Paths
    pdf_dir: str = "downloaded_pdfs"
    qdrant_path: str = "./qdrant_db"
    collection_name: str = "rti_docs"
    hash_file: str = "indexed_hashes.json"  # For incremental ingestion

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_low(cls, v: float) -> float:
        if v > 0.3:
            raise ValueError("Temperature > 0.3 increases hallucination risk. Keep it low.")
        return v

    model_config = {"env_file": ".env"}

config = AppConfig()
