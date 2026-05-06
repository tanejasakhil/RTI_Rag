from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    aistudio_api_key: str
    hf_token: Optional[str] = None
    model_name: str = "gemma-4-31b-it"
    aistudio_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Retrieval
    top_k_retrieve: int = 15        # Cast wide net first
    top_n_rerank: int = 5           # Keep best 5 after reranking
    similarity_cutoff: float = 0.4  # Discard weak matches

    # Chunking
    chunk_size: int = 512           # Smaller chunks for precision
    chunk_overlap: int = 50         # Minimal overlap to reduce waste

    # Model
    max_tokens: int = 4096
    context_window: int = 131072    # Gemma 4 31B supports 128K context
    timeout: float = 120.0          # seconds; large outputs need more time
    temperature: float = 0.1        # Low = less hallucination

    # Rate limiting
    max_requests_per_minute: int = 28  # Buffer below AI Studio's 30 RPM

    # Models
    embed_model: str = "BAAI/bge-base-en-v1.5"           # 109M params, ~440MB VRAM
    reranker_model: str = "BAAI/bge-reranker-base"  # 109M params, ~440MB VRAM
    translation_model: str = "Helsinki-NLP/opus-mt-hi-en"
    translation_model_revision: str = "refs/pr/2"  # safetensors branch, avoids torch.load CVE

    # Paths
    pdf_dir: str = "RTI_Docs"
    qdrant_path: str = "./qdrant_db"
    collection_name: str = "rti_docs"
    parsed_json: str = "parsed_docs.json"   # intermediate parse+translate cache

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_low(cls, v: float) -> float:
        if v > 0.3:
            raise ValueError("Temperature > 0.3 increases hallucination risk. Keep it low.")
        return v

    model_config = {"env_file": ".env", "extra": "ignore"}

config = AppConfig()
