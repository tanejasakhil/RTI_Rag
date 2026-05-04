from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

# ── Input Validation ──

class UserQuery(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Question too short — please be more specific.")
        if len(v) > 1000:
            raise ValueError("Question too long — please trim it down.")
        return v


# ── Chunk Validation (used during ingestion) ──

class DocumentChunk(BaseModel):
    text: str
    source_file: str
    page_number: Optional[int] = None
    char_count: int = Field(default=0)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text is empty — skipping.")
        return v.strip()

    @model_validator(mode="after")
    def compute_char_count(self) -> "DocumentChunk":
        self.char_count = len(self.text)
        return self


# ── LLM Output Validation ──

class RAGResponse(BaseModel):
    """Used to validate and structure the final response shown to the user.
    Wire this into the query flow to ensure every response includes sources."""
    answer: str = Field(description="Answer strictly based on the provided context")
    sources: List[str] = Field(description="Source document filenames cited")
    confidence: str = Field(description="Confidence level: high / medium / low")
    found_in_docs: bool = Field(description="True if answer was found in documents")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        if v not in ("high", "medium", "low"):
            raise ValueError(f"Confidence must be high/medium/low, got: {v}")
        return v
