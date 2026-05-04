import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, VlmConvertOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from llama_index.core import Document
import os
import re
import logging

logger = logging.getLogger(__name__)

# ── GPU Detection ──
HAS_GPU = torch.cuda.is_available()
if HAS_GPU:
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} — Granite-Docling will use CUDA")
else:
    logger.info("No GPU detected — Granite-Docling will run on CPU (slower but functional)")

# ── Non-English PDF filter ──
# These PDFs use Indic scripts that degrade embedding quality for English-language queries.
# Filter them at ingest time to avoid polluting the index with low-quality chunks.
SKIP_LANG_PATTERNS = [
    "hindi", "urdu", "telgu", "telugu", "punjabi", "odia",
    "gujarati", "kannada", "malayalam", "assamese", "marathi"
]

MIN_PDF_SIZE = 1024  # 1KB — anything smaller is corrupt/placeholder


def extract_years(text: str) -> list:
    """Extract all 4-digit years from text."""
    return list(set(re.findall(r'\b(19|20)\d{2}\b', text)))


def load_pdfs_with_granite_docling(pdf_dir: str) -> list:
    """Load and parse PDFs using Granite-Docling VLM for structure-aware extraction.
    
    Returns a list of LlamaIndex Document objects with metadata including
    year, amendment status, and language tags.
    """
    # Configure Granite-Docling VLM pipeline
    # Force CPU — the VLM vision encoder's activation memory exceeds 4GB VRAM
    # even though model weights are only ~1GB (attention scales quadratically
    # with image resolution). GPU is reserved for the embedding model instead.
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    import multiprocessing

    accelerator_options = AcceleratorOptions(
        num_threads=multiprocessing.cpu_count(),
        device=AcceleratorDevice.CPU,
    )

    vlm_options = VlmConvertOptions.from_preset("granite_docling")
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        accelerator_options=accelerator_options,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    documents = []

    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(pdf_dir, filename)

        # ── Skip corrupt/empty PDFs ──
        if os.path.getsize(filepath) < MIN_PDF_SIZE:
            logger.warning(f"Skipping likely-corrupt PDF ({os.path.getsize(filepath)} bytes): {filename}")
            continue

        # ── Skip non-English PDFs ──
        if any(p in filename.lower() for p in SKIP_LANG_PATTERNS):
            logger.info(f"Skipping non-English PDF: {filename}")
            continue

        try:
            result = converter.convert(filepath)
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            continue

        # Granite-Docling exports to markdown — tables become proper markdown tables
        markdown_text = result.document.export_to_markdown()

        if not markdown_text.strip():
            logger.warning(f"Empty content after parsing: {filename}")
            continue

        # ── Extract metadata ──
        filename_years = extract_years(filename)
        content_years = extract_years(markdown_text[:3000])
        all_years = filename_years or content_years

        doc = Document(
            text=markdown_text,
            metadata={
                "file_name": filename,
                "source_path": filepath,
                "year": max(all_years) if all_years else "unknown",
                "years_mentioned": all_years,
                "is_amendment": "amendment" in filename.lower() or "circular" in filename.lower(),
                "language": "en",
            }
        )
        documents.append(doc)

    return documents
