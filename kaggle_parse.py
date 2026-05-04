"""
Kaggle GPU Parsing Script for RTI PDFs
=======================================
Run this on Kaggle with T4x2 GPU accelerator.

Setup:
  1. Create a Kaggle dataset with all your PDFs uploaded
  2. Create a new notebook, attach the dataset
  3. Enable T4x2 GPU accelerator
  4. Paste this script into a cell and run
  5. Download the output JSON from /kaggle/working/parsed_documents.json

Install dependencies first (run in a separate cell):
  !pip install "docling[vlm]" -q
"""

import torch
import json
import os
import re
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — Update this to match your Kaggle dataset name
# ══════════════════════════════════════════════════════════════

# Change this to your actual Kaggle dataset path
PDF_DIR = "/kaggle/input/rti-pdfs"  # ← UPDATE if your dataset name differs
OUTPUT_FILE = "/kaggle/working/parsed_documents.json"

# ══════════════════════════════════════════════════════════════

# Non-English PDF filter
SKIP_LANG_PATTERNS = [
    "hindi", "urdu", "telgu", "telugu", "punjabi", "odia",
    "gujarati", "kannada", "malayalam", "assamese", "marathi"
]
MIN_PDF_SIZE = 1024  # 1KB


def extract_years(text: str) -> list:
    """Extract all 4-digit years from text."""
    return list(set(re.findall(r'\b(19|20)\d{2}\b', text)))


def main():
    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            logger.info(f"GPU {i}: {name} ({mem:.1f} GB)")
    else:
        logger.warning("No GPU detected! This will be very slow.")

    # Import docling
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions, VlmConvertOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # Configure Granite-Docling VLM — runs on GPU automatically
    vlm_options = VlmConvertOptions.from_preset("granite_docling")
    pipeline_options = VlmPipelineOptions(vlm_options=vlm_options)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    # Collect PDF files
    pdf_files = []
    for filename in sorted(os.listdir(PDF_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(PDF_DIR, filename)

        if os.path.getsize(filepath) < MIN_PDF_SIZE:
            logger.warning(f"Skipping corrupt/tiny PDF ({os.path.getsize(filepath)}B): {filename}")
            continue

        if any(p in filename.lower() for p in SKIP_LANG_PATTERNS):
            logger.info(f"Skipping non-English PDF: {filename}")
            continue

        pdf_files.append((filename, filepath))

    logger.info(f"Found {len(pdf_files)} English PDFs to parse")

    # Parse each PDF
    parsed_docs = []
    failed = []

    for idx, (filename, filepath) in enumerate(pdf_files, 1):
        logger.info(f"[{idx}/{len(pdf_files)}] Parsing: {filename}")
        start = time.time()

        try:
            result = converter.convert(filepath)
            markdown_text = result.document.export_to_markdown()

            if not markdown_text.strip():
                logger.warning(f"Empty content after parsing: {filename}")
                failed.append({"file": filename, "reason": "empty content"})
                continue

            # Extract metadata
            filename_years = extract_years(filename)
            content_years = extract_years(markdown_text[:3000])
            all_years = filename_years or content_years

            doc = {
                "text": markdown_text,
                "metadata": {
                    "file_name": filename,
                    "year": max(all_years) if all_years else "unknown",
                    "years_mentioned": all_years,
                    "is_amendment": "amendment" in filename.lower() or "circular" in filename.lower(),
                    "language": "en",
                }
            }
            parsed_docs.append(doc)

            elapsed = time.time() - start
            logger.info(f"  ✅ Done in {elapsed:.1f}s — {len(markdown_text)} chars extracted")

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"  ❌ Failed after {elapsed:.1f}s: {e}")
            failed.append({"file": filename, "reason": str(e)})

    # Save results
    output = {
        "total_parsed": len(parsed_docs),
        "total_failed": len(failed),
        "failed_files": failed,
        "documents": parsed_docs,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Parsing complete!")
    logger.info(f"  Parsed: {len(parsed_docs)} documents")
    logger.info(f"  Failed: {len(failed)} documents")
    logger.info(f"  Output: {OUTPUT_FILE}")
    logger.info(f"  Size:   {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")
    logger.info(f"{'='*60}")
    logger.info("Download parsed_documents.json and place it in your rti-rag/ folder")


if __name__ == "__main__":
    main()
