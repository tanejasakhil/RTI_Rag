import json
import os
import re
import gc
import logging

import fitz  # PyMuPDF
import torch
import pymupdf4llm
from langdetect import detect, LangDetectException
from llama_index.core import Document

logger = logging.getLogger(__name__)

MIN_PDF_SIZE = 1024        # 1 KB — skip corrupt/placeholder files
OCR_CHARS_PER_PAGE = 100  # below this average → treat as scanned, use OCR


def extract_years(text: str) -> list:
    return list(set(re.findall(r'\b(19|20)\d{2}\b', text)))


def _is_scanned(text: str, filepath: str) -> bool:
    """Return True if the PDF looks like a scan (too little embedded text)."""
    try:
        doc = fitz.open(filepath)
        n_pages = max(len(doc), 1)
        doc.close()
    except Exception:
        n_pages = 1
    return len(text.strip()) / n_pages < OCR_CHARS_PER_PAGE


def _ocr_with_easyocr(filepath: str, reader) -> str:
    """Render each PDF page to an image and run EasyOCR on it."""
    doc = fitz.open(filepath)
    page_texts = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        results = reader.readtext(img_bytes, detail=0, paragraph=True)
        page_texts.append("\n".join(results))
    doc.close()
    return "\n\n".join(page_texts)


# ── Phase 1: Parse all PDFs → JSON ──

def parse_pdfs_to_json(pdf_dir: str, output_path: str) -> list:
    """Extract text from every PDF; falls back to EasyOCR for scanned pages.

    Detects language per document and saves all results to output_path as JSON.
    Re-uses existing JSON if present — delete the file to force a full re-parse.
    """
    existing = {}
    if os.path.exists(output_path):
        try:
            existing = {d["file_name"]: d for d in json.load(open(output_path, encoding="utf-8"))}
            logger.info(f"Loaded {len(existing)} cached entries from {output_path}")
        except Exception:
            existing = {}

    docs = dict(existing)

    # Collect files that need processing before deciding whether to load EasyOCR
    pending = []
    for filename in sorted(os.listdir(pdf_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(pdf_dir, filename)
        if os.path.getsize(filepath) < MIN_PDF_SIZE:
            logger.warning(f"Skipping likely-corrupt PDF ({os.path.getsize(filepath)}B): {filename}")
            continue
        if filename not in existing:
            pending.append((filename, filepath))

    ocr_reader = None  # lazy-load only if a scanned PDF is found

    for filename, filepath in pending:
        # ── Fast path: embedded text extraction ──
        try:
            md_text = pymupdf4llm.to_markdown(filepath)
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            continue

        # ── OCR fallback for scanned PDFs ──
        if _is_scanned(md_text, filepath):
            logger.info(f"Scanned PDF detected, running EasyOCR: {filename}")
            if ocr_reader is None:
                import easyocr
                device_gpu = torch.cuda.is_available()
                ocr_reader = easyocr.Reader(["en", "hi"], gpu=device_gpu)
                logger.info(f"EasyOCR loaded (gpu={device_gpu})")
            try:
                md_text = _ocr_with_easyocr(filepath, ocr_reader)
            except Exception as e:
                logger.error(f"OCR failed for {filename}: {e}")
                continue

        if not md_text.strip():
            logger.warning(f"Empty content after parsing: {filename}")
            continue

        try:
            lang = detect(md_text[:3000])
        except LangDetectException:
            lang = "unknown"

        years_in_name = extract_years(filename)
        years_in_content = extract_years(md_text[:3000])
        all_years = years_in_name or years_in_content

        docs[filename] = {
            "file_name": filename,
            "source_path": filepath,
            "text": md_text,
            "language": lang,
            "translated": False,
            "ocr": ocr_reader is not None and _is_scanned("", filepath),
            "year": max(all_years) if all_years else "unknown",
            "years_mentioned": all_years,
            "is_amendment": "amendment" in filename.lower() or "circular" in filename.lower(),
        }
        logger.info(f"Parsed: {filename} | Lang: {lang} | Chars: {len(md_text)}")

    # Free EasyOCR GPU memory before translation / embedding phases
    if ocr_reader is not None:
        del ocr_reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("EasyOCR model unloaded.")

    doc_list = list(docs.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_list, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(doc_list)} documents to {output_path}")
    return doc_list


# ── Phase 2: Translate Hindi → English, drop other non-English ──

def _translate_chunks(text: str, tokenizer, model, device: str) -> str:
    """Translate text paragraph-by-paragraph in batches ≤400 tokens (MarianMT limit)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    translated_parts = []
    batch, batch_tokens = [], 0

    def flush(b):
        inputs = tokenizer(
            b, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs)
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    for para in paragraphs:
        n = len(para.split())
        if batch_tokens + n > 400 and batch:
            translated_parts.extend(flush(batch))
            batch, batch_tokens = [], 0
        batch.append(para)
        batch_tokens += n

    if batch:
        translated_parts.extend(flush(batch))

    return "\n\n".join(translated_parts)


def translate_and_filter(json_path: str) -> list:
    """Translate Hindi docs to English (IndicTrans2) and drop all other non-English docs.

    Overwrites json_path with the filtered, translated corpus.
    Returns the final list of English-only document dicts.
    """
    from config import config

    with open(json_path, encoding="utf-8") as f:
        docs = json.load(f)

    hindi_docs = [d for d in docs if d["language"] == "hi" and not d["translated"]]

    if hindi_docs:
        logger.info(f"Translating {len(hindi_docs)} Hindi document(s) with IndicTrans2...")
        from transformers import MarianMTModel, MarianTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_kwargs = {"revision": config.translation_model_revision, "token": config.hf_token}
        tokenizer = MarianTokenizer.from_pretrained(config.translation_model, **hf_kwargs)
        model = MarianMTModel.from_pretrained(
            config.translation_model, **hf_kwargs
        ).to(device)
        logger.info(f"Translation model loaded on {device}")

        for doc in hindi_docs:
            logger.info(f"  Translating: {doc['file_name']}")
            doc["text"] = _translate_chunks(doc["text"], tokenizer, model, device)
            doc["language"] = "en"
            doc["translated"] = True

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("IndicTrans2 model unloaded.")

    english_docs = [d for d in docs if d["language"] == "en"]
    dropped = len(docs) - len(english_docs)
    if dropped:
        langs = {d["language"] for d in docs if d["language"] not in ("en", "hi")}
        logger.info(f"Dropped {dropped} non-English document(s) (languages: {langs})")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(english_docs, f, indent=2, ensure_ascii=False)
    logger.info(f"Final English-only corpus: {len(english_docs)} documents → {json_path}")
    return english_docs


# ── Phase 3: Convert JSON → LlamaIndex Documents ──

def load_documents_from_json(json_path: str) -> list:
    """Load the filtered JSON corpus as LlamaIndex Document objects."""
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    documents = []
    for r in records:
        documents.append(Document(
            text=r["text"],
            metadata={
                "file_name": r["file_name"],
                "source_path": r["source_path"],
                "year": r["year"],
                "years_mentioned": r["years_mentioned"],
                "is_amendment": r["is_amendment"],
                "language": r["language"],
                "translated": r["translated"],
            }
        ))
    return documents
