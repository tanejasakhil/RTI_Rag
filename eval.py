"""
Evaluation framework for the RTI RAG system.

Loads eval questions from eval_set.json and runs retrieval recall +
keyword coverage metrics. Results are printed per question and
summarised per category (central vs state-specific).
"""
import json
import logging
import time
from pathlib import Path

import openai

import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, LongContextReorder, SentenceTransformerRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set(path: Path = EVAL_SET_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_query_engine():
    """Build the same query engine used in app.py, non-streaming for eval."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.embed_model)
    Settings.llm = OpenAILike(
        api_key=config.aistudio_api_key,
        api_base=config.aistudio_api_base,
        model=config.model_name,
        max_tokens=config.max_tokens,
        context_window=config.context_window,
        temperature=config.temperature,
        timeout=config.timeout,
        is_chat_model=True,
    )

    client = qdrant_client.QdrantClient(path=config.qdrant_path)
    vector_store = QdrantVectorStore(client=client, collection_name=config.collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=config.top_k_retrieve)
    reranker = SentenceTransformerRerank(model=config.reranker_model, top_n=config.top_n_rerank)
    similarity_filter = SimilarityPostprocessor(similarity_cutoff=config.similarity_cutoff)
    reorder = LongContextReorder()

    strict_prompt = PromptTemplate(
        "You are a research assistant for RTI (Right to Information) documents.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the context provided below.\n"
        "2. If not in context, say: "
        "\"I could not find this information in the provided documents.\"\n"
        "3. Never infer, assume, or use outside knowledge.\n"
        "4. ALWAYS cite the source document filename and date.\n"
        "5. If multiple documents conflict, prefer the more recent one.\n"
        "6. If context is partial, state what you found and what is missing.\n"
        "7. STATE FALLBACK RULE: If a question asks about a specific state but no "
        "state-specific document is present in the context, answer using the central "
        "RTI Act (RTI-Act.pdf) and explicitly note: \"No state-specific rules were "
        "found for [state]; the answer is based on the central RTI Act, 2005, which "
        "applies by default unless the state has enacted its own rules.\"\n\n"
        "Context:\n-----------\n{context_str}\n-----------\n\n"
        "Question: {query_str}\nAnswer:"
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[similarity_filter, reranker, reorder],
        streaming=False,
    )
    query_engine.update_prompts({"response_synthesizer:text_qa_template": strict_prompt})
    return query_engine


def evaluate(query_engine, eval_set: list, out_path: Path) -> list:
    # Resume from existing results
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        logger.info(f"Resuming — {len(done_ids)} already done: {sorted(done_ids)}")
    else:
        results = []
        done_ids = set()

    for item in eval_set:
        if item["id"] in done_ids:
            continue

        expected_sources = set(item["expected_sources"])
        try:
            response = query_engine.query(item["question"])
        except (openai.InternalServerError, openai.APITimeoutError) as e:
            logger.warning(f"[{item['id']}] Transient error ({type(e).__name__}). Waiting 60s then retrying once...")
            time.sleep(60)
            response = query_engine.query(item["question"])

        retrieved_files = {node.metadata.get("file_name", "") for node in response.source_nodes}
        hits = expected_sources & retrieved_files
        recall = len(hits) / len(expected_sources) if expected_sources else 0.0

        answer_text = str(response).lower()
        keyword_hits = sum(1 for kw in item["expected_answer_contains"] if kw.lower() in answer_text)
        keyword_total = len(item["expected_answer_contains"])

        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "retrieval_recall": recall,
            "keyword_coverage": f"{keyword_hits}/{keyword_total}",
            "expected_sources": list(expected_sources),
            "retrieved_sources": list(retrieved_files),
            "answer_preview": str(response)[:300],
        }
        results.append(result)

        # Persist after every question so crashes don't lose progress
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[{item['id']}] Recall: {recall:.2f} | "
            f"Keywords: {keyword_hits}/{keyword_total} | "
            f"Q: {item['question'][:60]}..."
        )

    return results


def print_summary(results: list):
    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)

    print(f"\n{'='*65}")
    print(f"{'CATEGORY':<35} {'AVG RECALL':>10} {'QUESTIONS':>10}")
    print(f"{'='*65}")
    overall_recall = []
    for cat, items in sorted(categories.items()):
        avg = sum(i["retrieval_recall"] for i in items) / len(items)
        overall_recall.extend(i["retrieval_recall"] for i in items)
        print(f"{cat:<35} {avg:>10.2f} {len(items):>10}")

    print(f"{'='*65}")
    print(f"{'OVERALL':<35} {sum(overall_recall)/len(overall_recall):>10.2f} {len(results):>10}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    eval_set = load_eval_set()
    logger.info(f"Loaded {len(eval_set)} eval questions from {EVAL_SET_PATH}")

    logger.info("Building query engine...")
    qe = build_query_engine()

    out_path = Path(__file__).parent / "eval_results.json"
    logger.info("Running evaluation...")
    results = evaluate(qe, eval_set, out_path)

    print_summary(results)
    logger.info(f"Full results saved to {out_path}")
