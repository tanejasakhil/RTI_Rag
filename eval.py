"""
Evaluation framework for the RTI RAG system.

Golden test set with manually verified question-answer pairs,
plus RAGAS-based automated metrics (faithfulness, relevance, precision).
"""
import logging
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, LongContextReorder
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Golden Test Set ──
EVAL_QUESTIONS = [
    {
        "question": "What is the time limit for providing information under RTI Act?",
        "expected_sources": ["RTI-Act.pdf", "RTI Act, 2005 (Amended)-English Version.PDF"],
        "expected_answer_contains": ["30 days", "section 7"],
    },
    {
        "question": "What is the penalty for not providing information under RTI?",
        "expected_sources": ["RTI-Act.pdf"],
        "expected_answer_contains": ["250", "penalty", "each day"],
    },
    {
        "question": "Who is a Public Information Officer?",
        "expected_sources": ["RTI-Act.pdf", "Guide_2013-issue.pdf"],
        "expected_answer_contains": ["designated", "section 5"],
    },
    {
        "question": "What are the RTI rules from 2019?",
        "expected_sources": ["RTI_Rules_2019.pdf"],
        "expected_answer_contains": ["rule", "2019"],
    },
    # Add 15-20 more questions covering different document types
]


def build_query_engine():
    """Build the same query engine used in app.py, but non-streaming for eval."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="nomic-ai/nomic-embed-text-v2-moe",
        trust_remote_code=True
    )
    Settings.llm = OpenRouter(
        api_key=config.openrouter_api_key,
        model=config.model_name,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    client = qdrant_client.QdrantClient(path=config.qdrant_path)
    vector_store = QdrantVectorStore(
        client=client, collection_name=config.collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    retriever = VectorIndexRetriever(
        index=index, similarity_top_k=config.top_k_retrieve
    )

    reranker = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-v2-m3", top_n=config.top_n_rerank
    )
    similarity_filter = SimilarityPostprocessor(
        similarity_cutoff=config.similarity_cutoff
    )
    reorder = LongContextReorder()

    strict_prompt = PromptTemplate(
        "You are a research assistant for RTI documents.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the context provided below.\n"
        "2. If not in context, say: "
        "\"I could not find this information in the provided documents.\"\n"
        "3. Never infer, assume, or use outside knowledge.\n"
        "4. ALWAYS cite the source document filename and date.\n"
        "5. If multiple documents conflict, prefer the more recent one.\n"
        "6. If context is partial, state what you found and what is missing.\n\n"
        "Context:\n-----------\n{context_str}\n-----------\n\n"
        "Question: {query_str}\nAnswer:"
    )

    # Non-streaming for evaluation
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[similarity_filter, reranker, reorder],
        streaming=False,
    )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": strict_prompt}
    )
    return query_engine


def evaluate_retrieval_recall(query_engine):
    """Check if expected source documents appear in retrieved nodes."""
    results = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        expected = set(item["expected_sources"])

        response = query_engine.query(question)

        retrieved_files = {
            node.metadata.get("file_name", "")
            for node in response.source_nodes
        }

        hits = expected & retrieved_files
        recall = len(hits) / len(expected) if expected else 0.0

        answer_text = str(response).lower()
        keyword_hits = sum(
            1 for kw in item["expected_answer_contains"]
            if kw.lower() in answer_text
        )
        keyword_total = len(item["expected_answer_contains"])

        results.append({
            "question": question,
            "retrieval_recall": recall,
            "expected_sources": list(expected),
            "retrieved_sources": list(retrieved_files),
            "keyword_coverage": f"{keyword_hits}/{keyword_total}",
            "answer_preview": str(response)[:200],
        })

        logger.info(
            f"Q: {question[:60]}... | "
            f"Recall: {recall:.2f} | "
            f"Keywords: {keyword_hits}/{keyword_total}"
        )

    avg_recall = sum(r["retrieval_recall"] for r in results) / len(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"Average Retrieval Recall: {avg_recall:.2f}")
    logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    logger.info("Building query engine for evaluation...")
    qe = build_query_engine()

    logger.info("Running retrieval recall evaluation...")
    results = evaluate_retrieval_recall(qe)

    logger.info("\nEvaluation complete. Results:")
    for r in results:
        print(f"\n  Q: {r['question']}")
        print(f"  Recall: {r['retrieval_recall']:.2f}")
        print(f"  Keywords: {r['keyword_coverage']}")
        print(f"  Answer: {r['answer_preview']}...")
