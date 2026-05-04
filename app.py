import streamlit as st
import qdrant_client
import time
from collections import deque
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, LongContextReorder
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models import UserQuery
from config import config
from pydantic import ValidationError


class RateLimiter:
    def __init__(self, max_requests=18, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps = deque()

    def wait_if_needed(self):
        now = time.time()
        while self.timestamps and now - self.timestamps[0] > self.window:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_requests:
            wait_time = self.window - (now - self.timestamps[0]) + 0.5
            if wait_time > 0:
                st.info(f"Rate limit reached - waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
        self.timestamps.append(time.time())


@st.cache_resource
def load_query_engine():
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
        "You are a research assistant for RTI (Right to Information) documents.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using the context provided below.\n"
        "2. If the answer is not in the context, say exactly: "
        "\"I could not find this information in the provided documents.\"\n"
        "3. Never infer, assume, or use outside knowledge.\n"
        "4. ALWAYS cite the source document filename and date at the end.\n"
        "5. If multiple documents conflict, prefer the more recent one.\n"
        "6. If context is partial, state what you found and what is missing.\n\n"
        "Format your answer as:\n"
        "[Answer]\n...your answer here...\n\n"
        "[Sources]\n- filename.pdf (Year: XXXX)\n\n"
        "Context:\n-----------\n{context_str}\n-----------\n\n"
        "Question: {query_str}\nAnswer:"
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[similarity_filter, reranker, reorder],
        streaming=True,
    )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": strict_prompt}
    )
    return query_engine


# ── Streamlit UI ──
st.set_page_config(
    page_title="RTI Research Assistant", page_icon="📄", layout="centered"
)
st.title("📄 RTI Document Research Assistant")
st.caption(f"Model: `{config.model_name}` | Source: rti.dopt.gov.in")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rate_limiter" not in st.session_state:
    st.session_state.rate_limiter = RateLimiter(
        max_requests=config.max_requests_per_minute
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about the RTI documents..."):
    try:
        validated = UserQuery(question=prompt)
    except ValidationError as e:
        st.error(f"Invalid input: {e.errors()[0]['msg']}")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": validated.question}
    )
    with st.chat_message("user"):
        st.write(validated.question)

    st.session_state.rate_limiter.wait_if_needed()
    query_engine = load_query_engine()

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                response = query_engine.query(validated.question)
                output = st.write_stream(
                    token.delta for token in response.response_gen
                )

            with st.expander("📎 Sources used"):
                for node in response.source_nodes:
                    fname = node.metadata.get("file_name", "unknown")
                    year = node.metadata.get("year", "unknown")
                    score = round(node.score, 3) if node.score else "N/A"
                    st.markdown(
                        f"- **{fname}** | Year: `{year}` | Score: `{score}`"
                    )

        except Exception as e:
            output = f"Error querying documents: {str(e)}"
            st.error(output)

    st.session_state.messages.append(
        {"role": "assistant", "content": output}
    )
