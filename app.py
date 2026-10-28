import streamlit as st
from src.pipeline import AdvancedRAGPipeline

st.set_page_config(page_title="Track C: Advanced RAG System", layout="wide")
st.title("Track C — Advanced Corrective RAG Q&A System")
st.caption(
    "Techniques: Query Decomposition • Hybrid Search (Vector + BM25 RRF) • Cross-Encoder Re-Ranking • Corrective Loop (CRAG) • Grounded Citations"
)

@st.cache_resource
def get_pipeline() -> AdvancedRAGPipeline:
    return AdvancedRAGPipeline()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = get_pipeline()

pipeline: AdvancedRAGPipeline = st.session_state.pipeline

with st.sidebar:
    st.header("Session State & Inspection")
    st.metric("Indexed Chunks (15 Docs)", pipeline.total_chunks)
    st.metric("Conversation Turns", len(pipeline.state.turns))

    if st.button("Clear Conversation Memory"):
        pipeline.state.turns.clear()
        st.rerun()

    with st.expander("Inspect Raw Session State (JSON)"):
        st.code(pipeline.state.to_json(), language="json")

for turn in pipeline.state.turns:
    with st.chat_message("user"):
        st.write(turn.user_query)
    with st.chat_message("assistant"):
        st.write(turn.answer)
        with st.expander(
            f"Turn {turn.turn_id} Trace | CRAG: {turn.crag_grade} | Abstained: {turn.abstained} | Citations: {turn.citations}"
        ):
            st.markdown(f"**1. Rewritten / Decomposed Queries:** `{turn.transformed_queries}`")
            st.markdown(
                f"**2. Corrective Loop (CRAG):** `{turn.crag_grade}`"
                + (f" → Re-queried with: `{turn.crag_retry_query}`" if turn.crag_retry_query else "")
            )
            st.markdown("**3. Top Cross-Encoder Re-Ranked Chunks:**")
            st.dataframe(turn.reranked_chunks, use_container_width=True)
            st.markdown("**4. LLM Call Telemetry (Tokens & Latency):**")
            st.dataframe([c.__dict__ for c in turn.llm_calls], use_container_width=True)

query = st.chat_input("Ask a question about RAG, Hybrid Search, Embeddings, Chunking, or Vector DBs...")
if query:
    with st.chat_message("user"):
        st.write(query)
    with st.spinner("Running Query Rewrite → Hybrid RRF → Cross-Encoder → CRAG → Grounded Generation..."):
        pipeline.ask(query)
    st.rerun()
    