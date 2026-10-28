import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from src.config import Config
from src.prompts import (
    QUERY_TRANSFORM_SYSTEM_PROMPT,
    CRAG_GRADER_SYSTEM_PROMPT,
    GROUNDED_GENERATOR_SYSTEM_PROMPT,
)
from src.retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.state import SessionState, TurnRecord, LLMCallLog

class AdvancedRAGPipeline:
    def __init__(self):
        if not Config.GROQ_API_KEY or "paste_your_key" in Config.GROQ_API_KEY:
            raise ValueError(
                "Missing GROQ_API_KEY in .env! Get a free key at https://console.groq.com/keys and put GROQ_API_KEY=gsk_... in your .env file."
            )
        self.client = OpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL,
        )
        self.retriever = HybridRetriever()
        self.total_chunks = self.retriever.load_and_index(Config.DOCS_DIR)
        self.reranker = CrossEncoderReranker()
        self.state = SessionState()

    def _call_llm_json(
        self, stage: str, system_prompt: str, user_prompt: str, call_logs: List[LLMCallLog]
    ) -> Dict[str, Any]:
        """Calls the LLM in strict JSON mode and logs tokens + latency."""
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        content = response.choices[0].message.content or "{}"
        usage = response.usage

        call_logs.append(
            LLMCallLog(
                stage=stage,
                model=Config.LLM_MODEL,
                latency_ms=latency_ms,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                prompt_preview=user_prompt[:120].replace("\n", " ") + "...",
            )
        )
        return json.loads(content)

    def _transform_query(self, user_query: str, call_logs: List[LLMCallLog]) -> List[str]:
        """Technique 1: Rewrite & decompose multi-part questions using conversation history."""
        user_prompt = (
            f"CONVERSATION HISTORY:\n{self.state.get_recent_history()}\n\n"
            f"USER QUESTION:\n{user_query}"
        )
        data = self._call_llm_json(
            "query_transformation", QUERY_TRANSFORM_SYSTEM_PROMPT, user_prompt, call_logs
        )
        queries = [q.strip() for q in data.get("queries", []) if q.strip()]
        return queries if queries else [user_query]

    def _grade_context(
        self, user_query: str, reranked_chunks: List[Dict[str, Any]], call_logs: List[LLMCallLog]
    ) -> Tuple[str, str]:
        """Technique 4: Corrective Loop (CRAG) — grades retrieved context quality."""
        evidence_text = "\n\n".join(
            [f"[{c['source_file']}] (Score: {c['rerank_score']}): {c['text']}" for c in reranked_chunks]
        )
        user_prompt = f"USER QUESTION:\n{user_query}\n\nRETRIEVED CHUNKS:\n{evidence_text}"
        data = self._call_llm_json(
            "crag_context_grader", CRAG_GRADER_SYSTEM_PROMPT, user_prompt, call_logs
        )
        grade = data.get("grade", "WEAK").upper()
        rewritten = data.get("rewritten_query", "").strip()
        return grade, rewritten

    def _generate_answer(
        self, user_query: str, reranked_chunks: List[Dict[str, Any]], call_logs: List[LLMCallLog]
    ) -> Dict[str, Any]:
        """Technique 5: Grounded Answer generation with inline citations or abstention."""
        # Filter out clearly irrelevant chunks (Cross-Encoder logits below -4.0)
        valid_chunks = [c for c in reranked_chunks if c["rerank_score"] > -4.0]

        if not valid_chunks:
            evidence_str = "NO RELEVANT EVIDENCE CHUNKS FOUND."
        else:
            evidence_str = "\n\n---\n\n".join(
                [f"SOURCE FILE: {c['source_file']}\nCONTENT: {c['text']}" for c in valid_chunks]
            )

        user_prompt = (
            f"CONVERSATION HISTORY:\n{self.state.get_recent_history()}\n\n"
            f"EVIDENCE CHUNKS:\n{evidence_str}\n\n"
            f"USER QUESTION:\n{user_query}"
        )
        return self._call_llm_json(
            "grounded_generation", GROUNDED_GENERATOR_SYSTEM_PROMPT, user_prompt, call_logs
        )

    def ask(self, user_query: str) -> TurnRecord:
        call_logs: List[LLMCallLog] = []

        # 1. Query Transformation (Rewrite / Decompose)
        transformed_queries = self._transform_query(user_query, call_logs)

        # 2. Hybrid Retrieval (Vector + BM25 + RRF)
        candidates = self.retriever.search_multi_queries(transformed_queries, top_n=8)

        # 3. Cross-Encoder Re-Ranking
        reranked = self.reranker.rerank(user_query, candidates, top_k=Config.RERANK_TOP_K)

        # 4. Corrective Loop (Grade context, re-query if weak)
        crag_grade, rewritten_query = self._grade_context(user_query, reranked, call_logs)
        crag_retry_used = None

        if crag_grade == "WEAK" and rewritten_query:
            crag_retry_used = rewritten_query
            retry_candidates = self.retriever.search_multi_queries(
                transformed_queries + [rewritten_query], top_n=10
            )
            reranked = self.reranker.rerank(user_query, retry_candidates, top_k=Config.RERANK_TOP_K)

        # 5. Grounded Answer Generation with Citations / Abstention
        gen_output = self._generate_answer(user_query, reranked, call_logs)

        turn = TurnRecord(
            turn_id=len(self.state.turns) + 1,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            user_query=user_query,
            transformed_queries=transformed_queries,
            crag_grade=crag_grade,
            crag_retry_query=crag_retry_used,
            reranked_chunks=reranked,
            answer=gen_output.get("answer", ""),
            abstained=bool(gen_output.get("abstained", False)),
            citations=gen_output.get("citations", []),
            llm_calls=call_logs,
        )
        self.state.add_turn(turn)
        return turn