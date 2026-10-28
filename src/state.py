import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any

@dataclass
class LLMCallLog:
    stage: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_preview: str

@dataclass
class TurnRecord:
    turn_id: int
    timestamp: str
    user_query: str
    transformed_queries: List[str]
    crag_grade: str
    crag_retry_query: str | None
    reranked_chunks: List[Dict[str, Any]]
    answer: str
    abstained: bool
    citations: List[str]
    llm_calls: List[LLMCallLog]

@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: f"session-{int(datetime.now().timestamp())}")
    turns: List[TurnRecord] = field(default_factory=list)

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)

    def get_recent_history(self, max_turns: int = 3) -> str:
        """Returns recent conversation turns so the LLM can resolve pronouns."""
        if not self.turns:
            return "No previous conversation."
        lines = []
        for t in self.turns[-max_turns:]:
            lines.append(f"User (Turn {t.turn_id}): {t.user_query}")
            lines.append(f"Assistant (Turn {t.turn_id}): {t.answer}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Inspectable JSON state dump."""
        return json.dumps(asdict(self), indent=2)

    def format_transcript(self) -> str:
        """Printable human-readable transcript across all turns."""
        out = [
            "=" * 80,
            f"ADVANCED RAG SESSION TRANSCRIPT | ID: {self.session_id}",
            "=" * 80,
        ]
        for t in self.turns:
            out.append(f"\n--- TURN {t.turn_id} ({t.timestamp}) ---")
            out.append(f"User Query: {t.user_query}")
            out.append(f"Decomposed/Rewritten Queries: {t.transformed_queries}")
            out.append(
                f"CRAG Context Grade: {t.crag_grade}"
                + (f" -> Re-queried with: '{t.crag_retry_query}'" if t.crag_retry_query else "")
            )
            out.append("Top Re-ranked Chunks:")
            for c in t.reranked_chunks:
                out.append(
                    f"  * [{c['source_file']}] ({c['chunk_id']}) | Rerank: {c['rerank_score']} | RRF: {c['rrf_score']}"
                )
            out.append(f"Abstained: {t.abstained} | Citations: {t.citations}")
            out.append(f"\nAssistant Answer:\n{t.answer}")
            out.append("\nLLM Call Telemetry:")
            for call in t.llm_calls:
                out.append(
                    f"  - [{call.stage}] model={call.model} | latency={call.latency_ms}ms | tokens={call.total_tokens} (in={call.prompt_tokens}, out={call.completion_tokens})"
                )
        out.append("\n" + "=" * 80)
        return "\n".join(out)