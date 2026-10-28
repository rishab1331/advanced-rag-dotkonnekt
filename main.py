import argparse
from src.pipeline import AdvancedRAGPipeline

# 4-Turn Demo Conversation to prove all assignment criteria:
# Turn 1: Multi-part decomposition + Hybrid Retrieval + Citations
# Turn 2: Conversational memory (pronoun resolution: "they", "it")
# Turn 3: Specific technical retrieval (Corrective RAG & Vector DB design)
# Turn 4: Out-of-scope question to prove graceful Abstention (no hallucination)
DEMO_QUESTIONS = [
    "How does hybrid search work and why do we need document chunking?",
    "How does Reciprocal Rank Fusion combine those search results, and what happens if retrieval quality is weak?",
    "What are the main factors to consider in vector database design and OpenAI vs Google embeddings?",
    "What is the company's policy on remote work vacation days and laptop reimbursement?"
]

def run_demo(pipeline: AdvancedRAGPipeline, output_file: str = "sample_transcript.txt") -> None:
    print(f"\nIndexed {pipeline.total_chunks} chunks across your 15 documents.\n")
    print("=" * 80)
    print("RUNNING 4-TURN DEMO CONVERSATION")
    print("=" * 80)

    for q in DEMO_QUESTIONS:
        print(f"\nUser: {q}")
        turn = pipeline.ask(q)
        print(f"  -> Rewritten Sub-Queries: {turn.transformed_queries}")
        print(f"  -> CRAG Grade: {turn.crag_grade}" + (f" (Retry: '{turn.crag_retry_query}')" if turn.crag_retry_query else ""))
        print(f"  -> Abstained: {turn.abstained} | Citations: {turn.citations}")
        print(f"\nAssistant:\n{turn.answer}")
        print("-" * 80)

    transcript_text = pipeline.state.format_transcript()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    print(f"\nSaved complete multi-turn transcript & telemetry to: {output_file}")

def run_interactive(pipeline: AdvancedRAGPipeline) -> None:
    print("\n" + "=" * 75)
    print(f" Advanced RAG Q&A Service ({pipeline.total_chunks} chunks indexed)")
    print(" Commands: '/state' (JSON dump) | '/transcript' (Full log) | '/exit'")
    print("=" * 75)

    while True:
        try:
            user_input = input("\nAsk a question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "exit", "quit"):
            break
        if user_input.lower() == "/state":
            print("\n--- SESSION STATE JSON DUMP ---")
            print(pipeline.state.to_json())
            continue
        if user_input.lower() == "/transcript":
            print("\n" + pipeline.state.format_transcript())
            continue

        turn = pipeline.ask(user_input)
        print(f"\n[Rewritten Queries]: {turn.transformed_queries}")
        print(f"[CRAG Context Grade]: {turn.crag_grade}" + (f" -> Re-queried: '{turn.crag_retry_query}'" if turn.crag_retry_query else ""))
        print(f"[Abstained]: {turn.abstained} | [Citations]: {turn.citations}")
        print(f"\nAssistant:\n{turn.answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track C - Advanced RAG Q&A System")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a 4-turn example conversation and save sample_transcript.txt",
    )
    args = parser.parse_args()

    rag = AdvancedRAGPipeline()
    if args.demo:
        run_demo(rag)
    else:
        run_interactive(rag)