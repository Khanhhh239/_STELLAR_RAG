"""
STELLAR-RAG v4 (improve_RAG) — Interactive chat entry point.

Usage:
    python app.py

Starts an interactive terminal chat loop backed by the full EHRAG + HybGRAG
pipeline (hypergraph diffusion, critic validation, QDAP-S fusion).
"""
from __future__ import annotations

import io
import os
import sys

# ── Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent import Agent
from config import settings


def main() -> None:
    settings.ensure_dirs()
    agent = Agent()
    print("STELLAR-RAG v4 (EHRAG + HybGRAG) — Type 'exit' to quit.")
    print("After each answer you can rate it 1-5 to reinforce quality.\n")

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        ans, turn_id = agent.answer(q)
        print(f"\nAgent> {ans}\n")

        rating = input("Rate (1-5, Enter to skip)> ").strip()
        if rating in {"1", "2", "3", "4", "5"}:
            note = input("Optional note> ").strip()
            agent.memory.add_feedback(
                turn_id=turn_id,
                user_query=q,
                assistant_answer=ans,
                reward=int(rating),
                note=note,
            )
            # Online QDAP-S update: map 1-5 rating to [-1, +1] reward signal
            # 5 → +1.0 (perfect — reinforce the α blend used)
            # 3 →  0.0 (neutral — no update)
            # 1 → -1.0 (wrong — push toward balanced α = 0.5)
            qdap_reward = (int(rating) - 3) / 2.0
            agent.update_qdap_feedback(qdap_reward)
            print("Feedback saved.")


if __name__ == "__main__":
    main()
