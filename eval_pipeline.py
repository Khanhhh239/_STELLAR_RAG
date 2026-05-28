"""
STELLAR-RAG — Evaluation Pipeline

Reads prompts from eval_prompts.txt, runs each through the full RAG pipeline,
and produces detailed per-prompt metrics + a summary report.

Metrics captured
────────────────
Retrieval quality
  num_dense_hits      Number of dense-retrieval chunks after RRF fusion
  num_graph_hits      Number of graph-retrieval hits
  avg_dense_score     Mean RRF/reranker score of retrieved chunks
  max_dense_score     Best score in the retrieved set
  unique_sources      Number of distinct source documents
  context_length      Total chars in the context window sent to the LLM
  self_rag_triggered  Whether Self-RAG triggered a second wider retrieval
  self_rag_quality    Context quality score before Self-RAG decision [0–1]
  hyde_used           Whether HyDE (hypothetical doc) was applied

Query processing
  query_complexity    simple / compound / complex (router decision)
  num_variants        Number of query variants from the expander
  expansion_variants  The actual variant strings

Answer quality
  answer_length       Character count of the LLM's answer
  grounding_overlap   Token overlap between answer and context [0–1]
                      Good answers have ≥ 0.10; < 0.10 may indicate hallucination
  has_hallucination   Whether speculative markers were detected in the answer
  hallucination_text  The specific marker phrase (if any)
  guardrail_action    allow / warn / block from input guardrail

Performance
  total_ms            End-to-end wall-clock time including LLM generation

Usage
─────
  cd <project-root>
  .venv\\Scripts\\python eval_pipeline.py [prompts_file]

  Default prompts file: eval_prompts.txt
  Output files:         eval_report.json, eval_log.txt
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any

# ── Force UTF-8 stdout on Windows (cp1252 can't print Vietnamese)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent import Agent
from guardrail import OutputGuardrail, _no_accent


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    prompt: str

    # Guardrail
    guardrail_action: str = "allow"
    guardrail_reason: str = ""

    # Query processing
    query_complexity:   str        = "simple"
    num_variants:       int        = 1
    expansion_variants: list[str]  = field(default_factory=list)

    # Retrieval
    num_dense_hits:     int        = 0
    num_graph_hits:     int        = 0
    avg_dense_score:    float      = 0.0
    max_dense_score:    float      = 0.0
    unique_sources:     list[str]  = field(default_factory=list)
    context_length:     int        = 0
    self_rag_triggered: bool       = False
    self_rag_quality:   float      = 1.0
    hyde_used:          bool       = False

    # Answer quality
    answer:              str   = ""
    answer_length:       int   = 0
    grounding_overlap:   float = 0.0
    has_hallucination:   bool  = False
    hallucination_text:  str   = ""

    # Retrieval ranking quality
    ndcg_at_10:  float = 0.0   # nDCG@10 — answer-grounding proxy relevance
    qdap_alpha:  float = -1.0  # QDAP-S predicted α (-1 = RRF / not applicable)

    # Performance
    total_ms: float = 0.0

    # Detailed trace (for log file)
    retrieved_chunks_log: list[str] = field(default_factory=list)
    context_preview:      str       = ""


# ─────────────────────────────────────────────────────────────────────────────
# Grounding / hallucination helpers (mirrors OutputGuardrail logic)
# ─────────────────────────────────────────────────────────────────────────────

_MIN_GROUNDING = 0.10

_HALLUCINATION_PATTERNS: list = [
    re.compile(r"theo\s+(toi|minh)\s+(biet|nghi|hieu|suy\s*nghi|ung\s*doan)", re.I),
    re.compile(r"toi\s+(nghi|doan|tuong|cho\s+rang|uoc\s+tinh)\b", re.I),
    re.compile(r"\bi\s+(think|believe|guess|assume|suppose|imagine)\b", re.I),
    re.compile(r"\b(probably|likely|perhaps|maybe|possibly)\b.{0,20}\b(is|are|was|were|will)\b",
               re.I | re.S),
    re.compile(r"i[''`]m\s+not\s+(sure|certain|100%|fully)", re.I),
    re.compile(r"\b(co\s*le\s*la|duong\s*nhu\s*la|co\s*the\s*la)\b", re.I),
    re.compile(r"khong\s*chinh\s*xac\s*nhung", re.I),
    re.compile(r"\b(as\s+far\s+as\s+i\s+know|to\s+my\s+knowledge|i\s+recall)\b", re.I),
    re.compile(r"(based\s+on\s+my\s+training|my\s+knowledge\s+cutoff)", re.I),
]


def _compute_grounding(answer: str, context: str) -> float:
    if not context.strip():
        return 1.0   # no context available → no check
    a_norm = _no_accent(answer)
    c_norm = _no_accent(context)
    a_tok = set(re.findall(r"\w{3,}", a_norm))
    if not a_tok:
        return 1.0
    c_tok = set(re.findall(r"\w{3,}", c_norm))
    return round(len(a_tok & c_tok) / len(a_tok), 4)


def _detect_hallucination(answer: str) -> str:
    norm = _no_accent(answer)
    for pat in _HALLUCINATION_PATTERNS:
        m = pat.search(norm)
        if m:
            return m.group(0)[:80]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# nDCG@k — ranking quality metric (paper Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ndcg_at_k(hits: list[dict], answer: str, k: int = 10) -> float:
    """
    Compute nDCG@k using answer-grounding as a graded relevance proxy.

    Since we have no external ground-truth labels, we approximate relevance
    by measuring how much each retrieved chunk's text overlaps with the LLM's
    final answer — chunks that contributed to the answer are marked relevant.

    Graded relevance scale (matches paper's 0–3 scheme):
      3  ≥ 5 content tokens shared with the answer  (strongly relevant)
      2  3–4 shared tokens                           (moderately relevant)
      1  1–2 shared tokens                           (weakly relevant)
      0  no shared tokens                            (not relevant)

    DCG@k  = Σ_{i=1}^{k}  rel_i / log₂(i+1)
    IDCG@k = DCG of ideal ordering (sort rel descending)
    nDCG@k = DCG@k / IDCG@k

    Returns 0.0 if no chunks are retrieved or the answer is empty.
    """
    if not hits or not answer.strip():
        return 0.0

    a_norm = _no_accent(answer)
    a_tok  = set(re.findall(r"\w{3,}", a_norm))
    if not a_tok:
        return 0.0

    rel: list[int] = []
    for h in hits[:k]:
        text   = h.get("text") or h.get("text_preview") or ""
        c_norm = _no_accent(text)
        c_tok  = set(re.findall(r"\w{3,}", c_norm))
        n      = len(a_tok & c_tok)
        rel.append(3 if n >= 5 else 2 if n >= 3 else 1 if n >= 1 else 0)

    # Pad to exactly k positions with 0 (missing hits = irrelevant)
    while len(rel) < k:
        rel.append(0)

    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel, reverse=True)))

    return round(dcg / idcg, 4) if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Log formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_bar(label: str, value: float, width: int = 30) -> str:
    filled = int(value * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"{label:20s} |{bar}| {value:.3f}"


def _render_prompt_log(idx: int, r: EvalResult) -> str:
    lines: list[str] = []
    sep = "═" * 72

    lines.append(f"\n{sep}")
    lines.append(f"PROMPT {idx:02d}: {r.prompt}")
    lines.append(sep)

    lines.append("\n── GUARDRAIL ──────────────────────────────────────────────")
    lines.append(f"  Action : {r.guardrail_action.upper()}")
    if r.guardrail_reason:
        lines.append(f"  Reason : {r.guardrail_reason}")

    lines.append("\n── QUERY PROCESSING ────────────────────────────────────────")
    lines.append(f"  Complexity : {r.query_complexity}")
    lines.append(f"  Variants   : {r.num_variants}")
    for i, v in enumerate(r.expansion_variants):
        lines.append(f"    [{i}] {v}")

    lines.append("\n── RETRIEVAL ───────────────────────────────────────────────")
    lines.append(f"  Dense hits     : {r.num_dense_hits}")
    lines.append(f"  Graph hits     : {r.num_graph_hits}")
    lines.append(f"  Avg score      : {r.avg_dense_score:.4f}")
    lines.append(f"  Max score      : {r.max_dense_score:.4f}")
    lines.append(f"  Unique sources : {len(r.unique_sources)}  →  {', '.join(r.unique_sources) or '—'}")
    lines.append(f"  Context length : {r.context_length:,} chars")
    lines.append(f"  HyDE           : {'YES' if r.hyde_used else 'NO'}")
    q = r.self_rag_quality
    lines.append(f"  Self-RAG       : quality={q:.3f}  triggered={r.self_rag_triggered}")

    if r.retrieved_chunks_log:
        lines.append("\n── TOP RETRIEVED CHUNKS ────────────────────────────────────")
        for cl in r.retrieved_chunks_log:
            lines.append(cl)

    lines.append("\n── CONTEXT SENT TO LLM (first 600 chars) ──────────────────")
    preview = r.context_preview or "(empty)"
    lines.append(f"  {preview[:600]}")
    if len(r.context_preview) > 600:
        lines.append(f"  … [{len(r.context_preview):,} chars total]")

    lines.append("\n── ANSWER ──────────────────────────────────────────────────")
    lines.append(f"  {r.answer[:800]}")
    if len(r.answer) > 800:
        lines.append(f"  … [{len(r.answer)} chars total]")

    lines.append("\n── METRICS ─────────────────────────────────────────────────")
    overlap_label = "GOOD" if r.grounding_overlap >= _MIN_GROUNDING else "LOW (possible hallucination)"
    ndcg_label    = "GOOD" if r.ndcg_at_10 >= 0.70 else ("OK" if r.ndcg_at_10 >= 0.40 else "LOW")
    lines.append(f"  Answer length      : {r.answer_length} chars")
    lines.append(f"  nDCG@10            : {r.ndcg_at_10:.4f}  ({ndcg_label})")
    lines.append(f"  QDAP-S α           : {r.qdap_alpha:.4f}" if r.qdap_alpha >= 0 else "  QDAP-S α           : N/A (RRF fusion)")
    lines.append(f"  Grounding overlap  : {r.grounding_overlap:.3f}  ({overlap_label})")
    lines.append(f"  Hallucination      : {'YES — «' + r.hallucination_text + '»' if r.has_hallucination else 'NONE'}")
    lines.append(f"  Total time         : {r.total_ms:.0f} ms")
    lines.append("")

    return "\n".join(lines)


def _render_summary(results: list[EvalResult]) -> str:
    lines: list[str] = []
    sep = "═" * 72

    lines.append(f"\n{sep}")
    lines.append(f"SUMMARY  ({len(results)} prompts)")
    lines.append(sep)

    # Header
    lines.append(f"  {'#':>3}  {'nDCG':>6}  {'α':>5}  {'Grd':>5}  {'H':>2}  "
                 f"{'Src':>3}  {'Ctx':>5}  {'SRq':>5}  {'ms':>6}  Prompt")
    lines.append("  " + "─" * 78)

    for i, r in enumerate(results, 1):
        h     = "Y" if r.has_hallucination else "N"
        alpha = f"{r.qdap_alpha:>5.3f}" if r.qdap_alpha >= 0 else "  N/A"
        lines.append(
            f"  {i:>3}  {r.ndcg_at_10:>6.4f}  {alpha}  "
            f"{r.grounding_overlap:>5.3f}  {h:>2}  "
            f"{len(r.unique_sources):>3}  "
            f"{r.context_length:>5}  {r.self_rag_quality:>5.3f}  "
            f"{r.total_ms:>6.0f}  "
            f"{r.prompt[:46]}"
        )

    lines.append(
        "\n  Columns: nDCG=nDCG@10  α=QDAP-S_alpha  Grd=grounding_overlap  "
        "H=hallucination\n"
        "           Src=unique_sources  Ctx=context_chars  "
        "SRq=self_rag_quality  ms=latency"
    )

    # Aggregate stats
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    ndcg = [r.ndcg_at_10 for r in results]
    grd  = [r.grounding_overlap for r in results]
    hall = [r for r in results if r.has_hallucination]
    blk  = [r for r in results if r.guardrail_action == "block"]
    lat  = [r.total_ms for r in results]
    srq  = [r.self_rag_quality for r in results]
    srt  = [r for r in results if r.self_rag_triggered]
    # QDAP alpha stats (exclude -1.0 placeholders for RRF fallback)
    alphas = [r.qdap_alpha for r in results if r.qdap_alpha >= 0]

    lines.append(f"\n  Average nDCG@10           : {avg(ndcg):.4f}")
    lines.append(f"  Average grounding overlap : {avg(grd):.3f}")
    lines.append(f"  Average Self-RAG quality  : {avg(srq):.3f}")
    if alphas:
        lines.append(
            f"  QDAP-S α  avg/min/max     : "
            f"{avg(alphas):.3f} / {min(alphas):.3f} / {max(alphas):.3f}"
        )
    lines.append(f"  Self-RAG triggered        : {len(srt)}/{len(results)} prompts")
    lines.append(f"  Hallucination detected    : {len(hall)}/{len(results)} prompts")
    lines.append(f"  Guardrail blocks          : {len(blk)}/{len(results)} prompts")
    lines.append(f"  Average latency           : {avg(lat):.0f} ms")
    lines.append(f"  Min / Max latency         : {min(lat):.0f} / {max(lat):.0f} ms")

    lines.append(f"\n{sep}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(path: str) -> list[str]:
    prompts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def run_evaluation(
    prompts_file: str = "eval_prompts.txt",
    report_file:  str = "eval_report.json",
    log_file:     str = "eval_log.txt",
) -> list[EvalResult]:
    # Resolve paths relative to this script
    base = os.path.dirname(os.path.abspath(__file__))
    prompts_file = os.path.join(base, prompts_file)
    report_file  = os.path.join(base, report_file)
    log_file     = os.path.join(base, log_file)

    # ── Load prompts ──────────────────────────────────────────────────────
    if not os.path.exists(prompts_file):
        print(f"[ERROR] Prompts file not found: {prompts_file}")
        sys.exit(1)

    prompts = load_prompts(prompts_file)
    if not prompts:
        print("[ERROR] No prompts found in file (all lines are empty or comments).")
        sys.exit(1)

    print(f"[EVAL] Loaded {len(prompts)} prompts from {prompts_file}")
    print("[EVAL] Initialising agent… (loading index if present)")

    # ── Init agent ────────────────────────────────────────────────────────
    agent = Agent()

    results: list[EvalResult] = []
    log_parts: list[str]       = []

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx:02d}/{len(prompts)}] {prompt[:70]}")

        r = EvalResult(prompt=prompt)
        t0 = time.perf_counter()

        # ── Run agent ─────────────────────────────────────────────────────
        try:
            answer, _turn_id = agent.answer(prompt)
        except Exception as exc:
            answer = f"[EVAL ERROR] {exc}"
            print(f"  [!] Exception: {exc}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        r.total_ms = round(elapsed_ms, 1)

        # ── Read debug_info populated by agent ───────────────────────────
        di: dict[str, Any] = agent.debug_info

        r.guardrail_action   = di.get("guardrail_action", "allow")
        r.guardrail_reason   = di.get("guardrail_reason", "")
        r.query_complexity   = di.get("query_complexity", "simple")
        r.expansion_variants = di.get("expansion_variants", [prompt])
        r.num_variants       = len(r.expansion_variants)
        r.num_dense_hits     = di.get("num_dense_hits", 0)
        r.num_graph_hits     = di.get("num_graph_hits", 0)
        r.avg_dense_score    = di.get("avg_dense_score", 0.0)
        r.max_dense_score    = di.get("max_dense_score", 0.0)
        r.unique_sources     = di.get("unique_sources", [])
        r.context_length     = di.get("context_length", 0)
        r.self_rag_triggered = di.get("self_rag_triggered", False)
        r.self_rag_quality   = di.get("self_rag_quality", 1.0)
        r.hyde_used          = di.get("hyde_used", False)
        context              = di.get("context", "")

        r.answer        = answer
        r.answer_length = len(answer)

        # ── Compute answer quality metrics ────────────────────────────────
        r.grounding_overlap  = _compute_grounding(answer, context)
        halluc               = _detect_hallucination(answer)
        r.has_hallucination  = bool(halluc)
        r.hallucination_text = halluc

        # ── nDCG@10 — ranking quality (answer-grounding proxy) ────────────
        dense_hits_for_ndcg  = di.get("dense_hits", [])
        r.ndcg_at_10         = compute_ndcg_at_k(dense_hits_for_ndcg, answer, k=10)

        # ── QDAP-S alpha (first hit that carries it, else -1) ─────────────
        r.qdap_alpha = next(
            (float(h.get("qdap_alpha", -1.0)) for h in dense_hits_for_ndcg
             if "qdap_alpha" in h),
            -1.0,
        )

        # ── Build chunk log for human review ─────────────────────────────
        dense_hits = di.get("dense_hits", [])
        r.retrieved_chunks_log = []
        for rank, h in enumerate(dense_hits[:8], 1):
            src    = h.get("source", "?")
            page   = h.get("page", "?")
            sec    = (h.get("section") or "")[:50]
            score  = h.get("score", 0.0)
            text   = (h.get("text") or "")[:120].replace("\n", " ")
            r.retrieved_chunks_log.append(
                f"  [{rank}] score={score:.4f} | {src} | p{page}"
                + (f" | {sec}" if sec else "")
                + f"\n       \"{text}\""
            )

        r.context_preview = context[:1200] if context else ""

        # ── Print quick summary to terminal ──────────────────────────────
        grd_label  = "OK"  if r.grounding_overlap >= _MIN_GROUNDING else "LOW"
        ndcg_label = "GOOD" if r.ndcg_at_10 >= 0.70 else ("OK" if r.ndcg_at_10 >= 0.40 else "LOW")
        alpha_str  = f"α={r.qdap_alpha:.3f}" if r.qdap_alpha >= 0 else "α=RRF"
        print(
            f"  ✓ {elapsed_ms:.0f}ms"
            f" | nDCG@10={r.ndcg_at_10:.4f} [{ndcg_label}]"
            f" | {alpha_str}"
            f" | grd={r.grounding_overlap:.3f} [{grd_label}]"
            f" | hall={'Y' if r.has_hallucination else 'N'}"
            f" | dense={r.num_dense_hits} | ctx={r.context_length}c"
            f" | complexity={r.query_complexity}"
        )

        results.append(r)
        log_parts.append(_render_prompt_log(idx, r))

    # ── Summary ──────────────────────────────────────────────────────────
    summary = _render_summary(results)
    print(summary)
    log_parts.append(summary)

    # ── Save JSON report ─────────────────────────────────────────────────
    report_data = [asdict(r) for r in results]
    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump(report_data, fh, ensure_ascii=False, indent=2)
    print(f"\n[EVAL] JSON report → {report_file}")

    # ── Save detailed log ────────────────────────────────────────────────
    with open(log_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_parts))
    print(f"[EVAL] Detail log  → {log_file}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prompts_arg = sys.argv[1] if len(sys.argv) > 1 else "eval_prompts.txt"
    run_evaluation(prompts_file=prompts_arg)
