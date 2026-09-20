"""Latency benchmark for the Recruta pipeline (run this on YOUR machine — it
needs a live Groq key).

    python bench.py                       # both graphs, 3 runs each, sample CV/job
    python bench.py --mode fast --runs 5
    python bench.py --cv my_cv.txt --job posting.txt --mode both
    GROQ_REASONING_EFFORT=medium python bench.py --mode fast
    python bench.py --fake-delay 0.3      # offline plumbing check, no API calls

What it measures, per run:
  * wall-clock total and the moment each node finished (so you can see
    "time to first score", which is what a UI actually waits for)
  * every individual LLM call: model, seconds, input/output tokens, errors
  * 429/retry counts and an effective tokens-per-minute figure compared to
    Groq's free-tier limit (8,000 TPM for gpt-oss models) — rate limiting is
    the most likely hidden source of latency, because a 429 makes a call wait
    and retry instead of failing fast.
Across runs: min / p50 / p95 / max wall time and the verdict against --target.

Note: a bench "run" makes real, billable API calls and counts against Groq's
daily limits (free tier: 200k tokens/day for gpt-oss). Use --pause to space
runs out, and keep --runs small.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

ROOT = Path(__file__).parent
GROQ_FREE_TPM = 8000


class CallRecorder(BaseCallbackHandler):
    """Collects one record per LLM call (thread-safe: nodes run in parallel)."""

    raise_error = False

    def __init__(self, t0: float):
        self.t0 = t0
        self.lock = threading.Lock()
        self.open: dict = {}
        self.calls: list[dict] = []
        self.retries = 0

    # -- start -----------------------------------------------------------------
    def _start(self, run_id, metadata, kwargs):
        model = (metadata or {}).get("ls_model_name") or (kwargs.get("invocation_params") or {}).get("model_name") or "?"
        with self.lock:
            self.open[run_id] = {"model": model, "start": time.perf_counter() - self.t0}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._start(run_id, metadata, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):
        self._start(run_id, metadata, kwargs)

    # -- end / error -----------------------------------------------------------
    def _finish(self, run_id, in_tok, out_tok, error=None):
        end = time.perf_counter() - self.t0
        with self.lock:
            rec = self.open.pop(run_id, {"model": "?", "start": end})
            rec.update(end=end, seconds=end - rec["start"], in_tokens=in_tok, out_tokens=out_tok, error=error)
            self.calls.append(rec)

    def on_llm_end(self, response, *, run_id, **kwargs):
        in_tok = out_tok = 0
        try:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None) or {}
            in_tok, out_tok = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        except Exception:  # noqa: BLE001 — non-chat generation or no usage info
            pass
        if not (in_tok or out_tok):
            tu = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            in_tok, out_tok = tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0)
        self._finish(run_id, in_tok, out_tok)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, 0, 0, error=f"{type(error).__name__}: {str(error)[:1500]}")

    def on_retry(self, retry_state, *, run_id, **kwargs):
        with self.lock:
            self.retries += 1


def _fake_llms(delay: float) -> None:
    """Offline mode: replace every LLM with a sleeping stub (plumbing check only —
    the numbers it prints are NOT Groq latencies)."""
    from langchain_core.runnables import RunnableLambda

    import schemas
    from agents import cv_matcher, interview_prep, job_analyzer, report_writer, scorer, supervisor

    def factory(schema, *, tier="content", temperature=0.2):
        def call(_p):
            time.sleep(delay)
            n = schema.__name__
            return {
                "RoutingDecision": lambda: schemas.RoutingDecision(next_agent="END", reasoning="x"),
                "JobRequirements": lambda: schemas.JobRequirements(
                    job_title="T", company_name="C", required_skills=["Python"], seniority_level="Mid",
                    years_of_experience_required="n/a", keywords=[], company_context="c"),
                "MatchAnalysis": lambda: schemas.MatchAnalysis(
                    match_score=70, matched_skills=["Python"], missing_skills=[], strengths=[],
                    weaknesses=[], projects_to_highlight=[]),
                "ScoringResult": lambda: schemas.ScoringResult(overall_score=80, requirement_scores=[], methodology="m"),
                "InterviewPrep": lambda: schemas.InterviewPrep(likely_questions=["q"], talking_points=["t"]),
                "FinalReportOutput": lambda: schemas.FinalReportOutput(short_summary="s", final_report="# r"),
            }[n]()

        return RunnableLambda(call)

    for mod in (supervisor, job_analyzer, cv_matcher, scorer, interview_prep, report_writer):
        mod.get_structured_llm = factory


def run_once(mode: str, cv: str, job: str) -> dict:
    if mode == "fast":
        from graph_fast import build_fast_graph as build
        state = {"cv_text": cv, "job_posting_text": job, "messages": [], "errors": {}}
    else:
        from graph import build_graph as build
        state = {"cv_text": cv, "job_posting_text": job, "messages": []}

    t0 = time.perf_counter()
    rec = CallRecorder(t0)
    timeline: list[tuple[float, str]] = []
    error = None
    final: dict = {}
    try:
        for chunk in build().stream(state, config={"callbacks": [rec]}, stream_mode="updates"):
            now = time.perf_counter() - t0
            for node, update in chunk.items():
                timeline.append((now, node))
                final.update(update or {})
    except Exception as exc:  # noqa: BLE001 — a failed run is still data
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
    total = time.perf_counter() - t0

    def first(node):
        return next((t for t, n in timeline if n == node), None)

    llm_secs = sorted(c["seconds"] for c in rec.calls)
    tokens = sum(c["in_tokens"] + c["out_tokens"] for c in rec.calls)
    return {
        "mode": mode,
        "total_s": total,
        "error": error,
        "timeline": timeline,
        "time_to_requirements_s": first("job_analyzer"),
        "time_to_score_s": first("scorer"),
        "llm_calls": rec.calls,
        "n_llm_calls": len(rec.calls),
        "llm_seconds_sum": sum(llm_secs),
        "retries": rec.retries,
        "rate_limited": sum(1 for c in rec.calls if c["error"] and "429" in c["error"]),
        "tokens_total": tokens,
        "effective_tpm": tokens / total * 60 if total else 0,
        "tokens_by_model": _tokens_by_model(rec.calls),
        "output": {k: final.get(k) for k in ("job_requirements", "match_analysis", "score_breakdown")},
        "score": _score(final),
        "scorer_fallback": _scorer_fell_back(final),
        "node_errors": final.get("errors") or {},
    }


def _tokens_by_model(calls: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in calls:
        out[c["model"]] = out.get(c["model"], 0) + c["in_tokens"] + c["out_tokens"]
    return out


def _score(final: dict):
    sb = final.get("score_breakdown")
    return sb.get("overall_score") if isinstance(sb, dict) else None


def _scorer_fell_back(final: dict) -> bool:
    """True when the run "succeeded" but the weighted scorer silently failed
    and the crude matched/missing ratio was used instead (a fast run that
    hides this looks great and is worthless)."""
    sb = final.get("score_breakdown")
    return not isinstance(sb, dict) or str(sb.get("methodology", "")).startswith("Fallback")


def _short_error(err: str) -> str:
    """One-line view; the full text (incl. failed_generation / retry-after) is in --json."""
    return err.replace("\n", " ")[:260]


def _pct(values: list[float], p: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def print_run(i: int, r: dict) -> None:
    status = "FAILED: " + r["error"] if r["error"] else "ok"
    print(f"\n--- {r['mode']} run {i}: total {r['total_s']:.1f}s ({status})")
    print("  node finished at:", "  ".join(f"{n}@{t:.1f}s" for t, n in r["timeline"]))
    for c in r["llm_calls"]:
        err = f"\n        ERROR {_short_error(c['error'])}" if c["error"] else ""
        print(f"    LLM {c['model']:<22} {c['start']:5.1f}s → {c['end']:5.1f}s  ({c['seconds']:4.1f}s)  "
              f"in={c['in_tokens']:<5} out={c['out_tokens']:<5}{err}")
    flags = []
    if r["scorer_fallback"]:
        flags.append("SCORER FELL BACK to crude ratio (score is NOT the weighted score)")
    if r["node_errors"]:
        flags.append(f"node errors: {list(r['node_errors'])}")
    if flags:
        print("  !! " + " | ".join(flags))
    print(f"  final score: {r['score']}")
    print(f"  {r['n_llm_calls']} LLM calls, sum of call time {r['llm_seconds_sum']:.1f}s "
          f"(vs wall {r['total_s']:.1f}s), retries={r['retries']}, 429s={r['rate_limited']}")
    if r["tokens_total"]:
        per_model = ", ".join(f"{m.split('/')[-1]}={t}" for m, t in r["tokens_by_model"].items())
        print(f"  tokens: {r['tokens_total']} total ({per_model})")
        # Groq limits are per model, per minute — compare each model's own total.
        over = [m.split("/")[-1] for m, t in r["tokens_by_model"].items() if t > GROQ_FREE_TPM]
        if over:
            print(f"  !! {', '.join(over)} alone used > {GROQ_FREE_TPM} tokens in this run "
                  "(free-tier per-minute limit): back-to-back runs will hit 429s")


def summarize(mode: str, runs: list[dict], target: float) -> dict:
    ok = [r for r in runs if not r["error"]]
    line = {"mode": mode, "runs": len(runs), "failed": len(runs) - len(ok),
            "scorer_fallbacks": sum(1 for r in ok if r["scorer_fallback"])}
    if ok:
        t = [r["total_s"] for r in ok]
        line.update(min=min(t), p50=statistics.median(t), p95=_pct(t, 0.95), max=max(t))
        sc = [r["time_to_score_s"] for r in ok if r["time_to_score_s"] is not None]
        line["p50_time_to_score"] = statistics.median(sc) if sc else None
        line["p50_llm_calls"] = statistics.median(r["n_llm_calls"] for r in ok)
    return line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["current", "fast", "both"], default="both")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cv", default=str(ROOT / "examples" / "sample_cv.txt"))
    ap.add_argument("--job", default=str(ROOT / "examples" / "sample_job_posting.txt"))
    ap.add_argument("--target", type=float, default=15.0, help="acceptable total seconds (default 15)")
    ap.add_argument("--pause", type=float, default=65.0, help="seconds between runs; >=60 lets the per-minute token window reset so you measure latency, not throttling (default 65)")
    ap.add_argument("--fake-delay", type=float, default=None, help="offline: stub LLMs that sleep this long")
    ap.add_argument("--json", default=None, help="write all raw results to this file")
    args = ap.parse_args()

    if args.fake_delay is not None:
        _fake_llms(args.fake_delay)
        print(f"[offline stub mode: {args.fake_delay}s per fake call — these are NOT Groq latencies]")
    else:
        from dotenv import load_dotenv

        load_dotenv()
        import os

        if not os.getenv("GROQ_API_KEY"):
            print("GROQ_API_KEY is not set (.env). Use --fake-delay for an offline plumbing check.")
            return 2
        print(f"models: {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')} (content), "
              f"reasoning_effort={os.getenv('GROQ_REASONING_EFFORT', 'medium')}")

    if args.fake_delay is None:
        from graph import build_graph  # noqa: F401  (import cost paid before timing)
        import graph_fast

        t = time.perf_counter()
        graph_fast.warm_up()
        print(f"warm-up (LLM client construction, paid once per process): {time.perf_counter() - t:.2f}s")
    cv, job = Path(args.cv).read_text(encoding="utf-8"), Path(args.job).read_text(encoding="utf-8")
    print(f"CV: {len(cv)} chars (~{len(cv)//4} tokens) | job posting: {len(job)} chars (~{len(job)//4} tokens)")

    modes = ["current", "fast"] if args.mode == "both" else [args.mode]
    results: dict[str, list[dict]] = {m: [] for m in modes}
    # Interleave modes so a slow minute on the API doesn't bias one of them.
    for i in range(1, args.runs + 1):
        for m in modes:
            r = run_once(m, cv, job)
            results[m].append(r)
            print_run(i, r)
            if args.pause:
                time.sleep(args.pause)

    print("\n=========== SUMMARY (seconds) ===========")
    print(f"{'mode':<9}{'runs':>5}{'fail':>5}{'min':>7}{'p50':>7}{'p95':>7}{'max':>7}{'p50 →score':>12}{'LLM calls':>11}")
    summaries = [summarize(m, rs, args.target) for m, rs in results.items()]
    for s in summaries:
        if "p50" not in s:
            print(f"{s['mode']:<9}{s['runs']:>5}{s['failed']:>5}   all runs failed")
            continue
        tts = f"{s['p50_time_to_score']:.1f}" if s.get("p50_time_to_score") is not None else "-"
        print(f"{s['mode']:<9}{s['runs']:>5}{s['failed']:>5}{s['min']:>7.1f}{s['p50']:>7.1f}{s['p95']:>7.1f}"
              f"{s['max']:>7.1f}{tts:>12}{s['p50_llm_calls']:>11.0f}")
        if s["scorer_fallbacks"]:
            print(f"          !! {s['scorer_fallbacks']}/{len(ok)} runs used the crude fallback score — their times are not comparable")
        verdict = "MEETS" if s["p95"] <= args.target else "MISSES"
        print(f"          → p95 {s['p95']:.1f}s {verdict} the {args.target:.0f}s target")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nraw results → {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
