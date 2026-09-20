"""Find out which scorer request variants Groq actually accepts.

Background: in the benchmark the scorer works on gpt-oss-120b but the
gpt-oss-20b fallback fails with `400 Failed to validate JSON`, and Groq's
docs say `reasoning_format` is NOT supported for gpt-oss models (they use
`include_reasoning`). Instead of guessing, this sends the same small scoring
request under each variant and prints status, latency, tokens, and the FULL
error text (including `failed_generation`).

    python probe_scorer.py                    # all variants (~14 calls, spaced 20s)
    python probe_scorer.py --pause 30 --only 20b

Costs roughly 2-3k tokens per call.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

from agents.scorer import SYSTEM_PROMPT, USER_PROMPT  # noqa: E402
from schemas import ScoringResult  # noqa: E402

ROOT = Path(__file__).parent
JOB_REQ = {
    "job_title": "Backend Engineer", "seniority_level": "Mid",
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "LangGraph", "REST API design"],
    "years_of_experience_required": "Not specified in the posting",
    "nice_to_have_skills": ["Kubernetes"], "keywords": [], "company_context": "AI startup",
}


def variants(only: str | None):
    models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    efforts = ["low", "medium", "high"]
    # (label, extra ChatGroq kwargs, model_kwargs)
    shapes = [
        ("effort only", {}, {}),
        ("effort + reasoning_format=parsed (old round-8 code)", {"reasoning_format": "parsed"}, {}),
        ("effort + include_reasoning=false (per docs)", {}, {"include_reasoning": False}),
    ]
    for model, effort, (label, kw, mkw) in itertools.product(models, efforts, shapes):
        if only and only not in model:
            continue
        # keep the matrix small: full effort sweep only for the "effort only" shape
        if label != "effort only" and effort == "medium":
            continue
        yield model, effort, label, kw, mkw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=20.0)
    ap.add_argument("--only", default=None, help="substring of model id, e.g. 20b")
    args = ap.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY missing (.env)")
        return 2

    cv = (ROOT / "examples" / "sample_cv.txt").read_text(encoding="utf-8")
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
    inputs = {"job_requirements": json.dumps(JOB_REQ), "cv_text": cv,
              "match_analysis": "(not provided — derive every verdict from the CV alone)"}

    rows = []
    for model, effort, label, kw, mkw in variants(args.only):
        chat = ChatGroq(model=model, temperature=0, max_retries=0, reasoning_effort=effort,
                        model_kwargs=mkw, **kw)
        chain = prompt | chat.with_structured_output(ScoringResult, method="json_schema", strict=True,
                                                      include_raw=True)
        t0 = time.perf_counter()
        status, detail, out_tok = "OK", "", 0
        try:
            res = chain.invoke(inputs)
            raw = res.get("raw")
            out_tok = (getattr(raw, "usage_metadata", None) or {}).get("output_tokens", 0)
            if res.get("parsing_error"):
                status, detail = "PARSE-FAIL", str(res["parsing_error"])[:600]
            else:
                detail = f"score={res['parsed'].overall_score}"
        except Exception as exc:  # noqa: BLE001
            status, detail = type(exc).__name__, str(exc)[:1200].replace("\n", " ")
        dt = time.perf_counter() - t0
        rows.append((model.split("/")[-1], effort, label, status, dt, out_tok, detail))
        print(f"\n[{model.split('/')[-1]} | effort={effort} | {label}]\n   {status}  {dt:.1f}s  out_tokens={out_tok}\n   {detail}")
        time.sleep(args.pause)

    print("\n================ SUMMARY ================")
    print(f"{'model':<14}{'effort':<8}{'status':<18}{'sec':>6}{'out_tok':>9}  variant")
    for m, e, lab, st, dt, ot, _ in rows:
        print(f"{m:<14}{e:<8}{st:<18}{dt:>6.1f}{ot:>9}  {lab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
