"""
Score the analyst on evals/analyst_eval.jsonl:
grounded answers, first-try grounding, right tool (or a correct decline), latency.

    GEMINI_API_KEY=... python src/run_analyst_evals.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyst import ask  # noqa: E402

EVALS = os.path.join(os.path.dirname(__file__), "..", "evals", "analyst_eval.jsonl")
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "analyst_eval.json")


def load_cases(path=EVALS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge(case, answer):
    expected = case.get("expect_tools", [])
    if expected:
        right_tool = any(t in answer.tools_called for t in expected)
    else:
        words = [w.lower() for w in case.get("expect_any", [])]
        right_tool = any(w in answer.text.lower() for w in words) if words else True
    return {
        "grounded": answer.grounded,
        "first_try": answer.grounded and answer.attempts == 1,
        "right_tool": right_tool,
    }


# Space questions out so a free-tier key stays under its per-minute limit.
DELAY_S = float(os.environ.get("EVAL_DELAY", "8"))


def run(llm=None, cases=None, delay=None):
    cases = cases or load_cases()
    delay = DELAY_S if delay is None else delay
    rows = []
    for i, case in enumerate(cases):
        if i and delay:
            time.sleep(delay)
        start = time.perf_counter()
        try:
            answer = ask(case["question"], llm=llm)
            verdict = judge(case, answer)
            err = None
        except Exception as exc:  # a crash is a failed case, not a crashed eval
            answer, verdict, err = None, {"grounded": False, "first_try": False,
                                          "right_tool": False}, repr(exc)
        rows.append({
            "id": case["id"],
            "question": case["question"],
            **verdict,
            "passed": all(verdict.values()),
            "tools_called": answer.tools_called if answer else [],
            "ungrounded": answer.ungrounded if answer else [],
            "answer": answer.text if answer else None,
            "error": err,
            "latency_s": round(time.perf_counter() - start, 2),
        })

    n = len(rows)
    totals = {
        "n": n,
        "pass_rate": round(sum(r["passed"] for r in rows) / n, 3),
        "grounded_rate": round(sum(r["grounded"] for r in rows) / n, 3),
        "first_try_grounded_rate": round(sum(r["first_try"] for r in rows) / n, 3),
        "tool_selection_rate": round(sum(r["right_tool"] for r in rows) / n, 3),
        "median_latency_s": sorted(r["latency_s"] for r in rows)[n // 2],
    }
    return {"totals": totals, "cases": rows}


def main():
    result = run()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    t = result["totals"]
    print(f"Analyst eval on {t['n']} questions")
    print(f"  pass rate               {t['pass_rate']:.0%}")
    print(f"  grounded (final answer) {t['grounded_rate']:.0%}")
    print(f"  grounded on first try   {t['first_try_grounded_rate']:.0%}")
    print(f"  right tool / declines   {t['tool_selection_rate']:.0%}")
    print(f"  median latency          {t['median_latency_s']} s")
    for r in result["cases"]:
        if not r["passed"]:
            print(f"  FAIL {r['id']}: tools={r['tools_called']} ungrounded={r['ungrounded']} {r['error'] or ''}")


if __name__ == "__main__":
    main()
