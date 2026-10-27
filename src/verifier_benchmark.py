"""
Stress test for the assistant's number verifier. No API key needed.

The eval (run_analyst_evals.py) shows how often the real model invents a number.
This shows how reliably the verifier would catch one. It takes the real tool
outputs, writes answers that quote true numbers (the verifier must let them
through) and answers where one number has been nudged by 2-50% (the verifier
must block them), and counts both.

A nudged number can land, by chance, on another value that a tool really
returned. The verifier cannot tell those apart, so they are reported
separately as "collisions" rather than hidden.

    python src/verifier_benchmark.py
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyst  # noqa: E402
from analyst import NUMBER, TRIVIAL, ungrounded_numbers  # noqa: E402

OUT = os.path.join(analyst.MODELS, "verifier_benchmark.json")
SIZES = [0.02, 0.05, 0.10, 0.25, 0.50]


def tool_outputs():
    fc = analyst._forecast()
    if fc is not None:  # read the forecast as of its first hour, so it is never "expired"
        start = fc["timestamp"].min()
        analyst._now = lambda: start
    return [t() for t in analyst.TOOLS]


def claim_values(outputs):
    """Numbers a writer would plausibly quote: not ids, not tiny counts."""
    vals = set()
    for out in outputs:
        for token in NUMBER.findall(json.dumps(out, default=str)):
            v = float(token.replace(",", ""))
            if token in TRIVIAL or v == 0 or abs(v) > 100000:
                continue
            vals.add(v)
    return sorted(vals)


def render(v, rng):
    """Write a number the way an answer would."""
    if 0 < abs(v) < 1 and rng.random() < 0.5:
        return f"{v * 100:.1f}%"
    return f"{v:.1f}" if rng.random() < 0.5 else f"{v:g}"


def run(n=500, seed=7):
    """Each test answer is checked against the outputs of 1-2 tools, as a real
    answer is: the tool the number came from, sometimes plus one more."""
    rng = random.Random(seed)
    per_tool = tool_outputs()
    values = []
    for i, out in enumerate(per_tool):
        values += [(i, v) for v in claim_values([out])]

    false_flags = 0
    by_size = {s: {"tested": 0, "caught": 0, "collisions": 0} for s in SIZES}
    for _ in range(n):
        i, v = rng.choice(values)
        used = {i} | ({rng.randrange(len(per_tool))} if rng.random() < 0.5 else set())
        outputs = [per_tool[j] for j in sorted(used)]
        if ungrounded_numbers(f"The value is {render(v, rng)}.", outputs):
            false_flags += 1

        size = rng.choice(SIZES)
        wrong = v * (1 + size * rng.choice([-1, 1]))
        text = render(wrong, rng)
        if not ungrounded_numbers(f"The value is {text}.", [{"x": v}]):
            continue  # rounding undid the nudge; it is not a wrong number any more
        slot = by_size[size]
        slot["tested"] += 1
        if ungrounded_numbers(f"The value is {text}.", outputs):
            slot["caught"] += 1
        else:
            slot["collisions"] += 1

    tested = sum(s["tested"] for s in by_size.values())
    caught = sum(s["caught"] for s in by_size.values())
    return {
        "what": "verifier tested on real tool outputs with seeded wrong numbers",
        "true_values": len(values),
        "correct_answers_tested": n,
        "correct_answers_wrongly_blocked": false_flags,
        "wrong_answers_tested": tested,
        "wrong_answers_caught": caught,
        "catch_rate": round(caught / tested, 3) if tested else None,
        "by_error_size": {
            f"{int(s * 100)}%": {**v, "catch_rate": round(v["caught"] / v["tested"], 3)
                                 if v["tested"] else None}
            for s, v in by_size.items()
        },
        "note": "misses are collisions: the wrong number equals another true value in the data",
    }


def main():
    result = run()
    os.makedirs(analyst.MODELS, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Verifier stress test on {result['true_values']} real values")
    print(f"  wrong numbers caught: {result['wrong_answers_caught']}/{result['wrong_answers_tested']} "
          f"({result['catch_rate']:.1%})")
    print(f"  correct numbers wrongly blocked: {result['correct_answers_wrongly_blocked']}"
          f"/{result['correct_answers_tested']}")
    for size, v in result["by_error_size"].items():
        print(f"  off by {size:>4}: caught {v['caught']}/{v['tested']}  collisions {v['collisions']}")


if __name__ == "__main__":
    main()
