"""
Refresh the results block in README.md from the pipeline's own outputs.

Runs at the end of every nightly job, so the numbers in the README are always
the numbers the system produced last night. Everything between
<!-- RESULTS:START --> and <!-- RESULTS:END --> is rewritten; nothing else is.

    python src/update_readme.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
README = os.path.join(ROOT, "README.md")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _json(name):
    path = os.path.join(ROOT, "models", name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def nights_unattended():
    """Consecutive days, up to the latest, with a commit from the nightly job."""
    try:
        out = subprocess.run(
            ["git", "log", "--author=github-actions", "--format=%ad", "--date=format:%Y-%m-%d"],
            cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    import pandas as pd

    days = sorted(set(pd.to_datetime(out)))
    if not days:
        return None
    streak = 1
    for a, b in zip(reversed(days[:-1]), reversed(days[1:])):
        if (b - a).days != 1:
            break
        streak += 1
    return streak, days[-streak]


def test_count():
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
                             cwd=ROOT, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    m = re.search(r"(\d+) tests? collected", out)
    return int(m.group(1)) if m else None


def pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


def render():
    live, bt = _json("live_results.json") or {}, _json("horizon_results.json") or {}
    ev, vb = _json("analyst_eval.json"), _json("verifier_benchmark.json")
    o = live.get("overall")
    lines = ["| | |", "|---|---|"]
    if o:
        lines += [
            f"| **Live forecasts scored** | {live['n_forecasts_scored']} forecasts, "
            f"{live['n_points']:,} hours ({live['period']}) |",
            f"| **Live error (MAE)** | {o['MAE']:.2f} µg/m³: **{pct(o['skill_vs_persistence'])} lower "
            f"than persistence**, {pct(o['skill_vs_cams'])} lower than the CAMS forecast |",
        ]
        if live.get("days_beating_cams"):
            lines.append(f"| **Forecasts that beat CAMS** | {live['days_beating_cams']} |")
    if bt.get("mean_skill_vs_persistence") is not None:
        seasonal = bt.get("mean_skill_vs_seasonal")
        lines.append(
            f"| **Backtest** (5-fold rolling origin) | {pct(bt['mean_skill_vs_persistence'])} lower MAE "
            f"than persistence, {pct(seasonal)} than seasonal naive, "
            f"{pct(bt.get('mean_skill_vs_cams'))} than CAMS |")
        cov = [m.get("coverage") for m in (bt.get("intervals") or {}).values() if m.get("coverage")]
        if cov:
            iv = live.get("interval_80")
            live_part = (f", {iv['coverage']:.0%} over {iv['n_points']:,} live hours" if iv else "")
            lines.append(f"| **80% prediction range** | held the measured value "
                         f"{min(cov):.0%}–{max(cov):.0%} of the time in backtest{live_part} |")
    if ev:
        t = ev["totals"]
        caught = t.get("first_drafts_with_invented_numbers")
        lines.append(
            f"| **Assistant eval** ({t['n']} questions) | {pct(t['grounded_rate'])} of final answers "
            f"fully grounded, {pct(t['tool_selection_rate'])} right tool or correct decline"
            + (f"; verifier caught invented numbers in {caught} first drafts" if caught is not None else "")
            + " |")
    if vb:
        lines.append(
            f"| **Verifier stress test** | caught {vb['wrong_answers_caught']} of "
            f"{vb['wrong_answers_tested']} seeded wrong numbers ({pct(vb['catch_rate'])}); "
            f"blocked {vb['correct_answers_wrongly_blocked']} of {vb['correct_answers_tested']} correct ones |")
    nights = nights_unattended()
    if nights:
        lines.append(f"| **Unattended** | retrained every night for {nights[0]} nights in a row "
                     f"(since {nights[1]:%d %b %Y}) |")
    n_tests = test_count()
    if n_tests:
        lines.append(f"| **Tests** | {n_tests}, run before every retrain and on every push |")
    stamp = (live.get("generated_at") or "")[:10]
    return "\n".join(lines) + (f"\n\n<sub>Updated automatically by the nightly job · {stamp}</sub>"
                               if stamp else "")


def main():
    with open(README, encoding="utf-8") as f:
        text = f.read()
    if START not in text or END not in text:
        sys.exit("README has no results markers")
    block = f"{START}\n{render()}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    if new != text:
        with open(README, "w", encoding="utf-8") as f:
            f.write(new)
    print(render())


if __name__ == "__main__":
    main()
