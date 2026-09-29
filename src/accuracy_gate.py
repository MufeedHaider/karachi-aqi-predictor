"""
Accuracy gate and model registry.

After retraining, the new model must still beat CAMS and persistence, and must
not be clearly worse than the live model (skill vs CAMS may drop at most 5
points, 24-hour MAE may rise at most 15%). If it fails, nothing publishes and
yesterday's forecast stays live. Interval coverage is reported as a warning only.
Set GATE_OVERRIDE=1 to publish anyway.

Every run is appended to models/registry.jsonl (code SHA, metrics, decision).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd

RESULTS = "models/horizon_results.json"
REGISTRY = "models/registry.jsonl"
REPORT = "models/gate_report.json"

MAX_SKILL_DROP = 0.05          # absolute, on mean skill vs CAMS
MAX_MAE_RISE = 0.15            # relative, on 24-hour MAE
COVERAGE_BAND = (0.70, 0.90)   # for a nominal 80% interval


def previous_results(path=RESULTS):
    """The metrics of the currently published model, from git HEAD."""
    try:
        raw = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            check=True, capture_output=True, text=True,
        ).stdout
        return json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return None


def _mae_24(results):
    h = results.get("horizons", {}).get("24")
    return h.get("MAE") if h else None


def evaluate(new, old=None):
    """Return blocking checks as (name, passed, detail)."""
    checks = []

    cams = new.get("mean_skill_vs_cams")
    checks.append(("beats CAMS", cams is not None and cams > 0,
                   f"mean skill vs CAMS {cams:+.1%}" if cams is not None else "missing"))

    pers = new.get("mean_skill_vs_persistence")
    checks.append(("beats persistence", pers is not None and pers > 0,
                   f"mean skill vs persistence {pers:+.1%}" if pers is not None else "missing"))

    if old:
        old_cams = old.get("mean_skill_vs_cams")
        if cams is not None and old_cams is not None:
            drop = old_cams - cams
            checks.append(("no regression vs CAMS", drop <= MAX_SKILL_DROP,
                           f"{old_cams:+.1%} -> {cams:+.1%} (drop {drop:+.1%}, "
                           f"limit {MAX_SKILL_DROP:.0%})"))
        new24, old24 = _mae_24(new), _mae_24(old)
        if new24 is not None and old24:
            rise = new24 / old24 - 1
            checks.append(("no regression at 24h", rise <= MAX_MAE_RISE,
                           f"MAE {old24:.2f} -> {new24:.2f} ({rise:+.1%}, "
                           f"limit {MAX_MAE_RISE:.0%})"))

    return checks


def warnings(new):
    """Non-blocking checks, same shape as `evaluate`."""
    out = []
    lo, hi = COVERAGE_BAND
    for h, m in sorted((new.get("intervals") or {}).items(), key=lambda kv: int(kv[0])):
        cov = m.get("coverage")
        if cov is not None:
            out.append((f"interval coverage {h}h", lo <= cov <= hi,
                        f"{cov:.1%} (target 80%, warn outside {lo:.0%}-{hi:.0%})"))
    return out


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def registry_entry(new, checks, passed, overridden, warns=()):
    return {
        "recorded_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "code_sha": git_sha(),
        "forecast_issued_at": new.get("forecast_issued_at"),
        "evaluation": new.get("evaluation"),
        "mean_skill_vs_cams": new.get("mean_skill_vs_cams"),
        "mean_skill_vs_persistence": new.get("mean_skill_vs_persistence"),
        "mean_skill_vs_seasonal": new.get("mean_skill_vs_seasonal"),
        "mae_24h": _mae_24(new),
        "interval_coverage": {h: m.get("coverage")
                              for h, m in (new.get("intervals") or {}).items()},
        "gate": "pass" if passed else ("override" if overridden else "blocked"),
        "failed_checks": [name for name, ok, _ in checks if not ok],
        "warnings": [name for name, ok, _ in warns if not ok],
    }


def main():
    if not os.path.exists(RESULTS):
        sys.exit(f"{RESULTS} missing: run forecast_model.py first")
    with open(RESULTS) as f:
        new = json.load(f)
    old = previous_results()

    checks = evaluate(new, old)
    warns = warnings(new)
    passed = all(ok for _, ok, _ in checks)
    overridden = not passed and os.environ.get("GATE_OVERRIDE") == "1"

    print("Accuracy gate" + ("" if old else " (no previous model at HEAD: absolute checks only)"))
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<26} {detail}")
    for name, ok, detail in warns:
        print(f"  [{'ok  ' if ok else 'WARN'}] {name:<26} {detail}")

    entry = registry_entry(new, checks, passed, overridden, warns)
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(entry) + "\n")
    with open(REPORT, "w") as f:
        json.dump({"decision": entry["gate"],
                   "checks": [{"name": n, "passed": ok, "detail": d}
                              for n, ok, d in checks],
                   "warnings": [{"name": n, "passed": ok, "detail": d}
                                for n, ok, d in warns]}, f, indent=2)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### Accuracy gate: {entry['gate'].upper()}\n\n| check | result | detail |\n|---|---|---|\n")
            for name, ok, detail in checks:
                f.write(f"| {name} | {'pass' if ok else '**FAIL**'} | {detail} |\n")
            for name, ok, detail in warns:
                f.write(f"| {name} (warning) | {'ok' if ok else '**WARN**'} | {detail} |\n")

    if passed:
        print("Gate passed: new model may publish.")
    elif overridden:
        print("Gate FAILED but GATE_OVERRIDE=1: publishing anyway.")
    else:
        print("Gate FAILED: blocking publish. Yesterday's forecast stays live.")
        sys.exit(1)


if __name__ == "__main__":
    main()
