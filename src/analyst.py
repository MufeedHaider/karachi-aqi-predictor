"""
Forecast analyst: answers questions about the forecast without inventing numbers.

The LLM (Gemini, via function calling) can only read the pipeline's own outputs
through tools. Every number in its answer is then checked against what the tools
returned. An unverifiable answer is regenerated once, then withheld.

Needs GEMINI_API_KEY (optional GEMINI_MODEL). Tests use a scripted stand-in.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")

SYSTEM_PROMPT = """You answer questions about Karachi's PM2.5 and AQI forecast.

Rules:
- Get every fact from the tools. Never estimate, extrapolate or compute a new number.
- Quote numbers exactly as the tools return them (you may round to one decimal
  or convert a fraction like 0.457 to 45.7%).
- The forecast covers only the next 72 hours from its issue time. For anything
  further ahead, or any other city, say it is outside what this system forecasts.
- When accuracy matters to the answer, say how accurate the forecast has been
  (live track record first, backtest second).
- Mention the 80% range (p10 to p90) when it exists, so users see uncertainty.
- Health guidance: use the AQI category the tool gives; do not give medical advice
  beyond it.
- All times are Karachi local time. Use `now_karachi` from get_forecast to
  interpret "tonight", "tomorrow" or "this weekend". Never present an hour
  before now as upcoming.
- Reply in the user's language: English, Urdu or Roman Urdu.
- Be brief: two to five sentences.
"""


# --------------------------------------------------------------------- tools
# Plain functions with type hints and docstrings: the Gemini SDK turns these
# into function declarations automatically.

def _read_json(name):
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _forecast():
    path = os.path.join(DATA, "forecast_72hr.csv")
    return pd.read_csv(path, parse_dates=["timestamp"]) if os.path.exists(path) else None


def _now():
    """Current Karachi time (naive, like the data). Tests replace this."""
    return pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None).floor("h")


def get_forecast(hours_ahead: int = 72) -> dict:
    """PM2.5 and AQI forecast from the current hour onward, in Karachi time.

    Returns the current time, a summary (mean, peak and lowest hour) and the
    hourly rows for the next `hours_ahead` hours, including the 80% range.
    """
    fc = _forecast()
    if fc is None:
        return {"error": "no forecast has been published"}
    now = _now()
    upcoming = fc[fc["timestamp"] >= now]
    if upcoming.empty:
        return {"error": "the latest forecast has expired; no upcoming hours",
                "now_karachi": f"{now:%a %d %b %H:00}"}
    part = upcoming[upcoming["timestamp"] < now + pd.Timedelta(hours=max(1, int(hours_ahead)))]
    peak = part.loc[part["pm2_5_predicted"].idxmax()]
    low = part.loc[part["pm2_5_predicted"].idxmin()]
    anchor = fc["timestamp"].min() - pd.Timedelta(hours=1)
    cols = [c for c in ["timestamp", "pm2_5_predicted", "pm2_5_p10", "pm2_5_p90",
                        "aqi_predicted", "aqi_category", "cams_pm2_5"] if c in part.columns]
    rows = part[cols].copy()
    rows["timestamp"] = rows["timestamp"].dt.strftime("%a %d %b %H:00")
    return {
        "now_karachi": f"{now:%a %d %b %H:00}",
        "based_on_measurements_up_to": f"{anchor:%a %d %b %H:00}",
        "forecast_ends": f"{fc['timestamp'].max():%a %d %b %H:00}",
        "mean_pm2_5": round(float(part["pm2_5_predicted"].mean()), 1),
        "peak": {"time": f"{peak['timestamp']:%a %d %b %H:00}",
                 "pm2_5": float(peak["pm2_5_predicted"]),
                 "aqi": int(peak["aqi_predicted"]), "category": peak["aqi_category"]},
        "lowest": {"time": f"{low['timestamp']:%a %d %b %H:00}",
                   "pm2_5": float(low["pm2_5_predicted"]),
                   "aqi": int(low["aqi_predicted"]), "category": low["aqi_category"]},
        "hours": rows.to_dict(orient="records"),
    }


def get_live_track_record() -> dict:
    """How accurate the published forecasts have actually been, scored against
    what the ground monitors later measured, compared with CAMS and persistence."""
    live = _read_json("live_results.json")
    return live or {"error": "no live track record yet"}


def get_backtest_accuracy() -> dict:
    """Rolling-origin backtest accuracy at every reported lead time (1-72 hours):
    MAE, skill vs CAMS and persistence, and 80% range coverage."""
    res = _read_json("horizon_results.json")
    if not res:
        return {"error": "no backtest results"}
    intervals = res.get("intervals") or {}
    by_lead = {}
    for h, m in sorted(res.get("horizons", {}).items(), key=lambda kv: int(kv[0])):
        row = {k: m.get(k) for k in ["MAE", "cams_MAE", "skill_vs_cams",
                                       "skill_vs_persistence", "R2"] if k in m}
        if intervals.get(h, {}).get("coverage") is not None:
            row["range_80_coverage"] = intervals[h]["coverage"]
        by_lead[f"{h}h"] = row
    return {
        "evaluation": res.get("evaluation"),
        "mean_skill_vs_cams": res.get("mean_skill_vs_cams"),
        "mean_skill_vs_persistence": res.get("mean_skill_vs_persistence"),
        "by_lead_time": by_lead,
    }


def get_forecast_drivers() -> dict:
    """Top features behind the 24-hour forecast, by mean absolute SHAP value."""
    path = os.path.join(MODELS, "shap_importance.csv")
    if not os.path.exists(path):
        return {"error": "no SHAP results"}
    shap = pd.read_csv(path).head(8)
    return {"model": "24-hour XGBoost", "top_features": shap.round(2).to_dict(orient="records")}


def get_system_status() -> dict:
    """Data freshness, whether the last retrain passed its accuracy gate, and input drift."""
    res = _read_json("horizon_results.json") or {}
    gate = _read_json("gate_report.json") or {}
    drift = _read_json("drift_report.json") or {}
    return {
        "forecast_issued_from": res.get("forecast_issued_at"),
        "model_generated_at": res.get("generated_at"),
        "degraded_inputs": res.get("degraded_inputs", []),
        "accuracy_gate": gate.get("decision"),
        "input_drift": drift.get("overall"),
        "drifted_inputs": [k for k, v in (drift.get("features") or {}).items()
                           if v.get("level") == "major"],
    }


TOOLS = [get_forecast, get_live_track_record, get_backtest_accuracy,
         get_forecast_drivers, get_system_status]


# ----------------------------------------------------------------- grounding

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")

# Small integers carry no factual claim on their own ("in 3 hours", "80% range").
TRIVIAL = {str(i) for i in range(0, 13)} | {"24", "48", "72", "80", "100"}


def numbers_in(obj) -> set[float]:
    """Every number that appears in a tool output, in the forms a writer may use."""
    found = set()
    text = json.dumps(obj, default=str)
    for m in NUMBER.findall(text):
        v = float(m.replace(",", ""))
        found.update({v, round(v, 1), round(v), round(v * 100, 1), round(v * 100)})
    return found


def ungrounded_numbers(answer: str, tool_outputs: list) -> list[str]:
    """Numbers in the answer that no tool returned (within rounding)."""
    allowed = set()
    for out in tool_outputs:
        allowed |= numbers_in(out)
    bad = []
    for token in NUMBER.findall(answer):
        if token in TRIVIAL:
            continue
        v = float(token.replace(",", ""))
        if not any(abs(v - a) <= 0.051 for a in allowed):
            bad.append(token)
    return bad


# --------------------------------------------------------------------- agent

@dataclass
class Answer:
    text: str
    tools_called: list = field(default_factory=list)
    tool_outputs: list = field(default_factory=list)
    grounded: bool = True
    ungrounded: list = field(default_factory=list)
    attempts: int = 1


def _recording(tools):
    """Wrap tools so every call and its output is captured for grounding."""
    calls, outputs = [], []

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            out = fn(*args, **kwargs)
            calls.append({"name": fn.__name__, "args": kwargs or list(args)})
            outputs.append(out)
            return out
        return inner

    return [wrap(t) for t in tools], calls, outputs


class GeminiLLM:
    """Gemini with automatic function calling."""

    def __init__(self, api_key=None, model=None):
        from google import genai  # imported here so tests need no SDK

        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

    def run(self, question, tools, retries=3):
        """Free-tier keys hit per-minute limits (HTTP 429): wait and retry."""
        import time

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=6),
        )
        for attempt in range(retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=question, config=config)
                return response.text or ""
            except Exception as exc:
                busy = any(s in str(exc) for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
                if not busy or attempt == retries:
                    raise
                wait = re.search(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", str(exc))
                delay = float(next(g for g in wait.groups() if g)) if wait else 5 * (attempt + 1)
                time.sleep(min(delay + 1, 30))


def ask(question: str, llm=None, tools=TOOLS) -> Answer:
    llm = llm or GeminiLLM()
    wrapped, calls, outputs = _recording(tools)

    text = llm.run(question, wrapped)
    bad = ungrounded_numbers(text, outputs)
    attempts = 1
    if bad:
        attempts = 2
        retry = (f"{question}\n\nYour previous answer contained numbers that no tool "
                 f"returned: {', '.join(bad)}. Answer again using only numbers from "
                 "tool results.")
        text = llm.run(retry, wrapped)
        bad = ungrounded_numbers(text, outputs)

    if bad:
        # Never show an ungrounded claim. Fall back to what the tools said.
        text = ("I couldn't produce an answer I can fully back with the forecast data, "
                "so here is the data itself:\n\n"
                + "\n".join(json.dumps(o, default=str)[:600] for o in outputs[-2:]))

    return Answer(text=text, tools_called=[c["name"] for c in calls],
                  tool_outputs=outputs, grounded=not bad, ungrounded=bad,
                  attempts=attempts)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What will the air quality be like tonight?"
    a = ask(q)
    print(a.text)
    print(f"\n[tools: {', '.join(a.tools_called) or 'none'} | grounded: {a.grounded} "
          f"| attempts: {a.attempts}]")
