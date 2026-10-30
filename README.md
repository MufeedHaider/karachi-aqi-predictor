# Karachi AQI Predictor

[![Tests](https://github.com/MufeedHaider/karachi-aqi-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/MufeedHaider/karachi-aqi-predictor/actions/workflows/tests.yml)
[![Nightly retrain](https://github.com/MufeedHaider/karachi-aqi-predictor/actions/workflows/retrain.yml/badge.svg)](https://github.com/MufeedHaider/karachi-aqi-predictor/actions/workflows/retrain.yml)
[![Live app](https://img.shields.io/badge/live-dashboard-ff4b4b?logo=streamlit&logoColor=white)](https://karachi-aqi-predictor-8pjtbx9mwerlftrzvp3ckg.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)

Forecasts Karachi's **measured** PM2.5 and AQI 1–72 hours ahead, retrains itself
every night, scores every forecast it publishes against what the monitors later
measured, and answers questions about the forecast in English, Urdu or Roman
Urdu through a Gemini assistant that is not allowed to invent numbers.

**[→ Live dashboard](https://karachi-aqi-predictor-8pjtbx9mwerlftrzvp3ckg.streamlit.app)**:
the *Scoreboard* page shows each day's forecast next to what actually happened.

## Results

<!-- RESULTS:START -->
| | |
|---|---|
| **Live forecasts scored** | 57 forecasts, 3,987 hours (2026-09-04 to 2026-10-30) |
| **Live error (MAE)** | 6.85 µg/m³: **34% lower than persistence**, 42% lower than the CAMS forecast |
| **Forecasts that beat CAMS** | 54 of 57 |
| **Backtest** (5-fold rolling origin) | 20% lower MAE than persistence, 20% than seasonal naive, 47% than CAMS |
| **80% prediction range** | held the measured value 84%–86% of the time in backtest, 77% over 2,115 live hours |
| **Assistant eval** (30 questions) | 87% of final answers fully grounded, 87% right tool or correct decline; verifier caught invented numbers in 4 first drafts |
| **Verifier stress test** | caught 435 of 491 seeded wrong numbers (89%); blocked 0 of 500 correct ones |
| **Unattended** | retrained every night for 133 nights in a row (since 19 Jun 2026) |
| **Tests** | 84, run before every retrain and on every push |

<sub>Updated automatically by the nightly job · 2026-10-30</sub>
<!-- RESULTS:END -->

Persistence ("PM2.5 stays as it is now") is the honest headline baseline.
CAMS is ECMWF's operational forecast, and the live comparison uses the CAMS
forecast published at the same time; but CAMS is a 45 km global model with no
local bias correction, so beating it is expected rather than remarkable.

## How it works

```
OpenAQ (10 Karachi ground monitors)      Open-Meteo (weather + CAMS forecasts)
                  └────────────────┬─────────────────┘
                     48 features, nothing from the future
                                   ↓
           72 XGBoost models, one per lead hour, each predicting the change
                                   ↓
     rolling-origin backtest → accuracy gate → forecast + 80% range → Streamlit
                                   ↓
     next nights: forecast scored against measurements → live scoreboard
```

- **Target is measured PM2.5, not CAMS.** CAMS reads 37% low over Karachi across
  the year and 2.4× low in December. It is used as an input instead, together
  with how far it currently sits from the monitors, and the model learns to
  correct it. SHAP confirms the top two drivers are *measured minus CAMS (now)*
  and *CAMS forecast for the target hour*.
- **72 direct models.** Model *h* predicts *t+h* from what is known at *t*. A
  recursive version fed its own output back and produced a flat line.
- **Predict the change, not the level.** Tree models cannot extrapolate a
  near-identity mapping, so they predict ΔPM2.5 and add it to the current reading.

## Evaluation and what was fixed

The current version came out of an audit of an earlier one that reported a
near-perfect R². That score was leakage: two features (`pm2_5[t]/pm10[t]` and
`wind_speed[t]/pm2_5[t]`) let the same-hour target be rebuilt by division.

| Problem found | Fix |
|---|---|
| Target leakage through two ratio features | Label moved to *t+h*; leakage tests, including one that reproduces the old bug |
| Trained to imitate CAMS, which misses Karachi's winter | Target is measured PM2.5; CAMS became an input and a benchmark |
| One 80/20 split put the whole test set in summer | 5-fold rolling-origin backtest, every season tested |
| Recursive forecast came out flat | 72 direct per-horizon models with real forecast inputs |
| Interpolation and full-data outlier caps peeked at the future | Forward-fill only; caps from training rows only |
| A dead monitor feed shown as live | `src/freshness.py` rejects stale readings |
| Gaps in the AQI breakpoints, two EPA standards in use | One AQI module with a sweep test |

Backtests use analysed weather and CAMS rather than the forecasts that existed
at the time, so they are slightly optimistic. The live scoreboard removes that
caveat.

## Forecast assistant

Gemini (function calling) can only read the pipeline's outputs through five
read-only tools: forecast, live track record, backtest, SHAP drivers and system
status. A verifier then checks every number in the answer against what the tools
returned, matching the precision the answer used. An answer with a number no tool
produced is regenerated once, then withheld.

- **Eval:** 30 questions in `evals/analyst_eval.jsonl` (English, Urdu, Roman Urdu,
  out-of-scope, prompt injection, false premises), run nightly with a real key.
- **Verifier stress test:** `src/verifier_benchmark.py` writes answers from real
  tool outputs, nudges one number by 2–50%, and counts how many the verifier
  blocks. Misses are collisions, where the wrong number happens to equal another
  real value in the same data.

## Production safeguards

| | |
|---|---|
| **Accuracy gate** | A retrain that is worse than the live model does not publish |
| **Model registry** | `models/registry.jsonl`: one line per nightly version, with metrics and gate result |
| **80% prediction ranges** | Conformalised quantile regression, coverage checked in backtest and live |
| **Drift page** | This week's inputs against the same weeks last year |
| **Tests** | Run on every push (`tests.yml`) and before every retrain (`retrain.yml`) |

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests -q
streamlit run dashboard/app.py          # uses the committed data/ and models/
```

With Docker:

```bash
docker build -t karachi-aqi .                         # dashboard image
docker run -p 8501:8501 karachi-aqi                   # → http://localhost:8501

docker build --target pipeline -t karachi-aqi-pipeline .
docker run -e OPENAQ_API_KEY=... karachi-aqi-pipeline # tests, then the full nightly pipeline
```

The full pipeline needs a free [OpenAQ key](https://explore.openaq.org/register)
in `.env` as `OPENAQ_API_KEY`. The assistant needs `GEMINI_API_KEY`.

## Limitations

- One year of data and one winter so far; the live record covers September onward.
- Ground monitors are low-cost optical sensors (city median of about 6 per hour).
- Forecasts go live about 5 hours after their latest measurement, because the
  CAMS archive the pipeline needs lags real time.
- The verifier checks numbers, not wording: an answer can be grounded and still
  phrased poorly.

## License

MIT
