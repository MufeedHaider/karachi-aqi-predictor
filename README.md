# 🌫️ Karachi AQI Predictor

### Forecasting Karachi's *measured* PM2.5 up to 72 hours ahead — and beating the Copernicus model at it

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square&logo=streamlit)
![Tests](https://img.shields.io/badge/tests-39%20passing-brightgreen?style=flat-square)
![Skill](https://img.shields.io/badge/vs%20CAMS-%2B44.1%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
[![Live app](https://img.shields.io/badge/live-dashboard-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://karachi-aqi-predictor-8pjtbx9mwerlftrzvp3ckg.streamlit.app)

An end-to-end MLOps pipeline that forecasts PM2.5 in Karachi from 1 to 72 hours
ahead. It trains on **measurements from the city's ground monitor network**, not
on model output, and it is benchmarked against the operational forecast from
CAMS — the Copernicus Atmosphere Monitoring Service, run by ECMWF.

Across all lead times it reduces CAMS's error by **44.1%**.

**[→ Live dashboard](https://karachi-aqi-predictor-8pjtbx9mwerlftrzvp3ckg.streamlit.app)** — updated daily by the automated retrain.

---

## The problem this project actually solves

Most air-quality projects for cities like Karachi train on reanalysis data from
a global atmospheric model, because that data is free and complete. This one did
too, at first. Then the model was checked against what Karachi's monitors
actually measured over the same year:

| Month | Measured PM2.5 | CAMS says | CAMS error |
|---|---|---|---|
| Dec 2025 | **91.4** | 37.8 | **2.4× too low** |
| Nov 2025 | 76.4 | 40.9 | 1.9× too low |
| Feb 2026 | 64.2 | 33.4 | 1.9× too low |
| Jun 2026 | 24.2 | 23.5 | accurate |
| Aug 2026 | 22.3 | 21.9 | accurate |

CAMS is a 45 km global grid. Over Karachi it misses the entire winter pollution
season — the months when air quality is actually dangerous — and it reads 37%
low across the year overall. It also flattens the daily cycle: measured PM2.5
peaks at **54 µg/m³ around midnight** as the nocturnal boundary layer collapses
and traps emissions, while CAMS shows 31 and barely moves.

A model trained to predict CAMS would inherit all of that and report it as
accuracy.

**So CAMS was demoted from target to feature.** The model now predicts measured
PM2.5, and receives CAMS as one input among many — because the simulation does
carry real information about transport, chemistry and regional dust. Alongside
it the model sees how far CAMS currently sits from the monitors, and learns to
correct it.

That combination is what the SHAP analysis confirms is happening:

| Feature | Mean \|SHAP\| |
|---|---|
| **Measured minus CAMS (now)** | 7.71 |
| **CAMS forecast for target hour** | 4.91 |
| Forecast wind speed | 3.59 |
| CAMS PM10 | 2.06 |
| Measured PM2.5 3h mean | 1.93 |

The top two features are the physics prior and the correction to it.

---

## Results

Every figure below comes from a **rolling-origin backtest**: the model is
retrained at five points in time and scored only on the window after each cut.
Training never sees its own future, and every season appears in a test set.

A single 80/20 chronological split would have put the entire test set in June to
September — Karachi's calm, low-pollution season, where CAMS happens to be
accurate. That split flattered the model by 20% and never tested December once.

### Accuracy by lead time

| Lead | Model MAE | Persistence | **CAMS** | vs persistence | **vs CAMS** | R² |
|---|---|---|---|---|---|---|
| 1 hour | **2.89** | 3.13 | 9.82 | +7.6% | **+70.6%** | 0.865 |
| 3 hours | **4.90** | 6.04 | 9.81 | +18.8% | **+50.0%** | 0.640 |
| 6 hours | **5.84** | 7.98 | 9.76 | +26.8% | **+40.2%** | 0.526 |
| 12 hours | **5.78** | 8.10 | 9.72 | +28.7% | **+40.6%** | 0.519 |
| 24 hours | **5.84** | 6.59 | 9.66 | +11.4% | **+39.5%** | 0.518 |
| 48 hours | **6.12** | 7.55 | 9.60 | +18.9% | **+36.2%** | 0.492 |
| 72 hours | **6.51** | 7.84 | 9.51 | +17.0% | **+31.5%** | 0.367 |

**Mean skill: +44.1% against CAMS, +18.5% against persistence.**

Two baselines, because each answers a different question. Persistence ("nothing
changes") is hard to beat at short leads on a smooth hourly series — it is the
sanity check. CAMS is the real competition: an operational forecast from a
national meteorological agency, using the same lead times.

### Algorithms, at the 24-hour horizon

| Model | MAE | R² | vs persistence | vs CAMS |
|---|---|---|---|---|
| **XGBoost ✓** | **5.90** | **0.512** | **+10.4%** | **+38.9%** |
| Random Forest | 6.04 | 0.488 | +8.4% | +37.5% |
| Ridge Regression | 8.20 | 0.111 | −24.5% | +15.1% |
| *Persistence* | *6.59* | *0.335* | *0.0%* | *+31.8%* |
| *CAMS forecast* | *9.66* | *−0.121* | *−46.6%* | *0.0%* |
| *Hourly climatology* | *29.25* | *−5.339* | *−344%* | *−203%* |

XGBoost is selected at the 24-hour horizon, which is the lead time people plan
around. At the 1-hour horizon Random Forest is marginally ahead (2.82 vs 2.92) —
close enough to be run-to-run noise, and not worth shipping two algorithms for.

Ridge loses to persistence — the relationship is not linear, and saying so is
more useful than omitting the row. Hourly climatology fails badly because with
one year of data a seasonal climatology cannot be built for months the training
period has never seen; hour-of-day alone cannot represent Karachi's seasonal
swing from 22 to 91 µg/m³.

### Per-season detail, reported in full

24-hour horizon, one row per backtest fold:

| Test window | Observed mean | MAE | vs persistence | vs CAMS |
|---|---|---|---|---|
| Feb 21 – Apr 01 | 39.1 | 10.65 | +14.5% | +35.7% |
| Apr 01 – May 10 | 33.1 | 6.83 | +18.4% | +44.3% |
| May 10 – Jun 18 | 24.1 | 5.30 | **−13.2%** | +35.8% |
| Jun 18 – Jul 27 | 24.8 | 3.49 | +7.5% | +38.9% |
| Jul 27 – Sep 03 | 22.7 | 3.22 | +11.7% | +41.2% |

**The model loses to persistence in one fold** — mid-May to mid-June, when the
sea breeze settles in and PM2.5 goes quiet, so "nothing changes" becomes very
hard to improve on. It still beats CAMS by 36% in that window. Four folds out of
five is an honest result; a model that won everywhere would be a sign that
something was leaking.

---

## How it works

**One direct model per lead time.** Seventy-two XGBoost regressors. Model *h*
sees features observed at time *t* and predicts directly at *t+h* — nothing is
fed back into itself.

**Each predicts the change, not the level.** Gradient-boosted trees cannot
extrapolate a near-identity mapping, so asking for the absolute level makes them
lose to persistence at short leads. The models output Δ PM2.5, added to the
current measurement.

**Each gets real forecast inputs for the target hour** — the Open-Meteo weather
forecast and the CAMS PM2.5 forecast. Both genuinely exist at forecast time.

> **Backtesting caveat, stated plainly:** backtests substitute *analysed*
> weather and CAMS for *forecast* weather and CAMS, because neither archive
> stores what was predicted at the time. Real forecasts carry their own error,
> so live accuracy will run somewhat below these figures. The CAMS baseline is
> handicapped identically, so the comparison between the two stays fair.

```
Open-Meteo (CAMS + weather)          OpenAQ (Karachi ground monitors)
            ↓                                        ↓
    src/fetch_data.py                    src/ground_truth.py
            └──────────────┬─────────────────────────┘
                           ↓
             src/feature_engineering.py     48 features · EPA AQI · no lookahead
                           ↓
                src/train_model.py          3 algorithms vs 3 baselines
                           ↓
              src/forecast_model.py         72 direct models → backtest → forecast
                           ↓
              src/explain_model.py          SHAP on the 24-hour model
                           ↓
                dashboard/app.py            Streamlit, all figures read from disk
                           ↓
              .github/workflows             tests, then full retrain, daily 00:00 UTC
```

`src/aqi.py` is the single source of truth for the EPA scale.
`src/evaluation.py` owns the baselines and the rolling-origin backtest.

---

## Data

| Source | Role |
|---|---|
| **OpenAQ** — 10 Karachi monitors | **Target.** Measured PM2.5, city median |
| Open-Meteo / CAMS archive | Feature: simulated pollutants + weather |
| Open-Meteo forecast | Feature: weather expected at the target hour |
| Open-Meteo / CAMS forecast | Feature **and** benchmark |

Ground truth is 9,535 hourly readings from 2025-07-31 onward, from monitors at
NED University, Aga Khan University, the Urban Resource Center, WWF-Pakistan and
others. Median of **6 stations per hour**; 99.3% hourly coverage.

> **The honest caveat about the sensors.** Most of these are low-cost optical
> monitors — they report pm1 and particle counts, the signature of that
> hardware — and such units are known to over-read PM2.5 in humid conditions.
> Two things guard against it. The pipeline never trusts one station: it takes
> the median across all reporting monitors for each hour, so a few bad units
> cannot move the number. And the discrepancy with CAMS was checked against
> humidity directly — if it were a humidity artefact, the gap would widen as
> humidity rose. It does not. The measured/CAMS ratio is 1.8 in dry air and 1.7
> in humid air; it tracks *season*, not moisture. The gap is real.

The project originally used the WAQI feed from the US Consulate station. That
station stopped reporting in **March 2025** and the API kept serving its last
value, so the dashboard displayed an eighteen-month-old number as current air
quality. It has been removed, and `src/freshness.py` now rejects any reading
that lags the record by more than six hours.

---

## Setup

```bash
git clone https://github.com/MufeedHaider/karachi-aqi-predictor.git
cd karachi-aqi-predictor
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Ground-truth data needs a free [OpenAQ key](https://explore.openaq.org/register)
in a `.env` file:

```
OPENAQ_API_KEY=your_key_here
```

Run the pipeline:

```bash
python -m pytest tests/ -q
python src/fetch_data.py
python src/ground_truth.py --history 400
python src/feature_engineering.py
python src/train_model.py
python src/forecast_model.py
python src/explain_model.py
streamlit run dashboard/app.py
```

The repo ships `data/forecast_72hr.csv`, `data/recent_history.csv` and
`data/ground_history.csv`, so the dashboard renders on a fresh clone before you
run anything.

### Deployment

Live at **[karachi-aqi-predictor-8pjtbx9mwerlftrzvp3ckg.streamlit.app](https://karachi-aqi-predictor-8pjtbx9mwerlftrzvp3ckg.streamlit.app)**, deployed on Streamlit
Community Cloud from `dashboard/app.py`.
Note `dashboard/requirements.txt`: Streamlit looks for a dependency file beside
the entrypoint before the repository root, and the dashboard needs only
streamlit, pandas and plotly. The root file carries xgboost, shap and
scikit-learn for training and CI; installing those on the hosted app wasted
hundreds of megabytes and got it CPU-throttled on the first deploy.

The hosted app reads the artefacts committed by the nightly retrain, so it
updates once a day rather than continuously.

---

## AQI scale (US EPA, 2024 revision)

| AQI | Category | PM2.5 (µg/m³) |
|---|---|---|
| 0–50 | 🟢 Good | 0.0–9.0 |
| 51–100 | 🟡 Moderate | 9.1–35.4 |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | 35.5–55.4 |
| 151–200 | 🔴 Unhealthy | 55.5–125.4 |
| 201–300 | 🟣 Very Unhealthy | 125.5–225.4 |
| 301–500 | 🟤 Hazardous | 225.5+ |

Bands print as `9.1`–`35.4` but are implemented as contiguous intervals on upper
bounds. Implementing the printed bounds literally leaves a gap between every
band — see below.

---

## Correctness

`pytest tests/` — **39 tests**, run in CI before any retrain is allowed to commit.

- **`test_aqi.py`** sweeps the whole concentration range in 0.05 µg/m³ steps and
  asserts nothing falls between two bands; checks band edges against the 2024
  table; asserts monotonicity; asserts missing pollutants are skipped rather
  than treated as zero.
- **`test_no_leakage.py`** asserts the label is always measured PM2.5 at
  *t+horizon*, that target timestamps are exactly that many hours later, that no
  feature correlates above 0.999 with the label, and that outlier caps come from
  training rows only. One test deliberately reproduces the original leak to
  confirm the detection is sensitive to the bug it exists to catch.
- **`test_freshness.py`** replays the exact stale reading that was being
  displayed as live.

---

## Rebuild notes

This project was audited and substantially rebuilt. An earlier version reported
R² 0.978 and MAE 0.78 µg/m³. Those numbers were real arithmetic on an invalid
setup, and they are recorded here rather than quietly deleted.

| Issue | Detail | Resolution |
|---|---|---|
| **Target leakage** | Trained on `pm2_5[t]` while `pm_ratio = pm2_5[t]/pm10[t]` and `wind_dispersion = wind_speed[t]/pm2_5[t]` were inputs. The target was recoverable from the features to machine precision. | Label moved to *t+h*. Leakage guards in the trainer and in tests. |
| **Predicting a simulation** | Trained on CAMS output, which reads 37% low over Karachi and 2.4× low in December — so the model learned to reproduce that error. | Target is now measured PM2.5 from ground monitors. CAMS became a feature and the benchmark. |
| **Flattering test split** | A single 80/20 split put the whole test set in the calm summer season; December was never tested. | Rolling-origin backtest, 5 folds, every season tested. |
| **Flat forecasts** | Recursive prediction with every exogenous input frozen produced a line varying 2 µg/m³ over 72 hours. | 72 direct per-horizon models with real forecast inputs. |
| **Train/serve skew** | The three highest-importance features meant different things at fit time and forecast time. | Direct forecasting removes the feedback loop entirely. |
| **Fabricated AQI values** | Breakpoints written as literal `(54, 154)` pairs left gaps; PM10 of 54.9 matched no band and fell through to `return 500`. 125 training rows were labelled Hazardous at ordinary pollution levels. | `src/aqi.py` uses contiguous upper bounds, with a sweep test. |
| **Two EPA standards** | The forecaster used the pre-2024 PM2.5 table while the dashboard used the 2024 one — up to 25 AQI points apart on the same reading. | One shared module. |
| **Hardcoded secret** | A working WAQI token was the default value in `fetch_data.py`, published in a public repo and its history. | Environment-only; token rotated. |
| **Dead station shown as live** | The WAQI feed stopped in March 2025; the API kept serving that value and the dashboard showed it as current. | Feed removed, replaced by OpenAQ; `src/freshness.py` rejects stale readings. |
| **Dashboard crash** | The ground-station branch referenced a variable defined only in the fallback branch — `NameError` whenever a live reading existed. | Branch rewritten and tested in both states. |
| **Stale display** | Every metric and SHAP value on the dashboard was a hardcoded literal, so the daily retrain never changed what was shown. | All figures read from `models/*.json` and `models/shap_importance.csv`. |
| **Unused ingestion** | The forecast API was fetched and then discarded by the feature step. | Forecast weather and the CAMS forecast are both real inputs now. |
| **Lookahead in preprocessing** | Outlier caps used full-dataset percentiles; gaps were filled by interpolation, which averages the *next* observation. | Caps from training rows only; forward-fill only. |

---

## Author

**Mufeed Haider** · [@MufeedHaider](https://github.com/MufeedHaider)

Built during the 10Pearls Shine Internship — Data Science & AI track.

## License

MIT — see [LICENSE](LICENSE).
