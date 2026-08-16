# Hyperliquid Funding-Basis Strategy: CM3070 Final Project

Multi-asset market-data capture and cost-realistic analysis for a funding-basis (long-spot / short-perp) strategy on Hyperliquid perpetuals. Two parts: a continuous data-capture layer, and `hlq`, a unit-tested analysis package that turns the market data into backtests.

Capture runs six WebSocket connection groups (4 streaming + 2 ctx, 108 subs total) inside Hyperliquid's ~21-24 sub-per-connection cap, with hourly Parquet rotation, schema-enforced on disk and queryable via DuckDB. REST fetchers backfill funding, mark, and spot candles.

## Setup

Python 3.10+.

```bash
git clone https://github.com/louis-0/hyperliquid.git
cd hyperliquid
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## The `hlq` analysis package

Small, unit-tested modules:

| Module | Provides |
|---|---|
| `hlq.data` | Parquet loaders (funding/marks/spot), hour alignment, a look-ahead guard |
| `hlq.stats` | Annualised Sharpe, daily aggregation, moving-block bootstrap CI (Künsch 1989), Probabilistic/Deflated Sharpe (Bailey & López de Prado 2012/2014) |
| `hlq.costs` | Taker/maker fee + slippage cost model, per-turnover round-trip |
| `hlq.signals` | Funding-carry and basis-drift PnL engines, and the bear/bull/calm regime label |
| `hlq.portfolio` | Equal-weight basket over the coins' common window |
| `hlq.execution` | Volume-weighted fill price and slippage from an L2 order book |
| `hlq.features` | Leakage-safe feature matrix for the next-hour funding model |
| `hlq.explain` | SHAP attribution over the fitted model (Lundberg & Lee 2017) |
| `hlq.baselines` | Naive prediction baselines (majority class, persistence) |
| `hlq.timing` | Next-hour funding-sign model (XGBoost) and its SHAP drivers, shared by the ML backtest and the dashboard |
| `hlq.results` | Run store keyed by a config hash, with no silent overwrite |

Backtests are thin scripts over the package:

```bash
python scripts/run_chassis.py         # funding-carry chassis: per coin + equal-weight basket
python scripts/run_basis_drift.py     # basis-drift-inclusive PnL: per coin + basket
python scripts/run_regime.py          # basis-drift Sharpe split by bear/bull/calm regime
python scripts/run_l2_slippage.py     # realised slippage walked from the captured order books
python scripts/run_ml_feasibility.py  # XGBoost vs naive baselines, with SHAP drivers
```

Each run records its config and headline numbers under `results/` (gitignored), keyed by a config hash, so a run is reproducible and re-running is idempotent.

Tests run off the committed `data_sample/`, so they need no real data:

```bash
pytest -q
```

## The advisor dashboard

A read-only, cost-aware advisor over `hlq` (FastAPI + Jinja2). The cost-floor calculator lets a user enter their own fee tier, borrow, and basis drift and returns net APR, net Sharpe, and a deploy verdict against the He et al. (2024) anchors (1.8 retail, 3.5 market-maker); the default retail-taker scenario returns "do not deploy". A signal panel shows the per-coin degradation ladder (funding-only to basis-drift to realistic net Sharpe) and the macro regime as of the data's last timestamp. An explainability panel trains the funding-sign model on demand and reports its top SHAP drivers, and a smart-money cohort panel shows a frozen cohort's aggregate net directional flow and share of turnover per coin, never individual wallets.

```bash
uvicorn app.main:app --reload      # then open http://127.0.0.1:8000
```

No live orders; it reads the local capture (point `HLQ_DATA_ROOT` elsewhere to override). The cohort panel reads `app/cohort_snapshot.json`, a small aggregate built once from the captured trades and a frozen cohort label set:

```bash
python scripts/build_cohort_snapshot.py /path/to/wallet_labels.parquet
```

Only the collapsed per-coin aggregate (net flow, share of turnover, leg count) is written; no wallet address enters the repo.

## Run (capture)

### Capture daemon

```bash
mkdir -p logs
nohup python3 -u ws_capture.py > logs/ws_capture.log 2>&1 &
```

Writes `data/ws/{coin}/{channel}/{YYYY-MM-DD}/{HH}.parquet`, rotating hourly. Stop with `pkill -TERM -f ws_capture.py`.

### REST history fetcher

```bash
python3 fetch_rest_history.py                                   # 90 days, 1h, all coins
python3 fetch_rest_history.py BTC ETH
python3 fetch_rest_history.py --interval 5m
python3 fetch_rest_history.py --interval 1h --start 2023-06-01
```

Writes `data/funding/`, `data/marks/` (1h), or `data/marks_{interval}/` for other intervals. Idempotent. Hyperliquid retention caps: 3y funding, 208d 1h marks, 17d 5m marks.

### Queries

```bash
# coverage_check.sql, run from data_sample/ (or data/)
cd data_sample
python3 -c "import duckdb; duckdb.sql(open('../queries/coverage_check.sql').read()).show()"
cd ..

# arch_replication.py takes the data root as argument
python3 queries/arch_replication.py data_sample
```

`coverage_check.sql` reports row counts and time spans per (channel, coin).
`arch_replication.py` is the 27/27 ARCH-effect replication from preliminary-report Section 4.3. Returns 1/1 on the BTC sample.

## Mapping to preliminary report Chapter 4

| Section | Code |
|---|---|
| 4.1 What the prototype is | `ws_capture.py`, `fetch_rest_history.py` |
| 4.2 Evaluation methodology | `queries/coverage_check.sql` |
| 4.3 Findings (27/27 ARCH) | `queries/arch_replication.py` |
| 4.4 Limitations | `ws_capture.py` `COIN_GROUPS` + `CTX_GROUPS` |

## Layout

```
hyperliquid/
├── ws_capture.py             WS daemon
├── fetch_rest_history.py     REST snapshot fetcher
├── hlq/                      analysis package (data, stats, costs, signals, portfolio, execution, features, explain, baselines, timing, results)
├── scripts/                  runnable backtests (run_chassis/basis_drift/regime/l2_slippage/ml_feasibility) + build_cohort_snapshot
├── app/                      advisor dashboard (FastAPI + Jinja2, read-only over hlq; cohort_snapshot.json)
├── tests/                    pytest suite (run off data_sample/)
├── requirements.txt
├── pytest.ini
├── queries/
│   ├── coverage_check.sql
│   └── arch_replication.py
├── data_sample/              BTC 24h + 90d REST (see MANIFEST.md)
├── data/                     gitignored
├── results/                  recorded run outputs, gitignored
└── logs/                     gitignored
```