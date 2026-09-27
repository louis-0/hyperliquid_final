# Hyperliquid Funding-Basis Strategy: CM3070 Final Project

Multi-asset market-data capture and cost-realistic analysis for a funding-basis (long-spot / short-perp) strategy on Hyperliquid perpetuals. Two parts: a continuous data-capture layer, and `hlq`, a unit-tested analysis package that turns the market data into backtests.

Capture runs eight WebSocket connection groups (5 streaming + 3 ctx, 148 subs total) inside Hyperliquid's ~21-24 sub-per-connection cap, with hourly Parquet rotation, schema-enforced on disk and queryable via DuckDB. REST fetchers backfill funding, mark, and spot candles.

## Setup

Python 3.10+.

```bash
git clone https://github.com/louis-0/hyperliquid_final.git
cd hyperliquid_final
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
| `hlq.microfeatures` | Hourly features from the captured tape (book imbalance, spread, taker flow, premium, cohort flow) via DuckDB |
| `hlq.flow` | Signed trade flows, fixed-length bars, rank correlation and top-decile forward moves |
| `hlq.results` | Run store keyed by a config hash, with no silent overwrite |

Backtests are thin scripts over the package:

```bash
python scripts/run_chassis.py         # funding-carry chassis: per coin + equal-weight basket
python scripts/run_basis_drift.py     # basis-drift-inclusive PnL: per coin + basket
python scripts/run_regime.py          # basis-drift Sharpe split by bear/bull/calm regime
python scripts/run_cost_sweep.py      # drift x borrow x fee-multiplier sensitivity on the drop-SOL basket
python scripts/run_dsr_ablation.py    # four rungs per coin and basket, deflated Sharpe
python scripts/run_baselines.py       # strategy rungs vs buy-and-hold and naive carry on one window
python scripts/run_l2_slippage.py     # realised slippage walked from the captured order books
python scripts/run_carry_ledger.py    # carry basket as one measured round trip per coin
python scripts/run_ml_feasibility.py  # XGBoost vs naive and classical baselines, with SHAP drivers
python scripts/run_fused_ml.py        # candle features vs candle + tape features on the same split
python scripts/run_ml_gated_ledger.py # carry gated by the funding-sign model, per-trade ledger
python scripts/run_obi.py             # order-book imbalance vs forward mid return on 5 s bars
python scripts/run_obi_ledger.py      # imbalance scalp as a per-trade ledger
python scripts/run_informed_flow.py   # frozen-cohort flow vs anonymous taker flow on 30 s bars
python scripts/run_perp_dispersion.py # short high-funding perps against low-funding perps
python scripts/build_wallet_labels.py # per-wallet PnL and the is_smart label set, frozen at --end
python scripts/data_layer_metrics.py  # capture coverage, latency, and hourly integrity
```

Each run records its config and headline numbers under `results/`, keyed by a config hash, so a run is reproducible and re-running is idempotent. The JSON records and every ledger are committed, the imbalance ledgers gzipped.

Tests run off the committed `data_sample/`, so they need no real data:

```bash
pytest
```

## The advisor dashboard

A read-only, cost-aware advisor over `hlq` (FastAPI + Jinja2). The cost-floor calculator lets a user enter their own fee tier, borrow, and any extra basis drift and returns net APR, net Sharpe, and a deploy verdict against the He et al. (2024) anchors (1.8 retail, 3.5 market-maker); the default retail-taker scenario (11 bp round trip, 1 bp per day borrow) returns "marginal". A signal panel shows the per-coin degradation ladder (funding-only, basis-drift, net of fees and borrow) and the macro regime as of the data's last timestamp. A ledger panel puts the three per-trade ledgers side by side (round trips, per-coin and mean nets, record hashes) with their cumulative curves drawn from the committed ledger files. An explainability panel trains the funding-sign model on demand and reports its top SHAP drivers, and a smart-money cohort panel shows a frozen cohort's aggregate net flow and share of turnover per coin, never individual wallets.

```bash
uvicorn app.main:app --reload      # then open http://127.0.0.1:8000
```

No live orders; it reads the local capture (point `HLQ_DATA_ROOT` elsewhere to override). The cohort panel reads `app/cohort_snapshot.json`, a small aggregate built once from the captured trades and a cohort label set frozen at 2026-06-23, over the post-freeze tape to 2026-08-22:

```bash
python scripts/build_cohort_snapshot.py data/wallet_labels_2026-06-23.parquet --start 2026-06-23 --end 2026-08-22
```

Only the collapsed per-coin aggregate (net flow, share of turnover, leg count) is written; no wallet address enters the repo. The ledger panel reads `app/ledger_snapshot.json` and `app/static/ledger_trilogy.png`, both built from the recorded ledgers:

```bash
python scripts/build_ledger_snapshot.py   # per-strategy table from the recorded ledger results
python scripts/build_ledger_figure.py      # cumulative curves from results/*ledger*.csv[.gz]
```

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
python3 fetch_rest_history.py --start 2023-12-01 --end 2026-08-23   # pinned window
python3 fetch_spot_history.py --start 2024-01-01 --end 2026-08-23   # spot pairs, @index ids
```

Writes `data/funding/`, `data/marks/` (1h), `data/spot/`, or `data/marks_{interval}/` for other intervals. Idempotent; `--end` pins a reproducible window.

### Data snapshot

The recorded results under `results/` were computed on a snapshot pinned at 2026-08-23 (last complete day 2026-08-22), re-fetchable with the pinned commands above:

| Layer | Window | Bound by |
|---|---|---|
| `data/funding/` | 2023-12-01 to 2026-08-22 (996 d, hourly) | window start; the venue serves coarser 8-hourly rows before roughly December 2023 |
| `data/marks/`, `data/spot/` | 2026-01-27 to 2026-08-23 (209 d, 207 complete) | venue retention of roughly 5,000 candles per series, a rolling cap |
| `data/ws/` | live since 2026-06-03 | not re-fetchable: the venue exposes no historical per-wallet tape, so this layer only grows forward |

Later-listed coins carry shorter funding series inside the same window (HYPE 626 d, ZEC 325 d, XMR 220 d). Funding-only rungs use the full funding window; basis-drift, realistic, and feature-based runs inherit the spot/marks intersection; tape-based analyses state their own as-of date.

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
├── fetch_rest_history.py     REST funding + mark-candle fetcher
├── fetch_spot_history.py     REST spot-candle fetcher (@index pairs)
├── hlq/                      analysis package (data, stats, costs, signals, portfolio, execution, features, explain, baselines, timing, microfeatures, flow, results)
├── scripts/                  runnable backtests, ledgers, and studies (see the list above) + build_wallet_labels, build_cohort_snapshot, build_ledger_snapshot, build_ledger_figure, data_layer_metrics
├── app/                      advisor dashboard (FastAPI + Jinja2, read-only over hlq; cohort_snapshot.json, ledger_snapshot.json)
├── tests/                    pytest suite (run off data_sample/)
├── requirements.txt
├── pytest.ini
├── queries/
│   ├── coverage_check.sql
│   └── arch_replication.py
├── data_sample/              BTC 24h + 90d REST (see MANIFEST.md)
├── data/                     gitignored
├── results/                  recorded run outputs (JSON records and ledgers committed; raw imbalance CSVs ignored)
└── logs/                     gitignored
```