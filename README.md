# W10 Feature Prototype: Hyperliquid Data Layer

Multi-asset WebSocket + REST capture for the CM3070 Final Project basis-trade strategy. Documented in Chapter 4 of the preliminary report.

Scope: data layer only. The W11-W14 feature pipeline, signal layer, backtester, and evaluation are pre-registered design work in Chapter 3 and not in this repo.

Six WebSocket connection groups (4 streaming + 2 ctx, 108 subs total) inside Hyperliquid's ~21-24 sub-per-connection cap. Hourly Parquet rotation, schema-enforced on disk, queryable via DuckDB.

## Setup

Python 3.10+.

```bash
git clone https://github.com/louis-0/hyperliquid.git
cd hyperliquid
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

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
`arch_replication.py` is the 27/27 ARCH-effect replication from prelim Section 4.3. Returns 1/1 on the BTC sample.

## Mapping to prelim Chapter 4

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
├── requirements.txt
├── queries/
│   ├── coverage_check.sql
│   └── arch_replication.py
├── data_sample/              BTC 24h + 90d REST (see MANIFEST.md)
├── data/                     gitignored
└── logs/                     gitignored
```
