# W10 Feature Prototype: Hyperliquid Data Layer

This repository contains the W10 prototype code for the CM3070 Final Project: a Hyperliquid perpetual-futures basis-trade strategy with an XGBoost timing layer. The prototype is described in Chapter 4 of the preliminary report.

The scope of this prototype is the **data layer**: the multi-asset WebSocket capture daemon, the REST history fetcher, the strictly-causal DuckDB query layer, and a small reproducible data sample. The W11-W14 feature pipeline, signal layer, backtester, and evaluation are pre-registered design work covered in Chapter 3 of the prelim and are not in this repo's W10 scope.

The prototype demonstrates feasibility of multi-asset Hyperliquid venue connectivity at scale: five parallel WebSocket connection groups within Hyperliquid's empirical operating range of approximately 21-24 subscriptions per connection, with hourly Parquet rotation and strict schema enforcement on disk. DuckDB-over-Parquet preserves a strictly causal feature pipeline so every feature at time t uses only data with venue_time at most t.
