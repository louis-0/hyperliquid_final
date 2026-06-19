# data_sample

BTC, 2026-06-18. One UTC day of WS captures plus the 90-day REST history. ~50 MB.

```
ws/BTC/trades/2026-06-18/{00..23}.parquet
ws/BTC/l2book/2026-06-18/{00..23}.parquet
ws/BTC/bbo/2026-06-18/{00..23}.parquet
ws/BTC/ctx/2026-06-18/{00..23}.parquet
funding/BTC.parquet
marks/BTC.parquet
```

Schemas: `pd.read_parquet(path).dtypes` or Section 4.2 of the preliminary report.
