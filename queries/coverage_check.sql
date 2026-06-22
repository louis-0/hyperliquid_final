-- Coverage check: row counts + time spans per (channel, coin).
-- Verifies the 90-day REST window and per-coin WS coverage.
--
-- Run from inside data/ or data_sample/:
--   duckdb -c ".read ../queries/coverage_check.sql"

SELECT 'funding' AS channel, coin, COUNT(*) AS rows,
       MIN(ts) AS earliest, MAX(ts) AS latest
FROM read_parquet('funding/*.parquet') GROUP BY coin
UNION ALL
SELECT 'marks', coin, COUNT(*), MIN(ts), MAX(ts)
FROM read_parquet('marks/*.parquet') GROUP BY coin
UNION ALL
-- ws_trades counts unique tids (deduplicating WebSocket reconnect replays)
SELECT 'ws_trades', coin, COUNT(DISTINCT tid), epoch_ms(MIN(captured_ms)), epoch_ms(MAX(captured_ms))
FROM read_parquet('ws/*/trades/*/*.parquet') GROUP BY coin
UNION ALL
SELECT 'ws_l2book', coin, COUNT(*), epoch_ms(MIN(captured_ms)), epoch_ms(MAX(captured_ms))
FROM read_parquet('ws/*/l2book/*/*.parquet') GROUP BY coin
UNION ALL
SELECT 'ws_bbo', coin, COUNT(*), epoch_ms(MIN(captured_ms)), epoch_ms(MAX(captured_ms))
FROM read_parquet('ws/*/bbo/*/*.parquet') GROUP BY coin
UNION ALL
SELECT 'ws_ctx', coin, COUNT(*), epoch_ms(MIN(captured_ms)), epoch_ms(MAX(captured_ms))
FROM read_parquet('ws/*/ctx/*/*.parquet') GROUP BY coin
ORDER BY channel, coin;
