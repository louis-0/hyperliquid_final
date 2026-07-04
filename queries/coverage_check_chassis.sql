-- Coverage check (chassis-only): row counts + time spans per (channel, coin).
-- Funding + marks windowed to MAX(ts) - 90d for exact 2,160 = 90 x 24 rows.
-- WebSocket streams unfiltered (capture started 2026-06-03).
--
-- Run from inside data/:
--   duckdb -c ".read ../queries/coverage_check_chassis.sql"

SELECT 'funding' AS channel, coin, COUNT(*) AS rows,
       MIN(ts) AS earliest, MAX(ts) AS latest
FROM read_parquet('funding/*.parquet')
WHERE coin IN ('BTC', 'ETH', 'SOL', 'NEAR', 'HYPE')
  AND ts > (SELECT MAX(ts) FROM read_parquet('funding/*.parquet') WHERE coin = 'BTC') - INTERVAL '90' DAY
GROUP BY coin
UNION ALL
SELECT 'marks', coin, COUNT(*), MIN(ts), MAX(ts)
FROM read_parquet('marks/*.parquet')
WHERE coin IN ('BTC', 'ETH', 'SOL', 'NEAR', 'HYPE')
  AND ts > (SELECT MAX(ts) FROM read_parquet('marks/*.parquet') WHERE coin = 'BTC') - INTERVAL '90' DAY
GROUP BY coin
UNION ALL
-- ws_trades counts unique tids (deduplicating WebSocket reconnect replays).
-- Narrow globs to chassis coins; xyz files may be mid-write by the daemon.
SELECT 'ws_trades', coin, COUNT(DISTINCT tid),
       epoch_ms(MIN(captured_ms)), epoch_ms(MAX(captured_ms))
FROM read_parquet(['ws/BTC/trades/*/*.parquet', 'ws/ETH/trades/*/*.parquet',
                   'ws/SOL/trades/*/*.parquet', 'ws/NEAR/trades/*/*.parquet',
                   'ws/HYPE/trades/*/*.parquet'])
GROUP BY coin
UNION ALL
SELECT 'ws_l2book', coin, COUNT(*),
       epoch_ms(MIN(captured_ms)), epoch_ms(MAX(captured_ms))
FROM read_parquet(['ws/BTC/l2book/*/*.parquet', 'ws/ETH/l2book/*/*.parquet',
                   'ws/SOL/l2book/*/*.parquet', 'ws/NEAR/l2book/*/*.parquet',
                   'ws/HYPE/l2book/*/*.parquet'])
GROUP BY coin
UNION ALL
SELECT 'ws_bbo', coin, COUNT(*),
       epoch_ms(MIN(captured_ms)), epoch_ms(MAX(captured_ms))
FROM read_parquet(['ws/BTC/bbo/*/*.parquet', 'ws/ETH/bbo/*/*.parquet',
                   'ws/SOL/bbo/*/*.parquet', 'ws/NEAR/bbo/*/*.parquet',
                   'ws/HYPE/bbo/*/*.parquet'])
GROUP BY coin
UNION ALL
SELECT 'ws_ctx', coin, COUNT(*),
       epoch_ms(MIN(captured_ms)), epoch_ms(MAX(captured_ms))
FROM read_parquet(['ws/BTC/ctx/*/*.parquet', 'ws/ETH/ctx/*/*.parquet',
                   'ws/SOL/ctx/*/*.parquet', 'ws/NEAR/ctx/*/*.parquet',
                   'ws/HYPE/ctx/*/*.parquet'])
GROUP BY coin
ORDER BY channel, coin;
