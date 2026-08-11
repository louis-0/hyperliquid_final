#!/usr/bin/env bash
# Silent-stall / freshness monitor for the ws_capture daemon.
# The restart-keepalive only catches a DEAD process (pgrep). This catches the
# other failure mode too: data stops growing while the process is still alive
# (a wedged websocket), and it also recovers a death faster than the */10
# keepalive. Alerts (log + desktop notify) and self-heals. cron: */2.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

STALE_MIN=4                       # alert/heal if newest parquet older than this
ALERT_LOG="ws_capture_alert.log"

newest=$(find data/ws -name '*.parquet' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
[ -z "$newest" ] && exit 0                                          # no data yet
age=$(( ($(date +%s) - ${newest%.*}) / 60 ))
[ "$age" -lt "$STALE_MIN" ] && exit 0                              # fresh, nothing to do

msg="$(date -Is) STALL: newest ws parquet is ${age}m old (>= ${STALE_MIN}m)"
echo "$msg" >> "$ALERT_LOG"
DISPLAY=:1 notify-send -u critical "HL spider STALLED" "$msg" 2>/dev/null || true

# wedged-but-alive → recycle so the keepalive can relaunch a clean process
if pgrep -f "ws_capture\.py" >/dev/null; then
    echo "$(date -Is)   process alive but stale, recycling" >> "$ALERT_LOG"
    pkill -f "ws_capture\.py"; sleep 2
fi
./ws_capture_keepalive.sh                                          # relaunch (idempotent)