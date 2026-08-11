#!/usr/bin/env bash
# Keep the Hyperliquid WS capture daemon alive: relaunch if it's not running.
# Idempotent: does nothing if ws_capture.py is already up, so it's safe to run
# frequently. cron: every 10 min + @reboot (see crontab).
set -uo pipefail
cd "$(dirname "$0")" || exit 1

VENV="/home/l/obsidian/uni/Uni/2026-Spring/Final Project/0-initial_research/perps/venv/bin/python3"

# already running? leave it.
pgrep -f "ws_capture\.py" >/dev/null && exit 0
[ -x "$VENV" ] || { echo "$(date -Is) no venv python at $VENV" >> ws_capture_keepalive.log; exit 1; }

setsid "$VENV" -u ws_capture.py >> ws_capture.log 2>&1 < /dev/null &
echo "$(date -Is) relaunched ws_capture (was down)" >> ws_capture_keepalive.log