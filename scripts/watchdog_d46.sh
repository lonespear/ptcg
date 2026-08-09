#!/bin/bash
# D46 watchdog pair — run BOTH inside tmux (never bare-background; the
# SIGHUP lesson from the v3 runs).
#
#   keeper:  ./scripts/watchdog_d46.sh keeper  <machine> <deadline_epoch> [hours] [wall_hours]
#            Runs the driver in the FOREGROUND in a loop; on driver exit,
#            relaunches with --resume until the deep final eval has been
#            written or the deadline passes.
#   stall:   ./scripts/watchdog_d46.sh stall   <machine> <deadline_epoch>
#            Every 5 min, kills the driver if its log has gone 45 min
#            without a write (era prints + deep-eval rounds both write);
#            the keeper then resumes it.
set -uo pipefail
cd "$(dirname "$0")/.."

ROLE="${1:?keeper|stall}"
MACHINE="${2:?laptop|sebastian}"
DEADLINE="${3:?deadline epoch seconds}"
DLOG="runs/driver_d46_${MACHINE}.log"
WLOG="runs/watchdog_d46_${MACHINE}.log"
PAT="run_seeded.py --run-id seeded_d46"

note() { echo "$(date '+%F %T') [$ROLE] $*" | tee -a "$WLOG"; }

if [ "$ROLE" = keeper ]; then
  HOURS="${4:-10.5}"; WALL="${5:-11.2}"
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if grep -q "final_eval.json written" "$DLOG" 2>/dev/null; then
      note "run complete — keeper exiting"; exit 0
    fi
    note "launching driver (--resume; hours=$HOURS wall=$WALL)"
    ./scripts/overnight_d46.sh "$MACHINE" \
        --hours "$HOURS" --wall-hours "$WALL" --resume || true
    sleep 20
  done
  note "deadline passed — keeper exiting"
elif [ "$ROLE" = stall ]; then
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 300
    if grep -q "final_eval.json written" "$DLOG" 2>/dev/null; then
      note "run complete — stall watcher exiting"; exit 0
    fi
    if pgrep -f "$PAT" >/dev/null 2>&1 && [ -f "$DLOG" ]; then
      age=$(( $(date +%s) - $(stat -f %m "$DLOG" 2>/dev/null \
                              || stat -c %Y "$DLOG") ))
      if [ "$age" -gt 2700 ]; then
        note "STALL: log silent ${age}s — killing driver"
        pkill -f "$PAT" || true
        sleep 15
        pkill -9 -f "$PAT" || true
        # Reap orphaned pool workers (PPID 1 after the parent dies) —
        # they hold the driver's tee pipe open, which otherwise blocks
        # the keeper's relaunch forever (2026-08-08 01:40 lesson).
        sleep 5
        for wp in $(pgrep -f "spawn_main" 2>/dev/null); do
          if [ "$(ps -o ppid= -p "$wp" | tr -d ' ')" = "1" ]; then
            note "reaping orphan worker $wp"
            kill -9 "$wp" 2>/dev/null || true
          fi
        done
      fi
    fi
  done
  note "deadline passed — stall watcher exiting"
else
  echo "unknown role: $ROLE" >&2; exit 1
fi
