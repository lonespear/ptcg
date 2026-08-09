#!/bin/sh
# Run a command with the engine RNG pinned, so games reproduce.
#
#   scripts/run_pinned.sh python scripts/specialist_gate.py --games 600 ...
#
# Pass the command DIRECTLY. Do not put `env VAR=x` in front of it: macOS
# strips DYLD_* when it launches a SIP-protected binary, and /usr/bin/env is
# one, so the preload is silently gone and the run is not reproducible. Set
# variables on this script instead —
#
#   CABT_FREE_ABILITY=0 scripts/run_pinned.sh python scripts/...
#
# Every tool that measures prints whether it ended up pinned; believe that
# line rather than this one.
#
# Builds the preload on first use. Sets CABT_BUDGET_MODE=count as well, since
# the two halves of reproducibility only work together: a pinned engine still
# diverges if the agent's wall-clock deadline changes how much it searches.
# Override by exporting CABT_BUDGET_MODE before calling.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
case "$(uname -s)" in
Darwin) lib="$root/tools/engine_seed/libengine_seed.dylib" ;;
*) lib="$root/tools/engine_seed/libengine_seed.so" ;;
esac
[ -f "$lib" ] || sh "$root/tools/engine_seed/build.sh" >&2
case "$(uname -s)" in
Darwin) DYLD_INSERT_LIBRARIES="$lib"; export DYLD_INSERT_LIBRARIES ;;
*) LD_PRELOAD="$lib"; export LD_PRELOAD ;;
esac
CABT_BUDGET_MODE=${CABT_BUDGET_MODE:-count}
export CABT_BUDGET_MODE
exec "$@"
