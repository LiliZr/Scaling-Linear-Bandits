#!/usr/bin/env bash
# launch.sh — thin wrapper around `nohup python main.py <targets> --experiment <exp>`.
#
# Equivalent to:  nohup python main.py <targets> --experiment <exp> > /dev/null 2>&1 &
# but echoes the PID so you can kill it later.
#
# Examples:
#   ./launch.sh medium slow
#   ./launch.sh medium rp
#   ./launch.sh medium bandits
#   ./launch.sh medium "linucb sclub"          # arbitrary list (quote it)
#   ./launch.sh medium linucb_rp               # single algo
#
# After launch, the log file lives at:
#   results/<exp>/nohup_<exp>_<auto-derived-suffix>.log
#
# To monitor:
#   tail -f results/medium/nohup_medium_slow.log

set -e

EXP="${1:?usage: ./launch.sh <experiment> <targets...>}"
shift
TARGETS="$*"

if [ -z "$TARGETS" ]; then
    TARGETS="all"
fi

mkdir -p "results/$EXP"

echo "Launching: python main.py $TARGETS --experiment $EXP"
# stdbuf -oL flushes line-by-line so tail -f shows live progress.
nohup stdbuf -oL python -u main.py $TARGETS \
    --experiment "$EXP" \
    > /dev/null 2>&1 &

echo "PID: $!"
echo "Log will appear in: results/$EXP/nohup_${EXP}_<suffix>.log"
echo "  (suffix is auto-derived from your targets)"
