#!/usr/bin/env bash
# get_movielens.sh — convenience downloader + preprocessor for MovieLens.
#
# Usage:
#   ./get_movielens.sh small    # ml-latest-small (~1MB, fast iteration)
#   ./get_movielens.sh 1m       # ml-1m (~6MB, 6K users, 4K movies)
#   ./get_movielens.sh 25m      # ml-25m (~250MB, 162K users, 62K movies)

set -e

VARIANT="${1:-small}"
case "$VARIANT" in
  small) URL="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
         DIR="ml-latest-small";;
  1m)    URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
         DIR="ml-1m";;
  25m)   URL="https://files.grouplens.org/datasets/movielens/ml-25m.zip"
         DIR="ml-25m";;
  *)     echo "Unknown variant '$VARIANT'. Use: small | 1m | 25m" >&2; exit 1;;
esac

DATA_ROOT="data/movielens_raw"
mkdir -p "$DATA_ROOT"

if [ ! -d "$DATA_ROOT/$DIR" ]; then
  echo "Downloading $URL..."
  wget -q --show-progress "$URL" -O /tmp/ml-tmp.zip
  unzip -q /tmp/ml-tmp.zip -d "$DATA_ROOT"
  rm /tmp/ml-tmp.zip
fi

OUT="data/movielens_${VARIANT}.npz"
echo "Preprocessing into $OUT..."
python data/movielens.py "$DATA_ROOT/$DIR" \
    --d-target 100 --D 2 \
    --min-user 20 --min-item 20 \
    --out "$OUT"

echo
echo "Done. Run experiments with:"
echo "  python main.py --experiment medium --data-npz $OUT"
