#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "[Company-Finder] Starting pipeline"

echo "1) Enhancing company data..."
$PYTHON enhance_data_companies.py

echo "2) Running full ranking pipeline (filter -> embed -> rank -> select)..."
$PYTHON embedding_ranking.py

echo "Done. Outputs are written to .tmp/ (filtered, ranked, final csv)."
