#!/bin/bash
set -e

# Usage: ./run_all.sh /path/to/data
# Run `bash scripts/prepare_data.sh /path/to/data` first to download datasets.
# Required on PATH: sz3, sperr2d, sperr3d, zfp, g++, python3

DATA_DIR="${1:?Usage: $0 <data-dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  TOPIQ Artifact Evaluation"
echo "  Data directory: $DATA_DIR"
echo "============================================================"

# 1. Ablation study (Table III)
echo ""
echo "[1/3] Running ablation study..."
python3 "$SCRIPT_DIR/ablation.py" \
    --data-dir "$DATA_DIR" \
    --datasets CESM NYX \
    --output-dir "$ROOT/results"

# 2. Prediction accuracy (Table IV)
echo ""
echo "[2/3] Running prediction accuracy (CESM first, then 3D datasets)..."
python3 "$SCRIPT_DIR/prediction_accuracy.py" \
    --data-dir "$DATA_DIR" \
    --datasets CESM \
    --output-dir "$ROOT/results"

python3 "$SCRIPT_DIR/prediction_accuracy.py" \
    --data-dir "$DATA_DIR" \
    --datasets NYX,SCALE,Hurricane \
    --output-dir "$ROOT/results"

# 3. Throughput benchmark
echo ""
echo "[3/3] Running throughput benchmark..."
python3 "$ROOT/benchmark/run_benchmark.py" \
    --data-dir "$DATA_DIR" \
    --compressor sz3 --reb 1e-3 \
    --n-queries 10000 \
    --output "$ROOT/results/benchmark_results.json"

echo ""
echo "============================================================"
echo "  Done. Results saved to: $ROOT/results/"
echo "============================================================"
