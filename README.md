# TOPIQ: Statistical Error Propagation for Quantity-of-Interest Prediction under Lossy Compression

Artifact for reproducing the experimental results in the paper.

## Prerequisites

- **Python 3.9+** with packages: `pip install numpy matplotlib torch`
- **Compressors** on PATH: [SZ3](https://github.com/szcompressor/SZ3), [SPERR](https://github.com/NCAR/SPERR), [ZFP](https://github.com/LLNL/zfp)
- **g++** (for C++ throughput benchmark)

## Data Preparation

```bash
bash scripts/prepare_data.sh data
```

Downloads all four datasets from [SDRBench](https://sdrbench.github.io/) (~5 GB), extracts the needed fields, and computes log-density transforms for NYX.

## Quick Start

```bash
./experiments/run_all.sh data
```

Runs all experiments sequentially and prints results to stdout. JSON files are saved to `results/`.

## Individual Experiments

### Ablation Study (Table III)

```bash
python experiments/ablation.py --data-dir data --datasets CESM NYX
```

### Prediction Accuracy (Table IV)

```bash
python experiments/prediction_accuracy.py --data-dir data --datasets CESM
python experiments/prediction_accuracy.py --data-dir data --datasets NYX,SCALE,Hurricane
```

### Throughput Benchmark

```bash
python benchmark/run_benchmark.py --data-dir data --compressor sz3 --reb 1e-3 --n-queries 10000
```