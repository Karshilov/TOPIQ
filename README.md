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

Downloads NYX, SCALE-LETKF, and Hurricane datasets from [SDRBench](https://sdrbench.github.io/) and extracts the needed fields. CESM-ATM fields are bundled in the Docker image.

**Note on CESM-ATM data:** The CESM-ATM dataset on SDRBench has been updated since our experiments, and SDRBench does not support version control. To exactly reproduce the numbers in the paper, use the original CESM fields bundled in the Docker image (or contact the authors). To reproduce the methodology and verify that TOPIQ achieves comparable accuracy, you may download the current CESM-ATM data directly from SDRBench and place the four fields (`CLDTOT`, `CLDHGH`, `FLUT`, `FLUTC`) in the data directory.

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