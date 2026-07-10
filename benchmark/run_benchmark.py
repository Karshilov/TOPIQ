#!/usr/bin/env python3
"""
TOPIQ throughput benchmark (Figure 6).

Compresses data, builds metadata, then runs the C++ benchmark
measuring TOPIQ prediction vs direct computation for 3 QoI families.

Usage:
    python run_benchmark.py --data-dir /path/to/1800x3600 [--compressor sz3]

Requirements: g++ compiler, sz3 or zfp on PATH.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np

SHAPE = (1800, 3600)
DTYPE = np.float32
META_BLOCK = (150, 150)
EVAL_BLOCK = (200, 200)
N_PIXELS = EVAL_BLOCK[0] * EVAL_BLOCK[1]


def iter_blocks(shape, block_size):
    ranges = [range(0, s, b) for s, b in zip(shape, block_size)]
    for starts in product(*ranges):
        sl = tuple(slice(s, min(s + b, dim))
                   for s, b, dim in zip(starts, block_size, shape))
        yield sl


def compress(fpath, abs_eb, shape, compressor):
    h, w = shape
    with tempfile.TemporaryDirectory() as td:
        zf = os.path.join(td, "c.bin")
        of = os.path.join(td, "d.f32")
        if compressor == "sz3":
            subprocess.run(["sz3", "-f", "-i", fpath, "-z", zf,
                            "-M", "ABS", str(abs_eb), "-2", str(h), str(w)],
                           check=True, capture_output=True)
            subprocess.run(["sz3", "-f", "-z", zf, "-o", of,
                            "-2", str(h), str(w)],
                           check=True, capture_output=True)
        elif compressor == "zfp":
            subprocess.run(["zfp", "-f", "-2", str(w), str(h),
                            "-i", fpath, "-z", zf, "-a", str(abs_eb), "-h", "-q"],
                           check=True, capture_output=True)
            subprocess.run(["zfp", "-h", "-z", zf, "-o", of, "-q"],
                           check=True, capture_output=True)
        else:
            raise ValueError(f"Unknown compressor: {compressor}")
        return np.fromfile(of, dtype=DTYPE).reshape(shape)


def compute_alpha(err, block_size):
    n_full = int(np.prod(block_size))
    pixel_vars, block_sums = [], []
    for sl in iter_blocks(err.shape, block_size):
        blk = err[sl]
        if blk.size < n_full:
            continue
        pixel_vars.append(float(np.var(blk)))
        block_sums.append(float(np.sum(blk)))
    mean_pv = np.mean(pixel_vars)
    var_bs = np.var(block_sums)
    return var_bs / (n_full * mean_pv) if mean_pv > 1e-30 else 1.0, n_full


def build_meta(orig, dec, block_size):
    meta = []
    for sl in iter_blocks(orig.shape, block_size):
        b0 = orig[sl].astype(np.float64)
        b1 = dec[sl].astype(np.float64)
        e = b1 - b0
        mu = float(np.mean(b0))
        mu_sq = float(np.mean(b0 ** 2))
        var_err = float(np.var(e))
        cov_xe = float(np.mean((b0 - mu) * e))
        meta.append({"slices": sl, "mu": mu, "mu_sq": mu_sq,
                      "var_err": var_err, "cov_xe": cov_xe})
    return meta


def export_meta_bin(meta, path):
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(meta)))
        for b in meta:
            sl = b["slices"]
            f.write(struct.pack("ii", sl[0].start, sl[1].start))
            f.write(struct.pack("ii", sl[0].stop, sl[1].stop))
            f.write(struct.pack("dddd", b["mu"], b["mu_sq"],
                                b["var_err"], b["cov_xe"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--field", default="CLDTOT_1_1800_3600.f32")
    parser.add_argument("--compressor", choices=["sz3", "zfp"], default="sz3")
    parser.add_argument("--reb", type=float, default=1e-3)
    parser.add_argument("--n-queries", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    fpath = os.path.join(args.data_dir, args.field)
    if not os.path.exists(fpath):
        print(f"Error: {fpath} not found")
        sys.exit(1)

    print(f"TOPIQ Throughput Benchmark")
    print(f"  Field: {args.field}, Compressor: {args.compressor}, reb={args.reb}")

    orig = np.fromfile(fpath, dtype=DTYPE).reshape(SHAPE)
    abs_eb = args.reb * float(orig.max() - orig.min())
    dec = compress(fpath, abs_eb, SHAPE, args.compressor)
    err = (dec - orig).astype(np.float64)

    alpha_raw, n_base = compute_alpha(err, EVAL_BLOCK)
    meta = build_meta(orig, dec, META_BLOCK)
    meta_bin = str(script_dir / "meta.bin")
    export_meta_bin(meta, meta_bin)
    print(f"  Alpha={alpha_raw:.2f}, meta_blocks={len(meta)}")

    orig_bin = str(script_dir / "orig.f32")
    dec_bin = str(script_dir / "dec.f32")
    orig.tofile(orig_bin)
    dec.tofile(dec_bin)

    cpp_src = str(script_dir / "topiq_bench.cpp")
    cpp_bin = str(script_dir / "topiq_bench")
    r = subprocess.run(["g++", "-O3", "-o", cpp_bin, cpp_src, "-lm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Compile error: {r.stderr}")
        sys.exit(1)

    r = subprocess.run(
        [cpp_bin, meta_bin, orig_bin, dec_bin,
         str(alpha_raw), str(n_base), str(args.n_queries), str(args.seed)],
        capture_output=True, text=True)
    print(r.stdout)

    with open(args.output, "w") as f:
        json.dump({"stdout": r.stdout, "field": args.field,
                   "compressor": args.compressor, "reb": args.reb,
                   "alpha": alpha_raw, "n_queries": args.n_queries}, f, indent=2)
    print(f"Saved to {args.output}")

    for p in [meta_bin, orig_bin, dec_bin]:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
