#!/usr/bin/env python3
"""
Ablation study (Table 1, Figure 3).

Validates two key TOPIQ modeling choices:
  1. Spatial correlation index alpha
  2. Data-error covariance Cov(x, e)

Four modes per configuration:
  full     - TOPIQ with all corrections
  no_alpha - force alpha=1 (i.i.d. assumption)
  no_cov   - force Cov(x,e)=0
  shuffled - randomly permute error pixels (control)

Usage:
    python ablation.py --data-dir /path/to/data [--datasets CESM NYX]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topiq.executor import execute_graph
from topiq.utils import (
    compute_alpha_raw, iter_blocks, make_tensor,
    SummaryAccumulator, GRAPH_MEAN_SQUARE,
    compress, shuffle_error,
)

REL_EBS = [1e-3, 1e-4, 1e-5]
COMPRESSORS = ["sz3", "sperr", "zfp"]


def eval_mean_square(orig, dec, alpha, eval_block, **kw):
    acc = SummaryAccumulator()
    for sl in iter_blocks(orig.shape, eval_block):
        b0, b1 = orig[sl], dec[sl]
        n = int(np.prod(b0.shape))
        real = float(np.mean(b1.astype(np.float64) ** 2) -
                     np.mean(b0.astype(np.float64) ** 2))
        t = make_tensor(b0, b1, alpha, n, **kw)
        out = execute_graph(GRAPH_MEAN_SQUARE, {"input": t, "N": n})["mean_sq"]
        acc.update(real, out.bias, out.var_err)
    return acc.to_dict()


def eval_weighted_sum(x_orig, x_dec, w_orig, w_dec, alpha_x, alpha_w,
                      eval_block, **kw):
    acc = SummaryAccumulator()
    for sl in iter_blocks(x_orig.shape, eval_block):
        x0, x1 = x_orig[sl], x_dec[sl]
        w0, w1 = w_orig[sl], w_dec[sl]
        n = int(np.prod(x0.shape))
        real = float(np.sum(x1.astype(np.float64) * w1.astype(np.float64)) -
                     np.sum(x0.astype(np.float64) * w0.astype(np.float64)))
        tx = make_tensor(x0, x1, alpha_x, n, **kw)
        tw = make_tensor(w0, w1, alpha_w, n, **kw)
        out = (tx * tw).sum(n)
        acc.update(real, out.bias, out.var_err)
    return acc.to_dict()


def get_datasets(data_dir):
    d = Path(data_dir)
    return {
        "CESM": {
            "data_dir": d,
            "shape": (1800, 3600), "dtype": np.float32,
            "fields": ["CLDTOT_1_1800_3600.f32", "CLDHGH_1_1800_3600.f32"],
            "ws_pairs": [("CLDTOT_1_1800_3600.f32", "CLDHGH_1_1800_3600.f32")],
            "eval_block": (200, 200), "alpha_block": (200, 200),
        },
        "NYX": {
            "data_dir": d,
            "shape": (512, 512, 512), "dtype": np.float32,
            "fields": ["dark_matter_density_log.f32", "baryon_density_log.f32",
                       "velocity_x.f32"],
            "ws_pairs": [("dark_matter_density_log.f32", "velocity_x.f32")],
            "eval_block": (32, 32, 32), "alpha_block": (32, 32, 32),
        },
        "SCALE": {
            "data_dir": d,
            "shape": (98, 1200, 1200), "dtype": np.float32,
            "fields": ["PRES-98x1200x1200.f32", "T-98x1200x1200.f32"],
            "ws_pairs": [("PRES-98x1200x1200.f32", "T-98x1200x1200.f32")],
            "eval_block": (32, 32, 32), "alpha_block": (32, 32, 32),
        },
        "Hurricane": {
            "data_dir": d,
            "shape": (100, 500, 500), "dtype": np.float32,
            "fields": ["TCf48.bin.f32", "Uf48.bin.f32"],
            "ws_pairs": [("TCf48.bin.f32", "Uf48.bin.f32")],
            "eval_block": (32, 32, 32), "alpha_block": (32, 32, 32),
        },
    }


def run_dataset(ds_name, ds_cfg, compressors, rel_ebs, out_dir):
    data_dir = ds_cfg["data_dir"]
    shape = ds_cfg["shape"]
    dtype = ds_cfg["dtype"]
    fields = ds_cfg["fields"]
    ws_pairs = ds_cfg["ws_pairs"]
    eval_block = ds_cfg["eval_block"]
    alpha_block = ds_cfg["alpha_block"]

    results = {}

    for cmp in compressors:
        print(f"\n{'#' * 70}")
        print(f"  [{ds_name}] Compressor: {cmp.upper()}")
        print(f"{'#' * 70}")

        for field_name in fields:
            fpath = str(data_dir / field_name)
            if not os.path.isfile(fpath):
                print(f"  WARNING: {fpath} not found, skipping")
                continue
            orig = np.fromfile(fpath, dtype=dtype).reshape(shape)
            vrange = float(orig.max() - orig.min())

            for reb in rel_ebs:
                aeb = reb * vrange
                key = f"{cmp}|{field_name}|{reb:.0e}"
                print(f"\n  {cmp} {field_name} reb={reb:.0e}")

                try:
                    dec = compress(fpath, aeb, cmp, shape, dtype)
                except Exception as e:
                    print(f"  SKIP: {e}")
                    continue

                err_f = dec.astype(np.float64) - orig.astype(np.float64)
                alpha = compute_alpha_raw(err_f, alpha_block)
                dec_shuf = shuffle_error(orig, dec)
                alpha_shuf = compute_alpha_raw(
                    dec_shuf.astype(np.float64) - orig.astype(np.float64),
                    alpha_block)

                entry = {
                    "dataset": ds_name, "compressor": cmp,
                    "field": field_name, "reb": reb,
                    "alpha": alpha["alpha_raw"],
                    "alpha_shuf": alpha_shuf["alpha_raw"],
                }

                r = {}
                r["full"] = eval_mean_square(orig, dec, alpha, eval_block)
                r["no_alpha"] = eval_mean_square(orig, dec, alpha, eval_block,
                                                 zero_alpha=True)
                r["no_cov"] = eval_mean_square(orig, dec, alpha, eval_block,
                                               zero_cov=True)
                r["shuffled"] = eval_mean_square(orig, dec_shuf, alpha_shuf,
                                                 eval_block)
                entry["mean_square"] = r
                for m, v in r.items():
                    print(f"    {m:12s}  cov_3s={v['coverage_3sigma']:.3f}  "
                          f"z_mean={v['z_mean']:+.4f}  z_std={v['z_std']:.4f}")

                results[key] = entry

        for x_name, w_name in ws_pairs:
            x_path = str(data_dir / x_name)
            w_path = str(data_dir / w_name)
            if not (os.path.isfile(x_path) and os.path.isfile(w_path)):
                continue
            x_orig = np.fromfile(x_path, dtype=dtype).reshape(shape)
            w_orig = np.fromfile(w_path, dtype=dtype).reshape(shape)
            vr_x = float(x_orig.max() - x_orig.min())
            vr_w = float(w_orig.max() - w_orig.min())

            for reb in rel_ebs:
                key = f"{cmp}|ws|{x_name}:{w_name}|{reb:.0e}"
                print(f"\n  {cmp} WS: {x_name} x {w_name} reb={reb:.0e}")
                try:
                    x_dec = compress(x_path, reb * vr_x, cmp, shape, dtype)
                    w_dec = compress(w_path, reb * vr_w, cmp, shape, dtype)
                except Exception as e:
                    print(f"  SKIP: {e}")
                    continue

                ax = compute_alpha_raw(
                    x_dec.astype(np.float64) - x_orig.astype(np.float64),
                    alpha_block)
                aw = compute_alpha_raw(
                    w_dec.astype(np.float64) - w_orig.astype(np.float64),
                    alpha_block)
                x_dec_s = shuffle_error(x_orig, x_dec, seed=42)
                w_dec_s = shuffle_error(w_orig, w_dec, seed=43)
                axs = compute_alpha_raw(
                    x_dec_s.astype(np.float64) - x_orig.astype(np.float64),
                    alpha_block)
                aws = compute_alpha_raw(
                    w_dec_s.astype(np.float64) - w_orig.astype(np.float64),
                    alpha_block)

                entry = {
                    "dataset": ds_name, "compressor": cmp,
                    "x_field": x_name, "w_field": w_name, "reb": reb,
                    "alpha_x": ax["alpha_raw"], "alpha_w": aw["alpha_raw"],
                }

                r = {}
                r["full"] = eval_weighted_sum(
                    x_orig, x_dec, w_orig, w_dec, ax, aw, eval_block)
                r["no_alpha"] = eval_weighted_sum(
                    x_orig, x_dec, w_orig, w_dec, ax, aw, eval_block,
                    zero_alpha=True)
                r["no_cov"] = eval_weighted_sum(
                    x_orig, x_dec, w_orig, w_dec, ax, aw, eval_block,
                    zero_cov=True)
                r["shuffled"] = eval_weighted_sum(
                    x_orig, x_dec_s, w_orig, w_dec_s, axs, aws, eval_block)
                entry["weighted_sum"] = r
                for m, v in r.items():
                    print(f"    {m:12s}  cov_3s={v['coverage_3sigma']:.3f}  "
                          f"z_mean={v['z_mean']:+.4f}  z_std={v['z_std']:.4f}")

                results[key] = entry

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_{ds_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Root data directory")
    p.add_argument("--datasets", nargs="+",
                   default=["CESM", "NYX", "SCALE", "Hurricane"])
    p.add_argument("--compressors", nargs="+", default=COMPRESSORS)
    p.add_argument("--rel-ebs", type=float, nargs="+", default=REL_EBS)
    p.add_argument("--output-dir", default=str(ROOT / "results"))
    args = p.parse_args()

    DATASETS = get_datasets(args.data_dir)
    out_dir = Path(args.output_dir)

    all_results = {}
    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}, skipping")
            continue
        print(f"\n{'*' * 70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'*' * 70}")
        results = run_dataset(ds_name, DATASETS[ds_name], args.compressors,
                              args.rel_ebs, out_dir)
        all_results[ds_name] = results

    if all_results:
        combined = out_dir / "ablation_all.json"
        with open(combined, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined: {combined}")
        print_summary(all_results)


def print_summary(all_results):
    print(f"\n{'=' * 70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Config':<40s} {'full':>8s} {'no_alpha':>10s} "
          f"{'no_cov':>10s} {'shuffled':>10s}")
    print(f"  {'─' * 64}")
    for ds_name, results in all_results.items():
        for k, entry in results.items():
            parts = k.split("|")
            cmp = parts[0]
            for qoi in ("mean_square", "weighted_sum"):
                if qoi not in entry:
                    continue
                r = entry[qoi]
                label = f"{ds_name}/{cmp}/{qoi[:6]}"
                if "x_field" in entry:
                    label = f"{ds_name}/{cmp}/ws"
                else:
                    field = entry.get("field", "")
                    field_short = field.split("_")[0] if field else ""
                    label = f"{ds_name}/{cmp}/{field_short}"
                vals = []
                for mode in ("full", "no_alpha", "no_cov", "shuffled"):
                    if mode in r:
                        vals.append(f"{r[mode]['z_std']:.3f}")
                    else:
                        vals.append("  ---")
                print(f"  {label:<40s} {vals[0]:>8s} {vals[1]:>10s} "
                      f"{vals[2]:>10s} {vals[3]:>10s}")
    print(f"{'=' * 70}")
    print(f"  full: complete model.  no_alpha: alpha=1.  "
          f"no_cov: Cov(x,e)=0.  shuffled: i.i.d. control.")
    print(f"  sigma_z >> 1 under no_alpha/no_cov shows the component is needed.")
    print(f"  shuffled ~ 1 confirms the correlation model is correct.")


if __name__ == "__main__":
    main()
