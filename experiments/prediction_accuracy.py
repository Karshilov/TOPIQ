#!/usr/bin/env python3
"""
Prediction accuracy evaluation (Table 2, Figures 4-5).

Evaluates across 4 datasets, 3 compressors, 8 error bounds, 5 QoI families:
  1. mean_square: mean(x^2)
  2. neural_net: mean(W2 * sigmoid(W1 * x + b1) + b2)
  3. weighted_sum: sum(x * w)
  4. cloudy_ratio: multi-field cloud radiative effect (CESM only)
  5. random_block: mean_square on random regions with meta interpolation (CESM only)

Usage:
    python prediction_accuracy.py --data-dir /path/to/data [--datasets CESM,NYX]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topiq.error_tensor import ErrorTensor
from topiq.executor import execute_graph
from topiq.utils import (
    compute_alpha_raw, iter_blocks, build_block_meta,
    get_uv_tensor, get_uv_tensor_from_meta,
    execute_graph_numpy, SummaryAccumulator,
    GRAPH_MEAN_SQUARE, logistic,
    compress, compute_bias_correction,
)

REL_EBS = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5]
COMPRESSORS = ["sz3", "sperr", "zfp"]

CLOUDY_FIELDS = [
    "CLDTOT_1_1800_3600.f32",
    "FLUT_1_1800_3600.f32",
    "FLUTC_1_1800_3600.f32",
]

GRAPH_CLOUDY = [
    {"op": "sub", "args": [CLOUDY_FIELDS[2], CLOUDY_FIELDS[1]], "output": "X"},
    {"op": "sub", "args": [CLOUDY_FIELDS[0], "tau"], "output": "d"},
    {"op": "mul", "args": ["d", "k"], "output": "kd"},
    {"op": "sigmoid", "args": ["kd"], "output": "m"},
    {"op": "mul", "args": ["m", "X"], "output": "mx"},
    {"op": "sum", "args": ["mx", "N"], "output": "num"},
    {"op": "sum", "args": ["m", "N"], "output": "den"},
    {"op": "add", "args": ["den", "eps_sum"], "output": "den_eps"},
    {"op": "div", "args": ["num", "den_eps"], "output": "cloudy"},
]

CLOUDY_TAU = 0.5
CLOUDY_K = 5.0
CLOUDY_EPS0 = 1e-6


def _inject_bias(tensor, bias_correction):
    if bias_correction != 0.0:
        tensor.bias = bias_correction
    return tensor


def eval_mean_square(orig, dec, alpha, eval_block, meta=None,
                     bias_correction=0.0):
    full_n = int(np.prod(eval_block))
    acc_uv = SummaryAccumulator()
    acc_bm = SummaryAccumulator()
    for sl in iter_blocks(orig.shape, eval_block):
        b0, b1 = orig[sl], dec[sl]
        n = int(np.prod(b0.shape))
        if n < full_n:
            continue
        real = float(np.mean(b1.astype(np.float64) ** 2) -
                     np.mean(b0.astype(np.float64) ** 2))
        t_uv = _inject_bias(get_uv_tensor(b0, b1, alpha, n), bias_correction)
        out_uv = execute_graph(
            GRAPH_MEAN_SQUARE, {"input": t_uv, "N": n})["mean_sq"]
        acc_uv.update(real, out_uv.bias, out_uv.var_err)
        if meta is not None:
            t_bm = get_uv_tensor_from_meta(meta, alpha, sl, n)
            if t_bm is not None:
                _inject_bias(t_bm, bias_correction)
                out_bm = execute_graph(
                    GRAPH_MEAN_SQUARE, {"input": t_bm, "N": n})["mean_sq"]
                acc_bm.update(real, out_bm.bias, out_bm.var_err)
    return acc_uv.to_dict(), acc_bm.to_dict()


def eval_weighted_sum(x_orig, x_dec, w_orig, w_dec, ax, aw,
                      eval_block, bias_correction_x=0.0,
                      bias_correction_w=0.0):
    full_n = int(np.prod(eval_block))
    acc = SummaryAccumulator()
    for sl in iter_blocks(x_orig.shape, eval_block):
        bx0, bx1 = x_orig[sl], x_dec[sl]
        bw0, bw1 = w_orig[sl], w_dec[sl]
        n = int(np.prod(bx0.shape))
        if n < full_n:
            continue
        real = float(np.sum(bx1.astype(np.float64) * bw1.astype(np.float64) -
                            bx0.astype(np.float64) * bw0.astype(np.float64)))
        tx = _inject_bias(get_uv_tensor(bx0, bx1, ax, n), bias_correction_x)
        tw = _inject_bias(get_uv_tensor(bw0, bw1, aw, n), bias_correction_w)
        out = (tx * tw).sum(n)
        acc.update(real, out.bias, out.var_err)
    return acc.to_dict()


def _make_block_nn(n_pix, H=32):
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(n_pix, H), nn.Sigmoid(), nn.Linear(H, 1))
    with torch.no_grad():
        model[0].weight.normal_(0, 1.0 / np.sqrt(n_pix))
        model[0].bias.zero_()
        model[2].weight.normal_(0, 1.0 / np.sqrt(H))
        model[2].bias.zero_()
    model.eval()
    return model


def eval_neural_net(orig, dec, alpha, eval_block, bias_correction=0.0):
    import torch
    full_n = int(np.prod(eval_block))
    acc = SummaryAccumulator()
    H = 32

    dmin = float(orig.min())
    dmax = float(orig.max())
    drange = max(dmax - dmin, 1.0)
    orig_norm = ((orig - dmin) / drange).astype(np.float32)
    dec_norm = ((dec - dmin) / drange).astype(np.float32)
    err_norm = dec_norm - orig_norm

    alpha_norm = compute_alpha_raw(err_norm, eval_block)

    model = _make_block_nn(full_n, H)
    W1 = model[0].weight.detach().numpy()
    b1_nn = model[0].bias.detach().numpy()
    W2 = model[2].weight.detach().numpy().flatten()
    b2_nn = model[2].bias.detach().numpy().item()
    W1_sum = W1.sum(axis=1)
    W1_sq_sum = (W1 ** 2).sum(axis=1)

    for sl in iter_blocks(orig.shape, eval_block):
        b0 = orig_norm[sl]
        b1_blk = dec_norm[sl]
        n = int(np.prod(b0.shape))
        if n < full_n:
            continue

        b0_flat = b0.astype(np.float64).ravel()
        b1_flat = b1_blk.astype(np.float64).ravel()
        e_flat = b1_flat - b0_flat

        with torch.no_grad():
            out_orig = model(torch.tensor(b0_flat, dtype=torch.float32).unsqueeze(0)).item()
            out_dec = model(torch.tensor(b1_flat, dtype=torch.float32).unsqueeze(0)).item()
        real = out_dec - out_orig

        mu = float(np.mean(b0_flat))
        sig2 = float(np.var(b0_flat))
        bv = float(np.mean(e_flat))
        vv = float(np.var(e_flat))
        cv = float(np.cov(b0_flat, e_flat)[0, 1]) if n > 1 else 0.0

        ar = alpha_norm['alpha_raw']
        nb = int(alpha_norm['n_base'])
        ra = max(0.0, min((ar - 1.0) / (nb - 1.0), 1.0)) if nb > 1 else 0.0
        an = 1.0 + (n - 1.0) * ra
        vc = max(0.0, min(((an - 1.0) * vv) / (n - 1.0), vv)) if n > 1 else 0.0
        vu = vv - vc

        pred_bias = b2_nn
        pred_var = 0.0
        for j in range(H):
            z_mu = mu * W1_sum[j] + b1_nn[j]
            z_var = W1_sq_sum[j] * vu + W1_sum[j] ** 2 * vc
            z_bias = bv * W1_sum[j]
            z_cov = cv * W1_sum[j]
            s = 1.0 / (1.0 + np.exp(-np.clip(z_mu, -500, 500)))
            ds = s * (1.0 - s)
            dds = ds * (1.0 - 2.0 * s)
            h_bias = ds * z_bias + 0.5 * dds * z_var + dds * z_cov
            h_var = ds ** 2 * z_var
            pred_bias += W2[j] * h_bias
            pred_var += W2[j] ** 2 * h_var

        acc.update(real, pred_bias, pred_var)

    return acc.to_dict()


def eval_cloudy(orig_map, dec_map, alpha_map, eval_block, bias_map=None):
    full_n = int(np.prod(eval_block))
    acc = SummaryAccumulator()
    fnames = CLOUDY_FIELDS
    if bias_map is None:
        bias_map = {}
    c0 = orig_map[fnames[0]]
    for sl in iter_blocks(c0.shape, eval_block):
        n = int(np.prod(c0[sl].shape))
        if n < full_n:
            continue
        eps_sum = CLOUDY_EPS0 * n
        consts = {"tau": CLOUDY_TAU, "k": CLOUDY_K, "N": n, "eps_sum": eps_sum}

        real0 = execute_graph_numpy(
            GRAPH_CLOUDY,
            {fn: orig_map[fn][sl] for fn in fnames} | consts)["cloudy"]
        real1 = execute_graph_numpy(
            GRAPH_CLOUDY,
            {fn: dec_map[fn][sl] for fn in fnames} | consts)["cloudy"]
        real = float(real1 - real0)

        out = execute_graph(GRAPH_CLOUDY, {
            fn: _inject_bias(
                get_uv_tensor(orig_map[fn][sl], dec_map[fn][sl],
                              alpha_map[fn], n),
                bias_map.get(fn, 0.0))
            for fn in fnames
        } | consts)["cloudy"]
        acc.update(real, out.bias, out.var_err)
    return acc.to_dict()


def eval_random_blocks(orig, dec, alpha, meta, n_samples=200,
                       block_size=(200, 200), seed=42):
    rng = np.random.default_rng(seed)
    h, w = orig.shape
    bh, bw = block_size
    acc_uv = SummaryAccumulator()
    acc_bm = SummaryAccumulator()
    for _ in range(n_samples):
        r = rng.integers(0, max(1, h - bh))
        c = rng.integers(0, max(1, w - bw))
        sl = (slice(r, r + bh), slice(c, c + bw))
        b0, b1 = orig[sl], dec[sl]
        n = int(np.prod(b0.shape))
        real = float(np.mean(b1.astype(np.float64) ** 2) -
                     np.mean(b0.astype(np.float64) ** 2))
        out_uv = execute_graph(
            GRAPH_MEAN_SQUARE,
            {"input": get_uv_tensor(b0, b1, alpha, n), "N": n})["mean_sq"]
        acc_uv.update(real, out_uv.bias, out_uv.var_err)
        t_bm = get_uv_tensor_from_meta(meta, alpha, sl, n)
        if t_bm is not None:
            out_bm = execute_graph(
                GRAPH_MEAN_SQUARE, {"input": t_bm, "N": n})["mean_sq"]
            acc_bm.update(real, out_bm.bias, out_bm.var_err)
    return acc_uv.to_dict(), acc_bm.to_dict()


# ── Dataset definitions ─────────────────────────────────────────────────

def get_datasets(data_dir):
    d = Path(data_dir)
    return {
        "CESM": {
            "data_dir": d,
            "shape": (1800, 3600), "dtype": np.float32,
            "fields": ["CLDTOT_1_1800_3600.f32", "CLDHGH_1_1800_3600.f32"],
            "ws_pairs": [("CLDTOT_1_1800_3600.f32", "CLDHGH_1_1800_3600.f32")],
            "cloudy_fields": CLOUDY_FIELDS,
            "eval_block": (200, 200),
            "alpha_block": (200, 200),
            "meta_block": (150, 150),
        },
        "NYX": {
            "data_dir": d,
            "shape": (512, 512, 512), "dtype": np.float32,
            "fields": ["dark_matter_density_log.f32", "baryon_density_log.f32",
                       "velocity_x.f32"],
            "ws_pairs": [("dark_matter_density_log.f32", "velocity_x.f32")],
            "eval_block": (32, 32, 32),
            "alpha_block": (32, 32, 32),
            "meta_block": (24, 24, 24),
        },
        "SCALE": {
            "data_dir": d,
            "shape": (98, 1200, 1200), "dtype": np.float32,
            "fields": ["PRES-98x1200x1200.f32", "T-98x1200x1200.f32"],
            "ws_pairs": [("PRES-98x1200x1200.f32", "T-98x1200x1200.f32")],
            "eval_block": (32, 32, 32),
            "alpha_block": (32, 32, 32),
            "meta_block": (24, 24, 24),
        },
        "Hurricane": {
            "data_dir": d,
            "shape": (100, 500, 500), "dtype": np.float32,
            "fields": ["TCf48.bin.f32", "Uf48.bin.f32"],
            "ws_pairs": [("TCf48.bin.f32", "Uf48.bin.f32")],
            "eval_block": (32, 32, 32),
            "alpha_block": (32, 32, 32),
            "meta_block": (24, 24, 24),
        },
    }


def run_dataset(ds_name, ds_cfg, out_dir):
    data_dir = ds_cfg["data_dir"]
    shape = ds_cfg["shape"]
    dtype = ds_cfg["dtype"]
    fields = ds_cfg["fields"]
    eval_block = ds_cfg["eval_block"]
    alpha_block = ds_cfg["alpha_block"]
    meta_block = ds_cfg["meta_block"]

    results = {}

    for cmp in COMPRESSORS:
        print(f"\n{'#' * 70}")
        print(f"  [{ds_name}] Compressor: {cmp.upper()}")
        print(f"{'#' * 70}")

        for reb in REL_EBS:
            orig_map, dec_map, alpha_map, meta_map = {}, {}, {}, {}
            all_fields = list(fields)
            if ds_name == "CESM" and ds_cfg.get("cloudy_fields"):
                for f in ds_cfg["cloudy_fields"]:
                    if f not in all_fields:
                        all_fields.append(f)

            skip_eb = False
            for fname in all_fields:
                fpath = str(data_dir / fname)
                if not os.path.isfile(fpath):
                    print(f"  WARNING: {fpath} not found, skipping")
                    skip_eb = True
                    break
                orig = np.fromfile(fpath, dtype=dtype).reshape(shape)
                aeb = reb * float(orig.max() - orig.min())
                try:
                    dec = compress(fpath, aeb, cmp, shape, dtype)
                except Exception as e:
                    print(f"  SKIP {fname} {cmp} reb={reb:.0e}: {e}")
                    skip_eb = True
                    break
                err_f = dec.astype(np.float64) - orig.astype(np.float64)
                alpha = compute_alpha_raw(err_f, alpha_block)
                meta = build_block_meta(orig, dec, meta_block)
                orig_map[fname] = orig
                dec_map[fname] = dec
                alpha_map[fname] = alpha
                meta_map[fname] = meta
                print(f"  {fname} reb={reb:.0e} alpha={alpha['alpha_raw']:.1f}")

            if skip_eb:
                continue

            bias_map = {}
            for fname in all_fields:
                if fname not in dec_map:
                    continue
                err_f = dec_map[fname].astype(np.float64) - orig_map[fname].astype(np.float64)
                bc = compute_bias_correction(err_f, eval_block)
                bias_map[fname] = bc
                if bc != 0.0:
                    print(f"  BIAS {fname}: correction={bc:+.3e}")

            entry = {"dataset": ds_name, "compressor": cmp, "reb": reb}

            for fname in fields:
                key = f"{cmp}|{fname}|{reb:.0e}|mean_square"
                uv, bm = eval_mean_square(
                    orig_map[fname], dec_map[fname],
                    alpha_map[fname], eval_block, meta_map[fname],
                    bias_correction=bias_map.get(fname, 0.0))
                results[key] = {
                    **entry, "field": fname, "qoi": "mean_square",
                    "alpha": alpha_map[fname]["alpha_raw"],
                    "uv": uv, "bmeta": bm,
                }
                print(f"  mean_sq {fname}: sigma_z={uv['z_std']:.2f} "
                      f"cov3={uv['coverage_3sigma']:.3f}")

            for fname in fields:
                key = f"{cmp}|{fname}|{reb:.0e}|neural_net"
                uv = eval_neural_net(
                    orig_map[fname], dec_map[fname],
                    alpha_map[fname], eval_block,
                    bias_correction=bias_map.get(fname, 0.0))
                results[key] = {
                    **entry, "field": fname, "qoi": "neural_net",
                    "alpha": alpha_map[fname]["alpha_raw"],
                    "uv": uv,
                }
                print(f"  nn {fname}: sigma_z={uv['z_std']:.2f} "
                      f"cov3={uv['coverage_3sigma']:.3f}")

            for x_name, w_name in ds_cfg.get("ws_pairs", []):
                if x_name not in orig_map or w_name not in orig_map:
                    continue
                key = f"{cmp}|ws|{x_name}:{w_name}|{reb:.0e}"
                uv = eval_weighted_sum(
                    orig_map[x_name], dec_map[x_name],
                    orig_map[w_name], dec_map[w_name],
                    alpha_map[x_name], alpha_map[w_name],
                    eval_block,
                    bias_correction_x=bias_map.get(x_name, 0.0),
                    bias_correction_w=bias_map.get(w_name, 0.0))
                results[key] = {
                    **entry, "qoi": "weighted_sum",
                    "x_field": x_name, "w_field": w_name, "uv": uv,
                }
                print(f"  ws {x_name} x {w_name}: sigma_z={uv['z_std']:.2f}")

            if ds_name == "CESM" and all(f in orig_map for f in CLOUDY_FIELDS):
                key = f"{cmp}|cloudy|{reb:.0e}"
                uv = eval_cloudy(orig_map, dec_map, alpha_map, eval_block,
                                 bias_map=bias_map)
                results[key] = {**entry, "qoi": "cloudy_ratio", "uv": uv}
                print(f"  cloudy: sigma_z={uv['z_std']:.2f}")

            if ds_name == "CESM":
                for fname in fields:
                    key = f"{cmp}|{fname}|{reb:.0e}|random_block"
                    uv, bm = eval_random_blocks(
                        orig_map[fname], dec_map[fname],
                        alpha_map[fname], meta_map[fname])
                    results[key] = {
                        **entry, "field": fname, "qoi": "random_block",
                        "alpha": alpha_map[fname]["alpha_raw"],
                        "uv": uv, "bmeta": bm,
                    }
                    print(f"  random {fname}: exact sigma_z={uv['z_std']:.2f} "
                          f"meta sigma_z={bm['z_std']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Root data directory containing dataset subdirs")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated dataset names or 'all'")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "results"))
    args = parser.parse_args()

    DATASETS = get_datasets(args.data_dir)

    if args.datasets == "all":
        ds_names = list(DATASETS.keys())
    else:
        ds_names = [s.strip() for s in args.datasets.split(",")]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for ds_name in ds_names:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}, skipping")
            continue
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'=' * 70}")
        results = run_dataset(ds_name, DATASETS[ds_name], out_dir)
        all_results[ds_name] = results

        out_path = out_dir / f"pred_acc_{ds_name}.json"
        with open(out_path, "w") as f:
            json.dump({ds_name: results}, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")

    combined = out_dir / "prediction_accuracy.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll done. Combined results: {combined}")

    print_summary(all_results)


def print_summary(all_results):
    total = in_range = total_1e3 = in_range_1e3 = 0
    per_ds = {}

    for ds_name, results in all_results.items():
        ds_t = ds_ir = ds_t3 = ds_ir3 = 0
        for k, v in results.items():
            if "uv" not in v or "random_block" in k:
                continue
            sz = v["uv"]["z_std"]
            ds_t += 1
            if 0.7 <= sz <= 1.3:
                ds_ir += 1
            if v.get("reb") == 0.001:
                ds_t3 += 1
                if 0.7 <= sz <= 1.3:
                    ds_ir3 += 1
        per_ds[ds_name] = (ds_t, ds_ir, ds_t3, ds_ir3)
        total += ds_t; in_range += ds_ir
        total_1e3 += ds_t3; in_range_1e3 += ds_ir3

    print(f"\n{'=' * 60}")
    print(f"  PREDICTION ACCURACY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  (excluding random_block evaluations)")
    print()
    for ds, (t, ir, t3, ir3) in per_ds.items():
        pct = 100 * ir / t if t else 0
        pct3 = 100 * ir3 / t3 if t3 else 0
        print(f"  {ds:12s}  all_ebs: {ir:3d}/{t:3d} ({pct:5.1f}%)  "
              f"reb=1e-3: {ir3:2d}/{t3:2d} ({pct3:5.1f}%)")
    print(f"  {'─' * 54}")
    if total > 0:
        print(f"  {'TOTAL':12s}  all_ebs: {in_range:3d}/{total:3d} "
              f"({100*in_range/total:5.1f}%)  "
              f"reb=1e-3: {in_range_1e3:2d}/{total_1e3:2d} "
              f"({100*in_range_1e3/total_1e3:5.1f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
