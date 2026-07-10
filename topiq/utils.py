from __future__ import annotations

import math
import subprocess
import tempfile
import os
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from .error_tensor import ErrorTensor


# ── QoI graph definitions ───────────────────────────────────────────────

GRAPH_MEAN_SQUARE = [
    {"op": "square", "args": ["input"], "output": "x_sq"},
    {"op": "mean", "args": ["x_sq", "N"], "output": "mean_sq"},
]


def logistic(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


# ── Block iteration ─────────────────────────────────────────────────────

def iter_blocks(shape, block_shape):
    ranges = [range(0, s, b) for s, b in zip(shape, block_shape)]
    for starts in product(*ranges):
        sl = tuple(slice(s, min(s + b, dim))
                   for s, b, dim in zip(starts, block_shape, shape))
        yield sl


def iter_full_blocks(shape, block_shape):
    def _rec(dim, prefix):
        if dim == len(shape):
            yield prefix
            return
        step = block_shape[dim]
        size = shape[dim]
        if size < step:
            return
        for start in range(0, size - step + 1, step):
            yield from _rec(dim + 1, prefix + (slice(start, start + step),))
    yield from _rec(0, tuple())


# ── Alpha computation ───────────────────────────────────────────────────

def compute_alpha_raw(err_field, alpha_block):
    err = err_field.astype(np.float64, copy=False)
    pixel_var = float(np.var(err, ddof=0))
    result = {
        "alpha_raw": float("nan"),
        "pixel_err_var": pixel_var,
        "n_base": int(np.prod(alpha_block)),
        "num_blocks_used": 0,
    }
    if pixel_var <= 0:
        return result

    full_blocks = list(iter_full_blocks(err.shape, alpha_block))
    use_blocks = full_blocks if len(full_blocks) >= 2 else list(iter_blocks(err.shape, alpha_block))
    if len(use_blocks) < 2:
        return result

    sums = np.array([float(np.sum(err[sl], dtype=np.float64)) for sl in use_blocks])
    sizes = np.array([int(np.prod(err[sl].shape)) for sl in use_blocks], dtype=np.float64)
    n_ref = float(np.mean(sizes))
    if n_ref <= 0:
        return result

    var_sums = float(np.var(sums, ddof=0))
    alpha = var_sums / (n_ref * pixel_var)
    result.update({
        "alpha_raw": float(alpha),
        "n_base": int(round(n_ref)),
        "num_blocks_used": len(use_blocks),
    })
    return result


# ── Block metadata ──────────────────────────────────────────────────────

def build_block_meta(orig, dec, meta_block):
    meta = []
    for sl in iter_blocks(orig.shape, meta_block):
        o = orig[sl].astype(np.float64).ravel()
        d = dec[sl].astype(np.float64).ravel()
        e = d - o
        meta.append({
            "slices": sl,
            "mu": float(np.mean(o)),
            "mu_sq": float(np.mean(o ** 2)),
            "var_err": float(np.var(e, ddof=0)),
            "cov_xe": float(np.mean(o * e) - np.mean(o) * np.mean(e)),
        })
    return meta


def stats_from_blockmeta(meta, slices):
    tw = wm = wm2 = wve = wcov = 0.0
    for b in meta:
        area = 1
        for s_test, s_meta in zip(slices, b["slices"]):
            lo = max(s_test.start, s_meta.start)
            hi = min(s_test.stop, s_meta.stop)
            if hi <= lo:
                area = 0
                break
            area *= (hi - lo)
        if area <= 0:
            continue
        tw += area
        wm += area * b["mu"]
        wm2 += area * b["mu_sq"]
        wve += area * b["var_err"]
        wcov += area * b["cov_xe"]
    if tw == 0:
        return None
    mu = wm / tw
    return {"mu": mu, "var": max(wm2 / tw - mu * mu, 0.0),
            "var_err": wve / tw, "cov_xe": wcov / tw}


# ── Tensor constructors ─────────────────────────────────────────────────

def get_uv_tensor(orig_patch, dec_patch, alpha_info, n_actual):
    err = (dec_patch - orig_patch).astype(np.float64, copy=False)
    orig = orig_patch.astype(np.float64, copy=False)
    cov = float(np.cov(orig.ravel(), err.ravel(), ddof=0)[0, 1]) if orig.size > 1 else 0.0
    t = ErrorTensor(
        float(np.mean(orig)),
        float(np.var(orig, ddof=0)),
        0.0,
        float(np.var(err, ddof=0)),
        cov_xe=cov,
    )
    if alpha_info is not None and np.isfinite(alpha_info["alpha_raw"]):
        t = t.with_alpha_base(alpha_info["alpha_raw"],
                              int(alpha_info["n_base"]), int(n_actual))
    return t


def get_uv_tensor_from_meta(meta, alpha_info, slices, n_actual):
    s = stats_from_blockmeta(meta, slices)
    if s is None:
        return None
    t = ErrorTensor(s["mu"], s["var"], 0.0, s["var_err"], cov_xe=s["cov_xe"])
    if alpha_info is not None and np.isfinite(alpha_info["alpha_raw"]):
        t = t.with_alpha_base(alpha_info["alpha_raw"],
                              int(alpha_info["n_base"]), int(n_actual))
    return t


def make_tensor(orig_patch, dec_patch, alpha_info, n_actual, *,
                zero_alpha=False, zero_cov=False):
    err = (dec_patch - orig_patch).astype(np.float64)
    orig = orig_patch.astype(np.float64)
    cov = float(np.cov(orig.ravel(), err.ravel(), ddof=0)[0, 1]) if orig.size > 1 else 0.0
    if zero_cov:
        cov = 0.0
    t = ErrorTensor(
        float(np.mean(orig)), float(np.var(orig, ddof=0)),
        0.0, float(np.var(err, ddof=0)), cov_xe=cov)
    if zero_alpha:
        return t
    if alpha_info is not None and np.isfinite(alpha_info["alpha_raw"]):
        t = t.with_alpha_base(alpha_info["alpha_raw"],
                              int(alpha_info["n_base"]), n_actual)
    return t


# ── Numpy graph executor ────────────────────────────────────────────────

def execute_graph_numpy(graph, inputs):
    ctx = dict(inputs)

    def get(value):
        return ctx[value] if isinstance(value, str) and value in ctx else value

    for node in graph:
        op = node["op"]
        args = [get(arg) for arg in node.get("args", [])]
        out = node["output"]
        if op == "add":
            ctx[out] = args[0] + args[1]
        elif op == "sub":
            ctx[out] = args[0] - args[1]
        elif op == "mul":
            ctx[out] = args[0] * args[1]
        elif op == "div":
            ctx[out] = args[0] / args[1]
        elif op == "sigmoid":
            ctx[out] = logistic(args[0])
        elif op == "sum":
            ctx[out] = float(np.sum(np.asarray(args[0], dtype=np.float64)))
        elif op == "mean":
            ctx[out] = float(np.mean(np.asarray(args[0], dtype=np.float64)))
        elif op == "square":
            ctx[out] = np.asarray(args[0], dtype=np.float64) ** 2
        else:
            raise ValueError(f"Unsupported numpy op: {op}")
    return ctx


# ── Bias correction ─────────────────────────────────────────────────────

def compute_bias_correction(err, eval_block):
    g = tuple(err.shape[d] // eval_block[d] for d in range(err.ndim))
    sl = tuple(slice(0, g[d] * eval_block[d]) for d in range(err.ndim))
    a = err[sl]
    newshape = []
    for d in range(err.ndim):
        newshape += [g[d], eval_block[d]]
    a = a.reshape(newshape)
    block_axes = tuple(2 * d + 1 for d in range(err.ndim))
    block_means = a.mean(axis=block_axes).ravel()

    if block_means.size < 3:
        return 0.0
    mean = float(np.mean(block_means))
    return mean


# ── Summary accumulator ─────────────────────────────────────────────────

class SummaryAccumulator:
    def __init__(self):
        self.count = 0
        self.skipped = 0
        self.sum_z = 0.0
        self.sum_z2 = 0.0
        self.cov1 = 0
        self.cov2 = 0
        self.cov3 = 0

    def update(self, real_bias, pred_bias, pred_var):
        real = np.asarray(real_bias, dtype=np.float64)
        real = real[np.isfinite(real)]
        if real.size == 0:
            return

        pred_std = math.sqrt(max(float(pred_var), 0.0))
        if pred_std < 1e-12:
            tol = 1e-15 + 1e-12 * max(1.0, abs(float(pred_bias)))
            exact = np.abs(real - float(pred_bias)) <= tol
            n = int(np.count_nonzero(exact))
            self.skipped += int(real.size) - n
            if n == 0:
                return
            self.count += n
            self.cov1 += n
            self.cov2 += n
            self.cov3 += n
            return

        z = (real - float(pred_bias)) / pred_std
        z = z[np.isfinite(z)]
        if z.size == 0:
            return

        n = int(z.size)
        self.count += n
        self.sum_z += float(np.sum(z))
        self.sum_z2 += float(np.sum(z * z))
        self.cov1 += int(np.sum(np.abs(z) <= 1.0))
        self.cov2 += int(np.sum(np.abs(z) <= 2.0))
        self.cov3 += int(np.sum(np.abs(z) <= 3.0))

    def to_dict(self):
        if self.count == 0:
            return {
                "num_points": 0, "skipped_points": int(self.skipped),
                "z_mean": float("nan"), "z_std": float("nan"),
                "coverage_1sigma": float("nan"),
                "coverage_2sigma": float("nan"),
                "coverage_3sigma": float("nan"),
            }
        n = float(self.count)
        z_mean = self.sum_z / n
        z_var = max(self.sum_z2 / n - z_mean * z_mean, 0.0)
        return {
            "num_points": int(self.count),
            "skipped_points": int(self.skipped),
            "z_mean": float(z_mean),
            "z_std": float(math.sqrt(z_var)),
            "coverage_1sigma": float(self.cov1 / n),
            "coverage_2sigma": float(self.cov2 / n),
            "coverage_3sigma": float(self.cov3 / n),
        }


# ── Compression wrappers ────────────────────────────────────────────────

def compress_sz3(fpath, abs_eb, shape, dtype):
    ndim = len(shape)
    tf = "-d" if dtype == np.float64 else "-f"
    with tempfile.TemporaryDirectory() as td:
        zf = os.path.join(td, "c.sz")
        of = os.path.join(td, "d.raw")
        subprocess.run(
            ["sz3", tf, "-i", fpath, "-z", zf, f"-{ndim}",
             *[str(s) for s in reversed(shape)], "-M", "ABS", "-A", str(abs_eb)],
            capture_output=True, check=True)
        subprocess.run(
            ["sz3", tf, "-z", zf, "-o", of, f"-{ndim}",
             *[str(s) for s in reversed(shape)]],
            capture_output=True, check=True)
        return np.fromfile(of, dtype=dtype).reshape(shape)


def compress_sperr(fpath, abs_eb, shape, dtype):
    ndim = len(shape)
    ft = "64" if dtype == np.float64 else "32"
    dims = [str(s) for s in reversed(shape)]
    sperr_bin = "sperr3d" if ndim == 3 else "sperr2d"
    df = "--decomp_d" if dtype == np.float64 else "--decomp_f"
    with tempfile.TemporaryDirectory() as td:
        bf = os.path.join(td, "c.sperr")
        of = os.path.join(td, "d.raw")
        subprocess.run(
            [sperr_bin, "-c", fpath, "--ftype", ft, "--dims", *dims,
             "--pwe", str(abs_eb), "--bitstream", bf, df, of],
            capture_output=True, check=True)
        return np.fromfile(of, dtype=dtype).reshape(shape)


def compress_zfp(fpath, abs_eb, shape, dtype):
    ndim = len(shape)
    tf = "-d" if dtype == np.float64 else "-f"
    with tempfile.TemporaryDirectory() as td:
        zf = os.path.join(td, "c.zfp")
        of = os.path.join(td, "d.raw")
        subprocess.run(
            ["zfp", tf, f"-{ndim}", *[str(s) for s in reversed(shape)],
             "-i", fpath, "-z", zf, "-a", str(abs_eb), "-h", "-q"],
            capture_output=True, check=True)
        subprocess.run(
            ["zfp", "-h", "-z", zf, "-o", of, "-q"],
            capture_output=True, check=True)
        return np.fromfile(of, dtype=dtype).reshape(shape)


def compress(fpath, abs_eb, cmp, shape, dtype):
    if cmp == "sz3":
        return compress_sz3(fpath, abs_eb, shape, dtype)
    elif cmp == "sperr":
        return compress_sperr(fpath, abs_eb, shape, dtype)
    elif cmp == "zfp":
        return compress_zfp(fpath, abs_eb, shape, dtype)
    raise ValueError(f"Unknown compressor: {cmp}")


# ── Error shuffling (for ablation) ──────────────────────────────────────

def shuffle_error(orig, dec, seed=42):
    err = (dec.astype(np.float64) - orig.astype(np.float64))
    rng = np.random.default_rng(seed)
    flat = err.ravel().copy()
    rng.shuffle(flat)
    return (orig.astype(np.float64) + flat.reshape(orig.shape)).astype(orig.dtype)
