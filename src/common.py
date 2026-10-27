"""共享工具：只读连接、预测模型指标、增量价值统计（NRI / IDI / 决策曲线）。

相对 C_prior_history_value 的唯一改动：
    bootstrap 一律**按患者聚类**重抽样（同一患者多次培养高度相关，
    朴素 bootstrap 会低估方差）。其余函数逐行沿用，保证与 C 项目可比。

本模块不写入任何数据文件，也不做个体级导出。
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # 中文 Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
DOCS = PROJECT / "docs"

import os

# Database locations. Put the DuckDB files in <project>/data/, or point the
# environment variables at them (ARMD_DB, MIMIC_DB, or MV_DATA_DIR for both).
DATA_DIR = Path(os.environ.get("MV_DATA_DIR", str(PROJECT / "data")))
ARMD_DB = Path(os.environ.get("ARMD_DB", str(Path(DATA_DIR) / "ARMD_MGB.duckdb")))
MIMIC_DB = Path(os.environ.get("MIMIC_DB", str(Path(DATA_DIR) / "mimiciv_v3_1.duckdb")))

SEED = 20261024


def connect(path: Path, threads: int = 8) -> duckdb.DuckDBPyConnection:
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {path}\n"
            "The scripts open both DuckDB files read-only and never copy them.\n"
            "Place the files in <project>/data/, or set ARMD_DB / MIMIC_DB "
            "(or MV_DATA_DIR) to their location."
        )
    con = duckdb.connect(str(path), read_only=True)
    con.execute(f"SET threads={threads}")
    return con


def ensure_dirs() -> None:
    for d in (TABLES, FIGURES, DOCS):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 基本性能
# ---------------------------------------------------------------------------
def metric_block(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        slope, intercept = calib_slope_intercept(y, p)
        out["calib_slope"] = slope
        out["calib_intercept"] = intercept
    else:
        out.update({"auroc": np.nan, "auprc": np.nan,
                    "calib_slope": np.nan, "calib_intercept": np.nan})
    return out


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, min(centre - half, p))
    hi = min(1.0, max(centre + half, p))
    return (lo, hi)


def calib_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    lp = np.log(pc / (1 - pc)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(lp, np.asarray(y).astype(int))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """十分位校准分箱（按预测概率排序切分）。"""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    chunks = np.array_split(order, n_bins)
    out = []
    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            continue
        lo, hi = wilson_ci(int(y[idx].sum()), len(idx))
        out.append({
            "bin": i + 1, "n": int(len(idx)),
            "pred_mean": float(p[idx].mean()), "obs_rate": float(y[idx].mean()),
            "obs_lo": lo, "obs_hi": hi,
        })
    return out


# ---------------------------------------------------------------------------
# 增量价值：NRI / IDI
# ---------------------------------------------------------------------------
def nri_continuous(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    up = p_new > p_old
    down = p_new < p_old
    ev = y == 1
    ne = ~ev
    nri_e = float(up[ev].mean() - down[ev].mean()) if ev.sum() else np.nan
    nri_ne = float(down[ne].mean() - up[ne].mean()) if ne.sum() else np.nan
    return {"nri_events": nri_e, "nri_nonevents": nri_ne, "nri": nri_e + nri_ne}


def idi(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    ev = y == 1
    ne = ~ev
    return float((p_new[ev].mean() - p_old[ev].mean())
                 - (p_new[ne].mean() - p_old[ne].mean()))


# ---------------------------------------------------------------------------
# 决策曲线
# ---------------------------------------------------------------------------
def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    y = np.asarray(y).astype(int)
    n = len(y)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def net_benefit_treat_all(y: np.ndarray, threshold: float) -> float:
    y = np.asarray(y).astype(int)
    prev = y.mean()
    return prev - (1 - prev) * (threshold / (1 - threshold))


# ---------------------------------------------------------------------------
# bootstrap（按患者聚类）
# ---------------------------------------------------------------------------
def _cluster_index(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """按患者有放回抽患者，再把该患者的全部培养拼起来（向量化）。"""
    _, inverse = np.unique(groups, return_inverse=True)
    k = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=k)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    order = np.argsort(inverse, kind="stable")
    rep = rng.multinomial(k, np.full(k, 1.0 / k))
    take = np.repeat(np.arange(k), rep)
    lens = counts[take]
    starts = offsets[take]
    sel = np.repeat(starts, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    return order[sel]


def bootstrap_delta_auroc(y, p_old, p_new, groups=None, B: int = 1000, seed: int = SEED) -> dict:
    """配对 bootstrap：ΔAUROC = AUROC(new) − AUROC(old)。groups 给出时按患者聚类。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    if groups is None:
        groups = np.arange(n)
    groups = np.asarray(groups)
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, n) if groups is None else _cluster_index(groups, rng)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(roc_auc_score(yy, p_new[idx]) - roc_auc_score(yy, p_old[idx]))
    d = np.array(deltas)
    obs = roc_auc_score(y, p_new) - roc_auc_score(y, p_old)
    return {
        "delta_auroc": float(obs),
        "delta_lo": float(np.percentile(d, 2.5)),
        "delta_hi": float(np.percentile(d, 97.5)),
        "p_gt0": float((d > 0).mean()),
        "n_boot": int(len(d)),
    }


def bootstrap_auroc(y, p, groups=None, B: int = 1000, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n) if groups is None else _cluster_index(groups, rng)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, p[idx]))
    v = np.array(vals)
    return {"auroc_lo": float(np.percentile(v, 2.5)), "auroc_hi": float(np.percentile(v, 97.5))}


def bootstrap_nri_idi(y, p_old, p_new, groups=None, B: int = 1000, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    nris, idis = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n) if groups is None else _cluster_index(groups, rng)
        yy, po, pn = y[idx], p_old[idx], p_new[idx]
        if len(np.unique(yy)) < 2:
            continue
        nris.append(nri_continuous(yy, po, pn)["nri"])
        idis.append(idi(yy, po, pn))
    nris, idis = np.array(nris), np.array(idis)
    return {
        "nri_lo": float(np.percentile(nris, 2.5)),
        "nri_hi": float(np.percentile(nris, 97.5)),
        "idi_lo": float(np.percentile(idis, 2.5)),
        "idi_hi": float(np.percentile(idis, 97.5)),
    }
