"""S10 MIMIC 时间外推的校准：为什么必须重新估计截距，以及重新校准后长什么样。

背景：训练期(2008–2014)耐药率 12.2%，测试期(2015–2022) 21.8%。
直接用训练集模型 -> 系统性低估。本脚本给出三种状态：

    raw        原始预测
    LR-IC      只重估截距（recalibration-in-the-large）
    LR-full    重估斜率+截距（Cox 校准）

产出：results/tables/c10_recalibration.csv, c10_year_auroc.csv, c10_calibration_bins.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from analysis import MODELS, run_models
from common import ARMD_DB, MIMIC_DB, TABLES, calib_slope_intercept, connect, ensure_dirs, metric_block
from features import build_armd, build_mimic, prepare


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def recalibrate(y, p):
    """返回 (p_IC, p_full, delta, slope, intercept)。

    LR-IC 的截距用**极大似然**求解（score equation Σ(y−σ(lp+a))=0），
    而不是"均值 logit 相减"——后者受 Jensen 不等式影响会高估校准后的风险。
    """
    from scipy.optimize import brentq

    y = np.asarray(y).astype(int)
    lp = logit(p)
    delta = float(brentq(lambda a: float(np.sum(y - sigmoid(lp + a))), -30, 30))
    p_ic = sigmoid(lp + delta)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(lp.reshape(-1, 1), y)
    p_full = lr.predict_proba(lp.reshape(-1, 1))[:, 1]
    return p_ic, p_full, delta, float(lr.coef_[0][0]), float(lr.intercept_[0])


def bins(y, p, n_bins=10):
    order = np.argsort(p)
    out = []
    for i, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx) == 0:
            continue
        out.append({"bin": i + 1, "n": len(idx),
                    "pred": float(np.asarray(p)[idx].mean()),
                    "obs": float(np.asarray(y)[idx].mean())})
    return out


def main():
    ensure_dirs()
    con = connect(MIMIC_DB)
    mimic = prepare(build_mimic(con))
    con.close()
    con = connect(ARMD_DB)
    armd = prepare(build_armd(con))
    con.close()

    rows, bin_rows, year_rows = [], [], []
    m3, m1 = "M3 +既往同菌药敏", "M1 +既往用药"

    # ---- MIMIC 时间外推 ----
    test = (mimic["year"] >= 2015).to_numpy()
    preds = run_models(mimic, test_mask=test)
    y = mimic["y"].to_numpy(int)[test]
    for name in (m1, m3):
        p = preds[name][test]
        p_ic, p_full, delta, slope, icept = recalibrate(y, p)
        for stage, pp in (("raw", p), ("LR-IC", p_ic), ("LR-full", p_full)):
            m = metric_block(y, pp)
            rows.append({"cohort": "MIMIC-IV", "model": name, "stage": stage,
                         "n": len(y), "events": int(y.sum()),
                         "auroc": m["auroc"], "brier": m["brier"],
                         "calib_slope": m["calib_slope"], "calib_intercept": m["calib_intercept"],
                         "mean_pred": float(np.mean(pp)), "obs_rate": float(y.mean()),
                         "delta_ic": delta, "fit_slope": slope, "fit_intercept": icept})
        for tag, pp in (("raw", p), ("LR-IC", p_ic)):
            for b in bins(y, pp):
                b.update(cohort="MIMIC-IV", model=name, stage=tag)
                bin_rows.append(b)

    # 逐年判别力
    yrs = mimic.loc[test, "year"].to_numpy()
    p_m3 = preds[m3][test]
    for yr in np.unique(yrs):
        mask = yrs == yr
        if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
            continue
        year_rows.append({"cohort": "MIMIC-IV", "year": int(yr), "n": int(mask.sum()),
                          "events": int(y[mask].sum()), "obs_rate": float(y[mask].mean()),
                          "mean_pred": float(p_m3[mask].mean()),
                          "auroc": float(metric_block(y[mask], p_m3[mask])["auroc"])})

    # ---- ARMD（CV，校准作为对照）----
    g = armd["patient_id"].to_numpy()
    preds_a = run_models(armd, groups=g)
    ya = armd["y"].to_numpy(int)
    for name in (m1, m3):
        p = preds_a[name]
        p_ic, p_full, delta, slope, icept = recalibrate(ya, p)
        for stage, pp in (("raw", p), ("LR-IC", p_ic), ("LR-full", p_full)):
            m = metric_block(ya, pp)
            rows.append({"cohort": "ARMD-MGB", "model": name, "stage": stage,
                         "n": len(ya), "events": int(ya.sum()),
                         "auroc": m["auroc"], "brier": m["brier"],
                         "calib_slope": m["calib_slope"], "calib_intercept": m["calib_intercept"],
                         "mean_pred": float(np.mean(pp)), "obs_rate": float(ya.mean()),
                         "delta_ic": delta, "fit_slope": slope, "fit_intercept": icept})
        for tag, pp in (("raw", p), ("LR-IC", p_ic)):
            for b in bins(ya, pp):
                b.update(cohort="ARMD-MGB", model=name, stage=tag)
                bin_rows.append(b)

    pd.DataFrame(rows).to_csv(TABLES / "c10_recalibration.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(TABLES / "c10_calibration_bins.csv", index=False)
    pd.DataFrame(year_rows).to_csv(TABLES / "c10_year_auroc.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== 校准三态 ===")
    print(df[["cohort", "model", "stage", "auroc", "brier", "calib_slope",
              "calib_intercept", "mean_pred", "obs_rate"]].round(3).to_string(index=False))
    print("\n=== MIMIC 逐年 ===")
    print(pd.DataFrame(year_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
