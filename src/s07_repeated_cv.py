"""S07 重复交叉验证：主增量对折分随机性有多稳？

ARMD：5 折患者分组 CV × 5 个种子
MIMIC：5 折患者分组 CV × 5 个种子（稳定性对照；主分析仍用时间外推）

产出：results/tables/c07_repeated_cv.csv（每个种子一行）+ 汇总打印
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from analysis import INCREMENTS, MODELS, fit_predict
from common import ARMD_DB, MIMIC_DB, TABLES, connect, ensure_dirs
from features import build_armd, build_mimic, prepare

SEEDS = [20261024, 1, 2, 3, 4]


def cv_predict(df, cols, groups, seed, n_splits=5):
    X = df[cols].to_numpy(float)
    y = df["y"].to_numpy(int)
    p = np.zeros(len(y))
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups)
    for tr, te in folds:
        p[te] = fit_predict(X[tr], y[tr], X[te])
    return p


def one_repeat(df, seed):
    g = df["patient_id"].to_numpy()
    y = df["y"].to_numpy(int)
    preds = {k: cv_predict(df, v, g, seed) for k, v in MODELS.items()}
    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "metric": "auroc", "value": roc_auc_score(y, p)})
    for old, new, label in INCREMENTS:
        rows.append({"model": label, "metric": "delta_auroc",
                     "value": roc_auc_score(y, preds[new]) - roc_auc_score(y, preds[old])})
    return rows


def main():
    ensure_dirs()
    con = connect(ARMD_DB)
    armd = prepare(build_armd(con))
    con.close()
    con = connect(MIMIC_DB)
    mimic = prepare(build_mimic(con))
    con.close()

    out = []
    for cohort, df in (("ARMD-MGB", armd), ("MIMIC-IV", mimic)):
        for seed in SEEDS:
            for r in one_repeat(df, seed):
                r.update(cohort=cohort, seed=seed)
                out.append(r)

    d = pd.DataFrame(out)
    d.to_csv(TABLES / "c07_repeated_cv.csv", index=False)
    agg = (d.groupby(["cohort", "metric", "model"])["value"]
             .agg(["mean", "std", "min", "max", "count"]).reset_index())
    print("\n=== 5 个种子 × 5 折：均值 ± SD ===")
    piv = agg.copy()
    piv["mean_sd"] = piv["mean"].round(3).astype(str) + " ± " + piv["std"].round(3).astype(str)
    for cohort in ["ARMD-MGB", "MIMIC-IV"]:
        print(f"\n-- {cohort} --")
        print(piv[piv.cohort == cohort][["metric", "model", "mean_sd", "min", "max"]].to_string(index=False))
    agg.to_csv(TABLES / "c07_repeated_cv_summary.csv", index=False)


if __name__ == "__main__":
    main()
