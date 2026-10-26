"""S06 非线性学习器复核：所有零结果是不是逻辑回归的产物？

用 XGBoost 重跑 M0–M5（同特征、同切分），看增量是否仍然成立。
产出：results/tables/c06_xgb_performance.csv, c06_xgb_increments.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import INCREMENTS, MODELS, run_models
from common import (ARMD_DB, MIMIC_DB, TABLES, bootstrap_auroc, bootstrap_delta_auroc,
                    connect, ensure_dirs, metric_block)
from features import build_armd, build_mimic, prepare


def summarize(cohort, df, eval_mask, preds, groups):
    y = df["y"].to_numpy(int)[eval_mask]
    g = groups[eval_mask]
    perf, inc = [], []
    for name in MODELS:
        m = metric_block(y, preds[name][eval_mask])
        ci = bootstrap_auroc(y, preds[name][eval_mask], g, B=500)
        m.update(cohort=cohort, model=name, auroc_lo=ci["auroc_lo"], auroc_hi=ci["auroc_hi"])
        perf.append(m)
    for old, new, label in INCREMENTS:
        d = bootstrap_delta_auroc(y, preds[old][eval_mask], preds[new][eval_mask], groups=g, B=500)
        inc.append({"cohort": cohort, "increment": label, **d})
    return perf, inc


def main():
    ensure_dirs()
    con = connect(ARMD_DB)
    armd = prepare(build_armd(con))
    con.close()
    con = connect(MIMIC_DB)
    mimic = prepare(build_mimic(con))
    con.close()

    perf, inc = [], []

    print("[1/2] ARMD：GroupKFold + XGBoost …", flush=True)
    g = armd["patient_id"].to_numpy()
    pa = run_models(armd, groups=g, learner="xgb")
    p, i = summarize("ARMD-MGB", armd, np.ones(len(armd), bool), pa, g)
    perf += p
    inc += i

    print("[2/2] MIMIC：时间外推 + XGBoost …", flush=True)
    test = (mimic["year"] >= 2015).to_numpy()
    gm = mimic["patient_id"].to_numpy()
    pm = run_models(mimic, test_mask=test, learner="xgb")
    p, i = summarize("MIMIC-IV", mimic, test, pm, gm)
    perf += p
    inc += i

    pd.DataFrame(perf).to_csv(TABLES / "c06_xgb_performance.csv", index=False)
    pd.DataFrame(inc).to_csv(TABLES / "c06_xgb_increments.csv", index=False)
    print("\n=== XGBoost 性能 ===")
    print(pd.DataFrame(perf)[["cohort", "model", "auroc", "auroc_lo", "auroc_hi"]].round(3).to_string(index=False))
    print("\n=== XGBoost 增量 ===")
    print(pd.DataFrame(inc)[["cohort", "increment", "delta_auroc", "delta_lo", "delta_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
