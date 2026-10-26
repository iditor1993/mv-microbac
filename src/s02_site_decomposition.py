"""S02 分层：既往同菌药敏来自「呼吸道」还是「非呼吸道」，能否互相迁移？

问题：临床最常见的一句直觉是"尿里长过耐药的铜绿，痰里大概也耐药"。
本脚本把既往同菌耐药拆成两个不重叠的指示变量，比较三种设定：

    M2b          只知长过同菌
    M3-resp      只用**呼吸道**既往耐药
    M3-nonresp   只用**非呼吸道**既往耐药
    M3-split     两者同时进模型
    M3-all       合并的 prior_same_res（主分析口径）

产出：results/tables/c02_site_*.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import fit_predict
from common import ARMD_DB, MIMIC_DB, TABLES, bootstrap_delta_auroc, connect, ensure_dirs
from features import ABX, CXBURDEN, DEMO, ORG, build_armd, build_mimic, prepare
from sklearn.model_selection import StratifiedGroupKFold

BASE = DEMO + ORG + ABX + CXBURDEN + ["prior_same_org"]
MODELS = {
    "M2b 只知长过同菌": BASE,
    "M3-resp 仅呼吸道史": BASE + ["prior_same_ast", "prior_same_res_resp"],
    "M3-nonresp 仅非呼吸道史": BASE + ["prior_same_ast", "prior_same_res_nonresp"],
    "M3-split 部位拆分": BASE + ["prior_same_ast", "prior_same_res_resp", "prior_same_res_nonresp"],
    "M3-all 合并（主分析）": BASE + ["prior_same_ast", "prior_same_res"],
}


def cv_predict(df, cols, groups, n_splits=5):
    X = df[cols].to_numpy(float)
    y = df["y"].to_numpy(int)
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=20261024).split(X, y, groups):
        p[te] = fit_predict(X[tr], y[tr], X[te])
    return p


def run(name, df):
    g = df["patient_id"].to_numpy()
    preds = {k: cv_predict(df, v, g) for k, v in MODELS.items()}
    from common import bootstrap_auroc, metric_block
    rows, incs = [], []
    y = df["y"].to_numpy(int)
    for k, p in preds.items():
        m = metric_block(y, p)
        ci = bootstrap_auroc(y, p, g, B=500)
        m.update(cohort=name, model=k, auroc_lo=ci["auroc_lo"], auroc_hi=ci["auroc_hi"])
        rows.append(m)
    for k in MODELS:
        if k == "M2b 只知长过同菌":
            continue
        d = bootstrap_delta_auroc(y, preds["M2b 只知长过同菌"], preds[k], groups=g, B=500)
        incs.append({"cohort": name, "model": k, **d})
    # 分层：按既往耐药来源
    strata = []
    for label, mask in [
        ("仅既往呼吸道耐药", (df.prior_same_res_resp > 0) & (df.prior_same_res_nonresp == 0)),
        ("仅既往非呼吸道耐药", (df.prior_same_res_nonresp > 0) & (df.prior_same_res_resp == 0)),
        ("两处均耐药", (df.prior_same_res_resp > 0) & (df.prior_same_res_nonresp > 0)),
        ("有同菌药敏但均敏感", (df.prior_same_ast > 0) & (df.prior_same_res == 0)),
    ]:
        mask = mask.to_numpy()
        if mask.sum() < 20:
            continue
        strata.append({"cohort": name, "stratum": label, "n": int(mask.sum()),
                       "events": int(y[mask].sum()), "resistance_rate": float(y[mask].mean())})
    return rows, incs, strata


def main():
    ensure_dirs()
    con = connect(ARMD_DB)
    armd = prepare(build_armd(con))
    con.close()
    con = connect(MIMIC_DB)
    mimic = prepare(build_mimic(con))
    con.close()

    perf, incs = [], []
    for name, df in [("ARMD-MGB", armd), ("MIMIC-IV", mimic)]:
        p, i, s = run(name, df)
        perf += p
        incs += i
        print(f"\n=== {name} ===")
        print(pd.DataFrame(s).round(3).to_string(index=False))
        print(pd.DataFrame(p)[["model", "auroc", "auroc_lo", "auroc_hi"]].round(3).to_string(index=False))
        print(pd.DataFrame(i)[["model", "delta_auroc", "delta_lo", "delta_hi"]].round(3).to_string(index=False))
        pd.DataFrame(s).to_csv(TABLES / f"c02_site_strata_{name.split('-')[0].lower()}.csv", index=False)

    pd.DataFrame(perf).to_csv(TABLES / "c02_site_performance.csv", index=False)
    pd.DataFrame(incs).to_csv(TABLES / "c02_site_increments.csv", index=False)
    print("\n表已写入", TABLES)


if __name__ == "__main__":
    main()
