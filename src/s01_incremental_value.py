"""S01 主分析：MV 人群下，既往微生物史相对既往用药史的增量价值。

用法
----
    python s01_incremental_value.py                    # 主分析：头孢吡肟
    python s01_incremental_value.py --drug meropenem --tag s03_meropenem

产出（全是聚合表，不含个体级数据）
----------------------------------
    results/tables/{tag}_cohort.csv            队列与既往史密度
    results/tables/{tag}_model_performance.csv 各模型 AUROC/AUPRC/Brier/校准
    results/tables/{tag}_increments.csv        ΔAUROC / NRI / IDI（按患者聚类）
    results/tables/{tag}_dca.csv               决策曲线
    results/tables/{tag}_stratified.csv        分层
    results/tables/{tag}_thresholds.csv        阈值性能
    results/tables/{tag}_calibration.csv       M3 十分位校准
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from analysis import DCA_NEW, DCA_OLD, analyse_cohort, merge, run_models
from common import ARMD_DB, MIMIC_DB, TABLES, connect, ensure_dirs
from features import (
    ABX, CXBURDEN, DEMO, HIST_OTHER, HIST_SAME, ORG,
    build_armd, build_mimic, prepare,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def cohort_summary(df: pd.DataFrame, cohort: str) -> dict:
    y = df["y"].to_numpy(int)
    return {
        "cohort": cohort,
        "n_cultures": len(df),
        "n_patients": int(df["patient_id"].nunique()),
        "cultures_per_patient": round(len(df) / df["patient_id"].nunique(), 2),
        "events": int(y.sum()),
        "event_rate": round(float(y.mean()), 4),
        "pct_prior_cx": round(float((df["n_cx_365d"] > 0).mean()), 4),
        "pct_prior_same_org": round(float(df["prior_same_org"].mean()), 4),
        "pct_prior_same_ast": round(float(df["prior_same_ast"].mean()), 4),
        "pct_prior_same_res": round(float(df["prior_same_res"].mean()), 4),
        "pct_prior_other_gn": round(float(df["prior_other_gn_org"].mean()), 4),
        "pct_prior_other_gn_res": round(float(df["prior_other_gn_res"].mean()), 4),
        "pct_prior_abx_365": round(float((df["days_since_abx"] <= 365).mean()), 4),
        "pct_prior_cefepime": round(float(df["d_cefepime"].mean()), 4),
        "pct_org_pa": round(float(df["org_pa"].mean()), 4),
        "pct_org_other_gn": round(float(df["org_other"].mean()), 4),
        "res_rate_if_prior_res": round(float(y[df["prior_same_res"] > 0].mean()), 4)
        if (df["prior_same_res"] > 0).any() else np.nan,
        "res_rate_if_no_prior_res": round(float(y[df["prior_same_res"] == 0].mean()), 4),
        "res_rate_if_prior_S": round(
            float(y[(df["prior_same_ast"] > 0) & (df["prior_same_res"] == 0)].mean()), 4)
        if ((df["prior_same_ast"] > 0) & (df["prior_same_res"] == 0)).any() else np.nan,
        "n_prior_res_cultures": int((df["prior_same_res"] > 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", default="cefepime", help="cefepime / meropenem / piperacillin_tazobactam")
    ap.add_argument("--tag", default="c01")
    ap.add_argument("--no-xgb", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    t0 = time.time()
    armd_drug = args.drug.lower()
    mimic_drug = args.drug.upper()

    print(f"[1/4] 构建 ARMD 队列（drug={armd_drug}）…", flush=True)
    con_a = connect(ARMD_DB)
    armd = prepare(build_armd(con_a, drug=armd_drug))
    con_a.close()
    print(f"      ARMD: {len(armd)} 培养 / {armd['patient_id'].nunique()} 人 / {int(armd['y'].sum())} 事件", flush=True)

    print(f"[2/4] 构建 MIMIC 队列（drug={mimic_drug}）…", flush=True)
    con_m = connect(MIMIC_DB)
    mimic = prepare(build_mimic(con_m, drug=mimic_drug))
    con_m.close()
    print(f"      MIMIC: {len(mimic)} 标本 / {mimic['patient_id'].nunique()} 人 / {int(mimic['y'].sum())} 事件", flush=True)

    results = []
    summaries = [cohort_summary(armd, "ARMD-MGB"), cohort_summary(mimic, "MIMIC-IV")]

    print("[3/4] ARMD：5 折患者分组 CV + 聚类 bootstrap …", flush=True)
    g_armd = armd["patient_id"].to_numpy()
    preds_a = run_models(armd, groups=g_armd)
    results.append(analyse_cohort("ARMD-MGB", armd, np.ones(len(armd), bool), preds_a, g_armd))

    print("[4/4] MIMIC：时间外推 2008–2014 → 2015–2022 …", flush=True)
    test_mask = (mimic["year"] >= 2015).to_numpy()
    g_mimic = mimic["patient_id"].to_numpy()
    preds_m = run_models(mimic, test_mask=test_mask)
    results.append(analyse_cohort("MIMIC-IV", mimic, test_mask, preds_m, g_mimic))

    out = merge(results)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(TABLES / f"{args.tag}_cohort.csv", index=False)
    out["perf"].to_csv(TABLES / f"{args.tag}_model_performance.csv", index=False)
    out["inc"].to_csv(TABLES / f"{args.tag}_increments.csv", index=False)
    out["dca"].to_csv(TABLES / f"{args.tag}_dca.csv", index=False)
    out["strat"].to_csv(TABLES / f"{args.tag}_stratified.csv", index=False)
    out["thresh"].to_csv(TABLES / f"{args.tag}_thresholds.csv", index=False)
    out["calib"].to_csv(TABLES / f"{args.tag}_calibration.csv", index=False)

    print("\n=== 队列 ===")
    print(pd.DataFrame(summaries).T.to_string())
    print("\n=== 模型性能（评估集）===")
    cols = ["cohort", "model", "n", "events", "auroc", "auroc_lo", "auroc_hi", "auprc", "brier"]
    print(out["perf"][cols].round(3).to_string(index=False))
    print("\n=== 增量（按患者聚类 bootstrap）===")
    cols = ["cohort", "increment", "delta_auroc", "delta_lo", "delta_hi", "p_gt0", "nri", "idi"]
    print(out["inc"][cols].round(3).to_string(index=False))
    print("\n=== 分层（M1 vs M3）===")
    if len(out["strat"]):
        cols = ["cohort", "stratum", "model", "n", "events", "auroc", "auroc_lo", "auroc_hi", "share"]
        print(out["strat"][cols].round(3).to_string(index=False))
    print(f"\n完成，用时 {time.time() - t0:.0f}s；表已写入 {TABLES}")


if __name__ == "__main__":
    main()
