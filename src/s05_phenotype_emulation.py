"""S05 表型模拟：ARMD 的"开机日窗口"规则如果在 MIMIC 上用，会错多少？

ARMD 没有拔管时间，只能用"mechvent 开机日距培养 ≤7 天"当代理。
本脚本把这条规则原样套到 MIMIC（有精确通气区间），做两件事：

1. 以"有创通气区间与培养重叠（±12h）"为金标准，算这条代理的敏感度/特异度/PPV/NPV；
2. 在"代理队列"和"金标准队列"上各跑一遍 M0–M5，比较主增量是否被错分扭曲。

产出：results/tables/c05_phenotype.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import MIMIC_DB, TABLES, connect, ensure_dirs
from features import build_mimic, prepare
from sens_common import cv_or_temporal, increments


def main():
    ensure_dirs()
    con = connect(MIMIC_DB)
    df = prepare(build_mimic(con, imv_filter=False))
    con.close()

    df["imv"] = pd.to_numeric(df["imv"], errors="coerce").fillna(0).astype(int)
    df["mv_start_days"] = pd.to_numeric(df["mv_start_days"], errors="coerce")
    proxy = (df["mv_start_days"] <= 7).fillna(False).to_numpy()
    truth = df["imv"].to_numpy().astype(bool)

    tp = int((proxy & truth).sum()); fp = int((proxy & ~truth).sum())
    fn = int((~proxy & truth).sum()); tn = int((~proxy & ~truth).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan

    rows = [{"cohort": "MIMIC-IV", "variant": "proxy_vs_truth", "model": "2x2",
             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
             "proxy_n": int(proxy.sum()), "truth_n": int(truth.sum())}]

    for name, mask in (("proxy (MV start ≤7d)", proxy), ("truth (IMV overlap ±12h)", truth)):
        sub = df[mask].reset_index(drop=True)
        g = sub["patient_id"].to_numpy()
        preds = cv_or_temporal(sub, g)
        inc = increments(sub, preds, np.ones(len(sub), bool), g, "MIMIC-IV", name)
        rows += inc
        print(f"{name}: n={len(sub)} events={int(sub.y.sum())}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "c05_phenotype.csv", index=False)
    print(f"\n2x2: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"sens={sens:.3f} spec={spec:.3f} ppv={ppv:.3f} npv={npv:.3f}")
    key = out[out.model.isin(["M3−M2b 同菌药敏（主）", "M1−M0 用药史"]) & (out.variant != "proxy_vs_truth")]
    print(key[["variant", "model", "n", "events", "auroc", "auroc_lo", "auroc_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
