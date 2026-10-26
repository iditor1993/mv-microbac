"""S04 通气定义的敏感性：ARMD 的开机日窗口 2/7/14/30 天；MIMIC 的重叠容差 ±12/±24h。

产出：results/tables/c04_mv_window.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import ARMD_DB, MIMIC_DB, TABLES, connect, ensure_dirs
from features import build_armd, build_mimic, prepare
from sens_common import cv_or_temporal, increments


def main():
    ensure_dirs()
    rows = []
    con = connect(ARMD_DB)
    for mv in (2, 7, 14, 30):
        df = prepare(build_armd(con, mv_days=mv))
        g = df["patient_id"].to_numpy()
        preds = cv_or_temporal(df, g)
        rows += increments(df, preds, np.ones(len(df), bool), g, "ARMD-MGB", f"MV≤{mv}d")
        print(f"ARMD MV<={mv}d n={len(df)}", flush=True)
    con.close()

    con = connect(MIMIC_DB)
    for tol in (12, 24):
        df = prepare(build_mimic(con, tol_hours=tol))
        g = df["patient_id"].to_numpy()
        test = (df["year"] >= 2015).to_numpy()
        preds = cv_or_temporal(df, g, test_mask=test)
        rows += increments(df, preds, test, g, "MIMIC-IV", f"IMV±{tol}h")
        print(f"MIMIC ±{tol}h n={len(df)}", flush=True)
    con.close()

    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "c04_mv_window.csv", index=False)
    key = d[d.model.isin(["M3−M2b 同菌药敏（主）", "M1−M0 用药史", "M2−M1 培养数量"])]
    print(key[["cohort", "variant", "model", "n", "events", "auroc", "auroc_lo", "auroc_hi"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
