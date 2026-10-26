"""S08 排除分析：近期用过结局药 / 多菌培养。

变体：all（参照）、exclude_cefepime_7d、exclude_polymicrobial。
两库均用 5 折患者分组 CV（保证各变体之间可比）。
产出：results/tables/c08_exclusions.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import ARMD_DB, MIMIC_DB, TABLES, connect, ensure_dirs
from features import build_armd, build_mimic, prepare
from sens_common import cv_or_temporal, increments


def main():
    ensure_dirs()
    con = connect(ARMD_DB)
    armd = prepare(build_armd(con))
    con.close()
    con = connect(MIMIC_DB)
    mimic = prepare(build_mimic(con))
    con.close()

    rows = []
    for cohort, df in (("ARMD-MGB", armd), ("MIMIC-IV", mimic)):
        variants = {
            "all": np.ones(len(df), bool),
            "exclude_cefepime_7d": (df["days_since_cefepime"] > 7).to_numpy(),
            "exclude_polymicrobial": (df["n_org"] <= 1).to_numpy(),
        }
        for name, mask in variants.items():
            sub = df[mask].reset_index(drop=True)
            g = sub["patient_id"].to_numpy()
            preds = cv_or_temporal(sub, g)
            rows += increments(sub, preds, np.ones(len(sub), bool), g, cohort, name)
            print(f"{cohort} {name} n={len(sub)} events={int(sub.y.sum())}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "c08_exclusions.csv", index=False)
    key = d[d.model.isin(["M3−M2b 同菌药敏（主）", "M1−M0 用药史", "M5−M4 他菌耐药"])]
    print(key[["cohort", "variant", "model", "n", "events", "auroc", "auroc_lo", "auroc_hi"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
