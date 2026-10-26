"""S09 MIMIC 单库：加上严重度与合并症后，既往同菌药敏还有没有增量？

严重度变量（取培养所在 ICU 住院的）：
    sofa_first   first_day_sofa.sofa
    vaso_any     培养前后 ±12h 内任何升压药
    lactate_max  培养前 24h 内乳酸最大值（itemid 50813）
    charlson     charlson_comorbidity_index

比较：M1+SEV、M2b+SEV、M3+SEV（时间外推 2008–2014 → 2015–2022）。
产出：results/tables/c09_severity.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import MODELS
from common import MIMIC_DB, TABLES, bootstrap_delta_auroc, connect, ensure_dirs, metric_block
from features import build_mimic, prepare
from sens_common import cv_or_temporal

SEV = ["sofa_first", "vaso_any", "lactate_max", "charlson"]


def add_severity(con, df):
    con.execute("""
        CREATE OR REPLACE TEMP TABLE sev AS
        SELECT c.cid,
               max(f.sofa) AS sofa_first,
               max(CASE WHEN coalesce(v.norepinephrine,0) > 0 OR coalesce(v.epinephrine,0) > 0
                          OR coalesce(v.dopamine,0) > 0 OR coalesce(v.vasopressin,0) > 0
                        THEN 1 ELSE 0 END) AS vaso_any,
               max(l.valuenum) AS lactate_max,
               max(ch.charlson_comorbidity_index) AS charlson
        FROM cohort c
        LEFT JOIN icu.icustays s
               ON s.subject_id = c.subject_id AND c.d BETWEEN s.intime AND s.outtime
        LEFT JOIN derived.first_day_sofa f ON f.stay_id = s.stay_id
        LEFT JOIN derived.vasoactive_agent v
               ON v.stay_id = s.stay_id
              AND v.starttime <= c.d + INTERVAL 12 HOUR
              AND v.endtime   >= c.d - INTERVAL 12 HOUR
        LEFT JOIN hosp.labevents l
               ON l.subject_id = c.subject_id AND l.itemid = 50813
              AND l.charttime BETWEEN c.d - INTERVAL 24 HOUR AND c.d
        LEFT JOIN derived.charlson ch ON ch.hadm_id = c.hadm_id
        GROUP BY 1
    """)
    sev = con.execute("SELECT * FROM sev").df()
    df = df.merge(sev, on="cid", how="left")
    for c in SEV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["vaso_any"] = df["vaso_any"].fillna(0)
    for c in ("sofa_first", "lactate_max", "charlson"):
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        med = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
        df[c] = v.fillna(med)
    return df


def main():
    ensure_dirs()
    con = connect(MIMIC_DB)
    df = prepare(build_mimic(con))
    df = add_severity(con, df)
    con.close()

    print("missing:", df[SEV].isna().sum().to_dict())
    print(df[SEV].describe().round(2).to_string())

    models = {
        "M1+SEV": MODELS["M1 +既往用药"] + SEV,
        "M2b+SEV": MODELS["M2b +既往同菌"] + SEV,
        "M3+SEV": MODELS["M3 +既往同菌药敏"] + SEV,
    }
    test = (df["year"] >= 2015).to_numpy()
    preds = cv_or_temporal(df, df["patient_id"].to_numpy(), test_mask=test, models=models)
    y = df["y"].to_numpy(int)[test]
    g = df["patient_id"].to_numpy()[test]

    rows = []
    for name in models:
        m = metric_block(y, preds[name][test])
        rows.append({"cohort": "MIMIC-IV", "model": name, "n": len(y), "events": int(y.sum()),
                     "auroc": m["auroc"], "brier": m["brier"],
                     "calib_slope": m["calib_slope"], "calib_intercept": m["calib_intercept"]})
    for old, new, label in [("M2b+SEV", "M3+SEV", "M3−M2b | +严重度"),
                            ("M1+SEV", "M3+SEV", "M3−M1 | +严重度")]:
        d = bootstrap_delta_auroc(y, preds[old][test], preds[new][test], groups=g, B=1000)
        rows.append({"cohort": "MIMIC-IV", "model": label, "n": len(y), "events": int(y.sum()),
                     "delta_auroc": d["delta_auroc"], "delta_lo": d["delta_lo"],
                     "delta_hi": d["delta_hi"], "p_gt0": d["p_gt0"]})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "c09_severity.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
