
"""S13 Full coefficients for the primary model (M3), for supplementary reporting."""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import MODELS
from common import ARMD_DB, MIMIC_DB, TABLES, connect
from features import build_armd, build_mimic, prepare

LABEL = {
 "age_mid":"Age (band midpoint)", "male":"Male sex", "org_pa":"P. aeruginosa present",
 "org_other":"Other non-enterobacterial Gram-negative organism",
 "days_since_abx":"Days since last antibiotic", "n_drugs_365d":"Distinct antibiotics, 365 days",
 "d_fq":"Prior fluoroquinolone", "d_bsc":"Prior third-generation cephalosporin", "d_carbapenem":"Prior carbapenem",
 "d_piptazo":"Prior piperacillin-tazobactam", "d_tmpsmx":"Prior trimethoprim-sulfamethoxazole",
 "d_nitro":"Prior nitrofurantoin", "d_amino":"Prior aminoglycoside", "d_glyco":"Prior glycopeptide",
 "d_cefepime":"Prior cefepime", "n_cx_90d":"Prior cultures, 90 days", "n_cx_365d":"Prior cultures, 365 days",
 "days_since_cx":"Days since last culture", "prior_same_org":"Prior isolation of same organism",
 "prior_same_ast":"Prior same-organism cefepime AST available", "prior_same_res":"Prior same-organism cefepime resistance",
}
rows = []
for name, fn, db, train_only in [("ARMD-MGB", build_armd, ARMD_DB, False), ("MIMIC-IV", build_mimic, MIMIC_DB, True)]:
    con = connect(db); df = prepare(fn(con)); con.close()
    if train_only:
        df = df[df.year <= 2014].reset_index(drop=True)
    X = df[MODELS["M3 +既往同菌药敏"]].to_numpy(float); y = df.y.to_numpy(int)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), y)
    for col, b in zip(MODELS["M3 +既往同菌药敏"], lr.coef_[0]):
        rows.append({"cohort": name, "variable": LABEL.get(col, col), "beta_std": b,
                     "odds_ratio_per_1SD": float(np.exp(b))})
d = pd.DataFrame(rows)
d.to_csv(os.path.join(TABLES, "c13_coefficients.csv"), index=False)
piv = d.pivot(index="variable", columns="cohort", values="odds_ratio_per_1SD")
order = [LABEL[c] for c in MODELS["M3 +既往同菌药敏"]]
piv = piv.loc[order]
L = ["**Table S4.** Coefficients of the primary model (M3) in each database\n",
     "| Variable | ARMD-MGB OR (per 1 SD) | MIMIC-IV OR (per 1 SD) |", "|---|---:|---:|"]
for v, r in piv.iterrows():
    L.append(f"| {v} | {r['ARMD-MGB']:.2f} | {r['MIMIC-IV']:.2f} |")
L += ["", "OR, odds ratio; SD, standard deviation. ARMD-MGB coefficients are from a model fitted to all cultures (performance was estimated separately by cross-validation); MIMIC-IV coefficients are from the model fitted to 2008-2014 data only, which was the model used for temporal validation. Predictors were standardised before fitting; the table therefore gives the change in odds per one standard deviation increase."]
out = os.path.join(TABLES, "c13_coefficients.md")
open(out, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
