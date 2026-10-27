"""S12 Recalibrated threshold performance for the temporally validated MIMIC-IV model.

Recalibration in the large is a monotone transform, so thresholds are re-expressed
on the raw probability scale rather than re-fitting the model.
"""

from __future__ import annotations
import json, sys, numpy as np, pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import MODELS, run_models
from common import MIMIC_DB, TABLES, connect
from features import build_mimic, prepare

def logit(p): p = np.clip(p, 1e-9, 1-1e-9); return np.log(p/(1-p))
def sig(x): return 1/(1+np.exp(-x))

con = connect(MIMIC_DB); df = prepare(build_mimic(con)); con.close()
test = (df["year"] >= 2015).to_numpy()
preds = run_models(df, test_mask=test)
y = df["y"].to_numpy(int)[test]

rows = []
for name in ["M1 +既往用药", "M3 +既往同菌药敏"]:
    p = preds[name][test]
    lp = logit(p)
    delta = float(brentq(lambda a: float(np.sum(y - sig(lp + a))), -30, 30))
    p_ic = sig(lp + delta)
    for t in (0.15, 0.20, 0.25):
        for stage, pp in (("raw", p), ("recalibrated", p_ic)):
            pred = pp >= t
            tp = int((pred & (y==1)).sum()); fp = int((pred & (y==0)).sum())
            fn = int((~pred & (y==1)).sum()); tn = int((~pred & (y==0)).sum())
            rows.append({"model": "M1" if name.startswith("M1") else "M3", "stage": stage,
                         "threshold": t, "flagged": pred.mean(),
                         "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp),
                         "ppv": tp/(tp+fp) if tp+fp else np.nan,
                         "npv": tn/(tn+fn) if tn+fn else np.nan,
                         "missed_per_1000": 1000*fn/len(y),
                         "flagged_per_resistant": (tp+fp)/tp if tp else np.nan,
                         "mean_pred": pp.mean(), "obs": y.mean(), "delta_ic": delta})
out = pd.DataFrame(rows)
out.to_csv(TABLES / "c12_recalibrated_thresholds.csv", index=False)
print(out.round(3).to_string(index=False))
