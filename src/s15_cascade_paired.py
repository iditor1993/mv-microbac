"""S15 Paired comparison of cascade strategies (M3 vs M1 vs rule)."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import run_models
from common import ARMD_DB, MIMIC_DB, TABLES, connect
import s14_cascade as S

def paired(d, fn_a, fn_b, B=1000, seed=20261024):
    rng = np.random.default_rng(seed)
    g = d.patient_id.to_numpy(); uniq, inv = np.unique(g, return_inverse=True)
    k = len(uniq); counts = np.bincount(inv, minlength=k)
    order = np.argsort(inv, kind="stable"); offs = np.concatenate([[0], np.cumsum(counts)])
    y_cef = d.y.to_numpy(int); y_mem = d.mem_r.to_numpy(int)
    d_ic, d_cu = [], []
    def metric(fn, idx):
        esc = fn(d, idx)
        return esc.mean(), np.where(esc, y_mem[idx], y_cef[idx]).mean()
    for _ in range(B):
        rep = rng.multinomial(k, np.full(k, 1.0/k))
        take = np.repeat(np.arange(k), rep)
        lens = counts[take]; starts = offs[take]
        sel = np.repeat(starts, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens)-lens, lens))
        idx = order[sel]
        cu_a, ic_a = metric(fn_a, idx); cu_b, ic_b = metric(fn_b, idx)
        d_cu.append(cu_a - cu_b); d_ic.append(ic_a - ic_b)
    cu_a, ic_a = metric(fn_a, np.arange(len(d))); cu_b, ic_b = metric(fn_b, np.arange(len(d)))
    return (cu_a - cu_b, np.percentile(d_cu, 2.5), np.percentile(d_cu, 97.5),
            ic_a - ic_b, np.percentile(d_ic, 2.5), np.percentile(d_ic, 97.5))

def main():
    con = connect(ARMD_DB); from features import build_armd, prepare
    armd = prepare(build_armd(con)); armd = S.add_mem_armd(con, armd); con.close()
    con = connect(MIMIC_DB); from features import build_mimic
    mimic = prepare(build_mimic(con)); mimic = S.add_mem_mimic(con, mimic); con.close()
    g = armd.patient_id.to_numpy(); pa = run_models(armd, groups=g)
    test = (mimic.year >= 2015).to_numpy(); pm = run_models(mimic, test_mask=test)

    rows = []
    for name, df, p1s, p3s, mask in [("ARMD-MGB", armd, "M1 +既往用药", "M3 +既往同菌药敏", np.ones(len(armd), bool)),
                                     ("MIMIC-IV", mimic, "M1 +既往用药", "M3 +既往同菌药敏", test)]:
        p1, p3 = (pa[p1s], pa[p3s]) if name == "ARMD-MGB" else (pm[p1s], pm[p3s])
        d, _ = S.run_cohort(name, df, p1, p3, mask)
        rule = lambda dd, ix: (dd.prior_same_res.to_numpy()[ix] > 0)
        for t in (0.20, 0.10, 0.05):
            f1 = (lambda tt: (lambda dd, ix: dd.p1.to_numpy()[ix] >= tt))(t)
            f3 = (lambda tt: (lambda dd, ix: dd.p3.to_numpy()[ix] >= tt))(t)
            cu, lo_cu, hi_cu, ic, lo_ic, hi_ic = paired(d, f3, f1)
            rows.append({"cohort": name, "comparison": "M3 vs M1", "coverage_target": int((1-t)*100),
                         "d_carbapenem": cu, "cu_lo": lo_cu, "cu_hi": hi_cu,
                         "d_inadequate": ic, "ic_lo": lo_ic, "ic_hi": hi_ic})
        cu, lo_cu, hi_cu, ic, lo_ic, hi_ic = paired(d, (lambda: (lambda dd, ix: dd.p3.to_numpy()[ix] >= 0.20))(), rule)
        rows.append({"cohort": name, "comparison": "M3(80%) vs rule", "coverage_target": 80,
                     "d_carbapenem": cu, "cu_lo": lo_cu, "cu_hi": hi_cu,
                     "d_inadequate": ic, "ic_lo": lo_ic, "ic_hi": hi_ic})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "c15_cascade_paired.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
