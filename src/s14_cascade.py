"""S14 Cascade simulation: does model-guided escalation change decisions?

Menu: cefepime (anti-pseudomonal cephalosporin) -> meropenem (carbapenem).
Strategies compared on specimens with both cefepime and meropenem AST:
  1. cefepime for all
  2. meropenem for all
  3. personal-antibiogram rule: escalate if prior same-organism cefepime-resistant
  4. M1-guided: escalate if predicted P(cefepime-R) >= t
  5. M3-guided: same, with the prior-susceptibility model
Coverage targets C = 80/90/95% correspond to escalation thresholds t = 0.20/0.10/0.05.

Outputs: c14_cascade.csv, c14_cascade_boot.csv, fig5_cascade.png
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import MODELS, run_models
from common import ARMD_DB, FIGURES, MIMIC_DB, TABLES, connect

GN = ("PSEUDOMONAS|KLEBSIELLA|ESCHERICHIA|ENTEROBACTER|SERRATIA|PROTEUS"
      "|CITROBACTER|ACINETOBACTER|STENOTROPHOMONAS")

def add_mem_armd(con, df):
    ids = df.cid.astype(int).tolist()
    con.execute("CREATE OR REPLACE TEMP TABLE ids(cid BIGINT)")
    con.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in ids])
    mem = con.execute(f"""
      SELECT order_proc_id_coded AS cid,
             max(CASE WHEN antibiotic='meropenem' AND CLSI_2022_pheno IN ('Resistant','Susceptible')
                      AND regexp_matches(organism,'{GN}') THEN 1 ELSE 0 END) AS has_mem,
             max(CASE WHEN antibiotic='meropenem' AND CLSI_2022_pheno='Resistant'
                      AND regexp_matches(organism,'{GN}') THEN 1 ELSE 0 END) AS mem_r
      FROM main.microbiology_cohort WHERE order_proc_id_coded IN (SELECT cid FROM ids)
      GROUP BY 1""").df()
    return df.merge(mem, on="cid", how="left").fillna({"has_mem": 0, "mem_r": 0})

def add_mem_mimic(con, df):
    ids = df.cid.astype(int).tolist()
    con.execute("CREATE OR REPLACE TEMP TABLE ids(cid BIGINT)")
    con.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in ids])
    mem = con.execute(f"""
      SELECT micro_specimen_id AS cid,
             max(CASE WHEN ab_name='MEROPENEM' AND interpretation IN ('S','R')
                      AND regexp_matches(org_name,'{GN}') THEN 1 ELSE 0 END) AS has_mem,
             max(CASE WHEN ab_name='MEROPENEM' AND interpretation='R'
                      AND regexp_matches(org_name,'{GN}') THEN 1 ELSE 0 END) AS mem_r
      FROM hosp.microbiologyevents WHERE micro_specimen_id IN (SELECT cid FROM ids)
      GROUP BY 1""").df()
    return df.merge(mem, on="cid", how="left").fillna({"has_mem": 0, "mem_r": 0})

def run_cohort(name, df, p1, p3, eval_mask):
    d = df[eval_mask].copy()
    d["p1"], d["p3"] = p1[eval_mask], p3[eval_mask]
    d = d[d.has_mem > 0].reset_index(drop=True)
    y_cef = d.y.to_numpy(int)
    y_mem = d.mem_r.to_numpy(int)
    out = []
    def add(strategy, escalate, note=""):
        esc = np.asarray(escalate).astype(bool)
        inadequate = np.where(esc, y_mem, y_cef)
        out.append({"cohort": name, "strategy": strategy, "n": len(d),
                    "carbapenem_use": float(esc.mean()),
                    "inadequate_coverage": float(inadequate.mean()),
                    "cefepime_resistant": float(y_cef.mean()),
                    "meropenem_resistant": float(y_mem.mean()), "note": note})
    add("Cefepime for all", np.zeros(len(d), bool))
    add("Meropenem for all", np.ones(len(d), bool))
    add("Rule: prior resistant -> meropenem", (d.prior_same_res > 0).to_numpy())
    for t in (0.20, 0.10, 0.05):
        add(f"M1-guided (coverage {int((1-t)*100)}%)", (d.p1 >= t).to_numpy())
    for t in (0.20, 0.10, 0.05):
        add(f"M3-guided (coverage {int((1-t)*100)}%)", (d.p3 >= t).to_numpy())
    return d, pd.DataFrame(out)

def boot(d, strategy_fn, B=1000, seed=20261024):
    rng = np.random.default_rng(seed)
    g = d.patient_id.to_numpy(); uniq, inv = np.unique(g, return_inverse=True)
    k = len(uniq); counts = np.bincount(inv, minlength=k)
    order = np.argsort(inv, kind="stable"); offs = np.concatenate([[0], np.cumsum(counts)])
    y_cef = d.y.to_numpy(int); y_mem = d.mem_r.to_numpy(int)
    cu, ic = [], []
    for _ in range(B):
        rep = rng.multinomial(k, np.full(k, 1.0/k))
        take = np.repeat(np.arange(k), rep)
        lens = counts[take]; starts = offs[take]
        sel = np.repeat(starts, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens)-lens, lens))
        idx = order[sel]
        esc = strategy_fn(d, idx)
        cu.append(esc.mean()); ic.append(np.where(esc, y_mem[idx], y_cef[idx]).mean())
    return float(np.percentile(cu, 2.5)), float(np.percentile(cu, 97.5)), float(np.percentile(ic, 2.5)), float(np.percentile(ic, 97.5))

def main():
    con = connect(ARMD_DB)
    from features import build_armd, prepare
    armd = prepare(build_armd(con)); armd = add_mem_armd(con, armd); con.close()
    con = connect(MIMIC_DB)
    from features import build_mimic
    mimic = prepare(build_mimic(con)); mimic = add_mem_mimic(con, mimic); con.close()

    g = armd.patient_id.to_numpy()
    pa = run_models(armd, groups=g)
    test = (mimic.year >= 2015).to_numpy()
    pm = run_models(mimic, test_mask=test)

    rows, boots = [], []
    for name, df, p1, p3, mask in [("ARMD-MGB", armd, pa["M1 +既往用药"], pa["M3 +既往同菌药敏"], np.ones(len(armd), bool)),
                                   ("MIMIC-IV", mimic, pm["M1 +既往用药"], pm["M3 +既往同菌药敏"], test)]:
        d, tab = run_cohort(name, df, p1, p3, mask)
        rows.append(tab)
        fns = {"Cefepime for all": lambda dd, ix: np.zeros(len(ix), bool),
               "Meropenem for all": lambda dd, ix: np.ones(len(ix), bool),
               "Rule: prior resistant -> meropenem": lambda dd, ix: (dd.prior_same_res.to_numpy()[ix] > 0)}
        for t in (0.20, 0.10, 0.05):
            fns[f"M1-guided (coverage {int((1-t)*100)}%)"] = (lambda tt: (lambda dd, ix: dd.p1.to_numpy()[ix] >= tt))(t)
            fns[f"M3-guided (coverage {int((1-t)*100)}%)"] = (lambda tt: (lambda dd, ix: dd.p3.to_numpy()[ix] >= tt))(t)
        for strat, fn in fns.items():
            lo_cu, hi_cu, lo_ic, hi_ic = boot(d, fn)
            boots.append({"cohort": name, "strategy": strat, "cu_lo": lo_cu, "cu_hi": hi_cu,
                          "ic_lo": lo_ic, "ic_hi": hi_ic})
    res = pd.concat(rows, ignore_index=True).merge(pd.DataFrame(boots), on=["cohort", "strategy"], how="left")
    res.to_csv(os.path.join(TABLES, "c14_cascade.csv"), index=False)
    pd.set_option("display.width", 200)
    for coh in ["ARMD-MGB", "MIMIC-IV"]:
        print("\n===", coh, "===")
        r = res[res.cohort == coh]
        print(r[["strategy", "n", "carbapenem_use", "cu_lo", "cu_hi", "inadequate_coverage", "ic_lo", "ic_hi"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
