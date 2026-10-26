"""复用分析逻辑：嵌套模型、增量检验、决策曲线、分层、阈值与校准。

与 C_prior_history_value/analysis.py 的差别
-------------------------------------------
* 模型集扩到 M0–M5（新增他菌菌种 / 他菌耐药两层）。
* 分层变量由“有无既往培养”改为“**有无既往同菌药敏**”（本队列 40–53% 有）。
* 所有区间用**按患者聚类** bootstrap。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import (
    bootstrap_auroc,
    bootstrap_delta_auroc,
    bootstrap_nri_idi,
    calibration_bins,
    idi,
    metric_block,
    net_benefit,
    net_benefit_treat_all,
    nri_continuous,
)
from features import ABX, CXBURDEN, DEMO, HIST_OTHER, HIST_SAME, ORG

MODELS = {
    "M0 人口学+菌种": DEMO + ORG,
    "M1 +既往用药": DEMO + ORG + ABX,
    "M2 +既往培养数量": DEMO + ORG + ABX + CXBURDEN,
    "M2b +既往同菌": DEMO + ORG + ABX + CXBURDEN + ["prior_same_org"],
    "M3 +既往同菌药敏": DEMO + ORG + ABX + CXBURDEN + HIST_SAME,
    "M4 +他菌菌种": DEMO + ORG + ABX + CXBURDEN + HIST_SAME + ["prior_other_gn_org"],
    "M5 +他菌耐药": DEMO + ORG + ABX + CXBURDEN + HIST_SAME + HIST_OTHER,
}
INCREMENTS = [
    ("M0 人口学+菌种", "M1 +既往用药", "M1−M0 用药史"),
    ("M1 +既往用药", "M2 +既往培养数量", "M2−M1 培养数量"),
    ("M2 +既往培养数量", "M2b +既往同菌", "M2b−M2 同菌菌种"),
    ("M2b +既往同菌", "M3 +既往同菌药敏", "M3−M2b 同菌药敏（主）"),
    ("M3 +既往同菌药敏", "M4 +他菌菌种", "M4−M3 他菌菌种"),
    ("M4 +他菌菌种", "M5 +他菌耐药", "M5−M4 他菌耐药"),
]
DCA_OLD = "M1 +既往用药"
DCA_NEW = "M3 +既往同菌药敏"
PRIMARY = ("M2b +既往同菌", "M3 +既往同菌药敏", "M3−M2b 同菌药敏（主）")
THRESHOLDS = (0.15, 0.20, 0.25)


def fit_predict(Xtr, ytr, Xte, learner: str = "lr") -> np.ndarray:
    if learner == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="auc", n_jobs=8, random_state=20261024,
        )
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)[:, 1]

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1]


def run_models(df: pd.DataFrame, test_mask: np.ndarray | None = None,
               groups: np.ndarray | None = None, n_splits: int = 5,
               models: dict[str, list[str]] | None = None,
               learner: str = "lr") -> dict[str, np.ndarray]:
    models = models or MODELS
    y = df["y"].to_numpy(int)
    out = {}
    for name, cols in models.items():
        X = df[cols].to_numpy(float)
        if test_mask is None:
            p = np.zeros(len(y))
            if groups is None:
                folds = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                        random_state=20261024).split(X, y)
            else:
                folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                             random_state=20261024).split(X, y, groups)
            for tr, te in folds:
                p[te] = fit_predict(X[tr], y[tr], X[te], learner=learner)
        else:
            p = np.full(len(y), np.nan)
            p[test_mask] = fit_predict(X[~test_mask], y[~test_mask], X[test_mask], learner=learner)
        out[name] = p
    return out


def _strat_metrics(name, mask, y, preds, groups_sub, total):
    ys = y[mask]
    if mask.sum() < 100 or len(np.unique(ys)) < 2:
        return None
    m = metric_block(ys, preds[name][mask])
    ci = bootstrap_auroc(ys, preds[name][mask], groups_sub, B=500)
    m.update(auroc_lo=ci["auroc_lo"], auroc_hi=ci["auroc_hi"], share=float(mask.sum() / total))
    return m


def analyse_cohort(cohort: str, df: pd.DataFrame, eval_mask: np.ndarray,
                   preds: dict[str, np.ndarray], groups: np.ndarray) -> dict:
    y_all = df["y"].to_numpy(int)
    y = y_all[eval_mask]
    g = np.asarray(groups)[eval_mask]
    perf, inc, dca, strat, thresh, calib = [], [], [], [], [], []

    for name in MODELS:
        m = metric_block(y, preds[name][eval_mask])
        ci = bootstrap_auroc(y, preds[name][eval_mask], g, B=1000)
        m.update(cohort=cohort, model=name, auroc_lo=ci["auroc_lo"], auroc_hi=ci["auroc_hi"])
        perf.append(m)

    for old, new, label in INCREMENTS:
        po, pn = preds[old][eval_mask], preds[new][eval_mask]
        d = bootstrap_delta_auroc(y, po, pn, groups=g, B=1000)
        nb = bootstrap_nri_idi(y, po, pn, groups=g, B=500)
        nri = nri_continuous(y, po, pn)
        inc.append({
            "cohort": cohort, "increment": label,
            "delta_auroc": d["delta_auroc"], "delta_lo": d["delta_lo"],
            "delta_hi": d["delta_hi"], "p_gt0": d["p_gt0"],
            "nri": nri["nri"], "nri_lo": nb["nri_lo"], "nri_hi": nb["nri_hi"],
            "idi": idi(y, po, pn), "idi_lo": nb["idi_lo"], "idi_hi": nb["idi_hi"],
        })

    p1, p3 = preds[DCA_OLD][eval_mask], preds[DCA_NEW][eval_mask]
    for t in np.arange(0.01, 0.99, 0.01):
        dca.append({
            "cohort": cohort, "threshold": round(float(t), 2),
            "nb_old": net_benefit(y, p1, t), "nb_new": net_benefit(y, p3, t),
            "nb_all": net_benefit_treat_all(y, t), "prevalence": float(y.mean()),
        })

    # 主分层：有无既往同菌药敏
    has_ast = df["prior_same_ast"].to_numpy() > 0
    for label, smask in [("有既往同菌药敏", eval_mask & has_ast),
                         ("无既往同菌药敏", eval_mask & ~has_ast)]:
        for name in (DCA_OLD, DCA_NEW):
            r = _strat_metrics(name, smask, y_all, preds, np.asarray(groups)[smask], eval_mask.sum())
            if r:
                r.update(cohort=cohort, stratum=label, model=name)
                strat.append(r)

    # 次要分层：菌种 / 既往结果方向
    pa = df["org_pa"].to_numpy() > 0
    for label, smask in [("含铜绿", eval_mask & pa), ("不含铜绿", eval_mask & ~pa)]:
        for name in (DCA_OLD, DCA_NEW):
            r = _strat_metrics(name, smask, y_all, preds, np.asarray(groups)[smask], eval_mask.sum())
            if r:
                r.update(cohort=cohort, stratum=label, model=name)
                strat.append(r)

    prior_res = df["prior_same_res"].to_numpy() > 0
    for label, smask in [("既往同菌耐药", eval_mask & prior_res),
                         ("既往同菌药敏全S", eval_mask & has_ast & ~prior_res),
                         ("无既往同菌药敏", eval_mask & ~has_ast)]:
        for name in (DCA_NEW,):
            r = _strat_metrics(name, smask, y_all, preds, np.asarray(groups)[smask], eval_mask.sum())
            if r:
                r.update(cohort=cohort, stratum=label, model=name)
                strat.append(r)

    # 阈值性能
    for name in (DCA_OLD, DCA_NEW):
        p = preds[name][eval_mask]
        for t in THRESHOLDS:
            pred = p >= t
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            tn = int((~pred & (y == 0)).sum())
            thresh.append({
                "cohort": cohort, "model": name, "threshold": t,
                "flagged": float(pred.mean()), "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "ppv": tp / (tp + fp) if tp + fp else np.nan,
                "npv": tn / (tn + fn) if tn + fn else np.nan,
                "missed_per_1000": 1000 * fn / len(y),
                "flagged_per_true_resistant": (tp + fp) / tp if tp else np.nan,
            })

    for name in (DCA_NEW,):
        for b in calibration_bins(y, preds[name][eval_mask], n_bins=10):
            b.update(cohort=cohort, model=name)
            calib.append(b)

    return {"perf": perf, "inc": inc, "dca": dca, "strat": strat,
            "thresh": thresh, "calib": calib}


def merge(results: list[dict]) -> dict[str, pd.DataFrame]:
    keys = ("perf", "inc", "dca", "strat", "thresh", "calib")
    return {k: pd.DataFrame(sum((r[k] for r in results), [])) for k in keys}
