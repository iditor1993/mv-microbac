"""敏感性分析共用：预测 + 增量汇总。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import INCREMENTS, MODELS, run_models
from common import bootstrap_auroc, bootstrap_delta_auroc, metric_block


def increments(df, preds, eval_mask, groups, cohort, variant, B=400):
    y = df["y"].to_numpy(int)[eval_mask]
    g = np.asarray(groups)[eval_mask]
    rows = []
    for name in MODELS:
        if name not in preds:
            continue
        m = metric_block(y, preds[name][eval_mask])
        ci = bootstrap_auroc(y, preds[name][eval_mask], g, B=B)
        rows.append({"cohort": cohort, "variant": variant, "model": name, "n": len(y),
                     "events": int(y.sum()), "auroc": m["auroc"],
                     "auroc_lo": ci["auroc_lo"], "auroc_hi": ci["auroc_hi"]})
    for old, new, label in INCREMENTS:
        if old not in preds or new not in preds:
            continue
        d = bootstrap_delta_auroc(y, preds[old][eval_mask], preds[new][eval_mask], groups=g, B=B)
        rows.append({"cohort": cohort, "variant": variant, "model": label, "n": len(y),
                     "events": int(y.sum()), "auroc": d["delta_auroc"],
                     "auroc_lo": d["delta_lo"], "auroc_hi": d["delta_hi"]})
    return rows


def cv_or_temporal(df, groups, test_mask=None, models=None):
    if test_mask is None:
        return run_models(df, groups=groups, models=models)
    return run_models(df, test_mask=test_mask, models=models)
