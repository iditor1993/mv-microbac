"""S11 交叉库外推（探索性）：ARMD 训练的模型直接预测 MIMIC，反之亦然。

注意：两库的既往培养计数分布差异极大（MIMIC 人均培养更多），
标准化只用训练库的参数，因此这是**transportability** 检验，不是性能估计。
产出：results/tables/c11_cross_database.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis import INCREMENTS, MODELS, fit_predict
from common import ARMD_DB, MIMIC_DB, TABLES, connect, ensure_dirs
from features import build_armd, build_mimic, prepare


def transfer(train, test, train_name, test_name):
    rows = []
    preds = {}
    for name, cols in MODELS.items():
        preds[name] = fit_predict(train[cols].to_numpy(float),
                                  train["y"].to_numpy(int),
                                  test[cols].to_numpy(float))
    y = test["y"].to_numpy(int)
    for name in MODELS:
        rows.append({"train": train_name, "test": test_name, "kind": "auroc",
                     "model": name, "value": roc_auc_score(y, preds[name]),
                     "n": len(y), "events": int(y.sum())})
    for old, new, label in INCREMENTS:
        rows.append({"train": train_name, "test": test_name, "kind": "delta_auroc",
                     "model": label,
                     "value": roc_auc_score(y, preds[new]) - roc_auc_score(y, preds[old]),
                     "n": len(y), "events": int(y.sum())})
    return rows


def main():
    ensure_dirs()
    con = connect(ARMD_DB)
    armd = prepare(build_armd(con))
    con.close()
    con = connect(MIMIC_DB)
    mimic = prepare(build_mimic(con))
    con.close()

    rows = transfer(armd, mimic, "ARMD-MGB", "MIMIC-IV")
    rows += transfer(mimic, armd, "MIMIC-IV", "ARMD-MGB")
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "c11_cross_database.csv", index=False)
    print(d[["train", "test", "kind", "model", "value", "n", "events"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
