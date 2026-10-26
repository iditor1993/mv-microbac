"""Publication figures (English labels only).

Outputs (300 dpi) to results/figures/:
    fig1_roc_curves.png        ROC curves M0-M3, both cohorts
    fig2_calibration.png       M3 calibration: raw vs intercept-only, MIMIC and ARMD
    fig3_dca.png               Decision curves (first quadrant) + delta net benefit
    fig4_increments.png        Forest plot of the six nested increments
    fig5_learner_comparison.png Bar chart, logistic vs XGBoost increments
    figS1_nested_auroc_ci.png  AUROC with 95% cluster-bootstrap CI (supplement)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, TABLES, ensure_dirs

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

C_ARMD, C_MIMIC = "#2E5E8C", "#C0504D"

MODEL_ORDER = ["M0 人口学+菌种", "M1 +既往用药", "M2 +既往培养数量",
               "M2b +既往同菌", "M3 +既往同菌药敏", "M4 +他菌菌种", "M5 +他菌耐药"]
MODEL_EN = {
    "M0 人口学+菌种": "M0 demographics + organism",
    "M1 +既往用药": "M1 + prior antibiotics",
    "M2 +既往培养数量": "M2 + culture count",
    "M2b +既往同菌": "M2b + prior organism",
    "M3 +既往同菌药敏": "M3 + prior susceptibility",
    "M4 +他菌菌种": "M4 + other organism",
    "M5 +他菌耐药": "M5 + other resistance",
}
INC_EN = {
    "M1−M0 用药史": "M1−M0\nprior antibiotics",
    "M2−M1 培养数量": "M2−M1\nculture count",
    "M2b−M2 同菌菌种": "M2b−M2\nprior organism",
    "M3−M2b 同菌药敏（主）": "M3−M2b\nprior susceptibility",
    "M4−M3 他菌菌种": "M4−M3\nother organism",
    "M5−M4 他菌耐药": "M5−M4\nother resistance",
}
ROC_STYLE = [
    ("M0 人口学+菌种", "M0 demographics", "#8c8c8c"),
    ("M1 +既往用药", "M1 + prior antibiotics", "#E69F00"),
    ("M2 +既往培养数量", "M2 + culture count", "#009E73"),
    ("M2b +既往同菌", "M2b + prior organism", "#56B4E9"),
    ("M3 +既往同菌药敏", "M3 + prior susceptibility", "#0072B2"),
]


def load(name):
    return pd.read_csv(TABLES / name)


# ---------------------------------------------------------------------------
def _roc_panels():
    """Recompute predictions on the fly; nothing individual-level is written."""
    from analysis import run_models
    from common import ARMD_DB, MIMIC_DB, connect
    from features import build_armd, build_mimic, prepare

    out = {}
    con = connect(ARMD_DB)
    armd = prepare(build_armd(con))
    con.close()
    g = armd["patient_id"].to_numpy()
    preds = run_models(armd, groups=g)
    y = armd["y"].to_numpy(int)
    out["ARMD-MGB (5-fold patient-grouped CV)"] = {
        "y": y, "preds": preds, "n": len(y), "events": int(y.sum())}

    con = connect(MIMIC_DB)
    mimic = prepare(build_mimic(con))
    con.close()
    test = (mimic["year"] >= 2015).to_numpy()
    preds = run_models(mimic, test_mask=test)
    y = mimic["y"].to_numpy(int)[test]
    out["MIMIC-IV (temporal validation)"] = {
        "y": y, "preds": {k: v[test] for k, v in preds.items()},
        "n": int(test.sum()), "events": int(y.sum())}
    return out


def fig1_roc():
    from sklearn.metrics import auc, roc_curve

    panels = _roc_panels()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    for ax, (title, d) in zip(axes, panels.items()):
        ax.plot([0, 1], [0, 1], "-", color="#cccccc", lw=1, zorder=1)
        for col, label, color in ROC_STYLE:
            fpr, tpr, _ = roc_curve(d["y"], d["preds"][col])
            ax.plot(fpr, tpr, color=color, lw=2.1 if col.startswith("M3") else 1.5,
                    label=f"{label} (AUC={auc(fpr, tpr):.3f})",
                    zorder=3 if col.startswith("M3") else 2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=.25, lw=.5)
        ax.text(0.97, 0.03, f"n={d['n']:,}, events={d['events']:,}",
                ha="right", va="bottom", fontsize=7.5, color="#555555")
        ax.legend(frameon=True, fontsize=7.2, loc="lower right",
                  framealpha=.95, edgecolor="#dddddd")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figS1_nested_ci():
    perf = load("c01_model_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, cohort, color in zip(axes, ["ARMD-MGB", "MIMIC-IV"], [C_ARMD, C_MIMIC]):
        d = perf[perf.cohort == cohort].set_index("model").loc[MODEL_ORDER]
        y = np.arange(len(d))[::-1]
        ax.errorbar(d.auroc, y, xerr=[d.auroc - d.auroc_lo, d.auroc_hi - d.auroc],
                    fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=3, ms=5)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_EN[m] for m in MODEL_ORDER], fontsize=8)
        ax.set_xlabel("AUROC (95% CI, patient-clustered bootstrap)")
        ax.set_title(cohort)
        ax.grid(axis="x", alpha=.3, lw=.5)
        ax.set_xlim(0.45, 0.90)
    axes[0].set_ylabel("Nested model")
    fig.tight_layout()
    fig.savefig(FIGURES / "figS1_nested_auroc_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_calibration():
    bins = load("c10_calibration_bins.csv")
    m3 = "M3 +既往同菌药敏"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, cohort, stage, title in [
        (axes[0], "MIMIC-IV", "raw", "MIMIC-IV: raw (model trained 2008-2014)"),
        (axes[1], "MIMIC-IV", "LR-IC", "MIMIC-IV: intercept recalibrated"),
        (axes[2], "ARMD-MGB", "raw", "ARMD-MGB: patient-grouped CV (reference)"),
    ]:
        d = bins[(bins.cohort == cohort) & (bins.model == m3) & (bins.stage == stage)]
        ax.plot([0, .85], [0, .85], "--", color="gray", lw=1)
        ax.plot(d.pred, d.obs, "o-", color=C_ARMD if cohort == "ARMD-MGB" else C_MIMIC, ms=4)
        ax.set_title(title, fontsize=8.2)
        ax.set_xlabel("Predicted probability (decile)")
        ax.grid(alpha=.3, lw=.5)
    axes[0].set_ylabel("Observed resistance proportion")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_dca():
    """Decision curves, first quadrant only.

    Net benefit diverges as threshold -> 1 (factor t/(1-t)). Plotting the whole
    0.01-0.99 grid and then clipping x made matplotlib scale y on the full data,
    flattening every curve. Slice before plotting.
    """
    dca = load("c01_dca.csv")
    dca = dca[(dca.threshold >= 0.05) & (dca.threshold <= 0.50)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, cohort, color in zip(axes[:2], ["ARMD-MGB", "MIMIC-IV"], [C_ARMD, C_MIMIC]):
        d = dca[dca.cohort == cohort]
        ax.plot(d.threshold, d.nb_all, ":", color="#777777", lw=1.4, label="Treat all")
        ax.plot(d.threshold, d.nb_old, "-", color="#9a9a9a", lw=1.8, label="M1: prior antibiotics")
        ax.plot(d.threshold, d.nb_new, "-", color=color, lw=2.4, label="M3: + prior susceptibility")
        ax.axvspan(0.15, 0.35, color=color, alpha=.05, lw=0)
        top = float(max(d.nb_new.max(), d.nb_old.max(), d.nb_all.max()))
        ax.set_ylim(0, top * 1.18)
        ax.set_xlim(0.05, 0.50)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"{cohort} (curves clipped at y<0)", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.grid(alpha=.3, lw=.5)
    ax = axes[2]
    for cohort, color in [("ARMD-MGB", C_ARMD), ("MIMIC-IV", C_MIMIC)]:
        d = dca[dca.cohort == cohort]
        delta = d.nb_new.to_numpy() - d.nb_old.to_numpy()
        ax.plot(d.threshold, delta, "-", color=color, lw=2.2, label=cohort)
        ax.fill_between(d.threshold, 0, delta, color=color, alpha=.12, lw=0)
    ax.axhline(0, color="black", lw=.8)
    ax.axvspan(0.15, 0.35, color="#888888", alpha=.06, lw=0)
    ax.set_xlim(0.05, 0.50)
    ax.set_ylim(0, None)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Delta net benefit (M3 - M1)")
    ax.set_title("Incremental net benefit", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.3, lw=.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_dca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_increments():
    inc = load("c01_increments.csv")
    order = list(dict.fromkeys(inc.increment))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    h = 0.36
    for cohort, color, off in [("ARMD-MGB", C_ARMD, h / 2), ("MIMIC-IV", C_MIMIC, -h / 2)]:
        d = inc[inc.cohort == cohort].set_index("increment").loc[order]
        y = np.arange(len(order))[::-1] + off
        ax.errorbar(d.delta_auroc, y, xerr=[d.delta_auroc - d.delta_lo, d.delta_hi - d.delta_auroc],
                    fmt="s", color=color, ecolor=color, capsize=3, ms=5, lw=1.2, label=cohort)
    ax.axvline(0, color="black", lw=.8)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels([INC_EN.get(o, o) for o in order], fontsize=8.2)
    ax.set_xlabel("Delta AUROC (95% CI)")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=.3, lw=.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_increments.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5_learner():
    lr = load("c01_increments.csv")
    xg = load("c06_xgb_increments.csv")
    order = list(dict.fromkeys(lr.increment))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    width = 0.38
    x = np.arange(len(order))
    for ax, cohort in zip(axes, ["ARMD-MGB", "MIMIC-IV"]):
        a = lr[lr.cohort == cohort].set_index("increment").loc[order]
        b = xg[xg.cohort == cohort].set_index("increment").loc[order]
        for off, d, color, name in [(-width / 2, a, "#0072B2", "Logistic regression"),
                                    (width / 2, b, "#D55E00", "XGBoost")]:
            err = np.vstack([d.delta_auroc - d.delta_lo, d.delta_hi - d.delta_auroc])
            ax.bar(x + off, d.delta_auroc, width, yerr=err, capsize=2.5,
                   color=color, alpha=.88, label=name,
                   error_kw={"lw": 1, "ecolor": "#333333"})
        ax.axhline(0, color="black", lw=.9)
        ax.set_xticks(x)
        ax.set_xticklabels([INC_EN.get(o, o) for o in order], fontsize=7.8)
        ax.set_ylabel("Delta AUROC")
        ax.set_title(cohort, fontsize=10)
        ax.grid(axis="y", alpha=.3, lw=.5)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_learner_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5_cascade():
    """Trade-off between carbapenem use and inadequate empirical coverage."""
    d = load("c14_cascade.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, cohort, color in zip(axes, ["ARMD-MGB", "MIMIC-IV"], [C_ARMD, C_MIMIC]):
        d["carbapenem_use"] = pd.to_numeric(d["carbapenem_use"], errors="coerce")
        d["inadequate_coverage"] = pd.to_numeric(d["inadequate_coverage"], errors="coerce")
        e = d[d.cohort == cohort]
        base = e[e.strategy == "Cefepime for all"].iloc[0]
        top = e[e.strategy == "Meropenem for all"].iloc[0]
        rule = e[e.strategy.str.startswith("Rule")].iloc[0]
        ax.scatter([base.carbapenem_use * 100], [base.inadequate_coverage * 100],
                   s=70, marker="s", color="#333333", zorder=5, label="Cefepime for all")
        ax.scatter([top.carbapenem_use * 100], [top.inadequate_coverage * 100],
                   s=70, marker="D", color="#333333", zorder=5, label="Meropenem for all")
        ax.scatter([rule.carbapenem_use * 100], [rule.inadequate_coverage * 100],
                   s=90, marker="*", color="#E69F00", zorder=6, label="Rule: prior resistant")
        for prefix, col, mk, lab in [("M1-guided", "#777777", "o", "M1-guided (prior antibiotics)"),
                                     ("M3-guided", color, "o", "M3-guided (+ prior susceptibility)")]:
            sub = e[e.strategy.str.startswith(prefix)].copy()
            sub["t"] = sub.strategy.str.extract(r"coverage (\d+)%").astype(int)
            sub = sub.sort_values("t")
            ax.plot(sub.carbapenem_use * 100, sub.inadequate_coverage * 100, "-", color=col, lw=2,
                    marker=mk, ms=5, label=lab, zorder=4)
            for _, r in sub.iterrows():
                ax.annotate(f"{r.t}%", (r.carbapenem_use * 100, r.inadequate_coverage * 100),
                            textcoords="offset points", xytext=(5, 6), fontsize=7, color=col)
        ax.set_xlim(-4, 104)
        ax.set_ylim(0, max(25, 1.15 * max(e.inadequate_coverage) * 100))
        ax.set_xlabel("Carbapenem use (%)")
        ax.set_ylabel("Inadequate empirical coverage (%)")
        ax.set_title(cohort, fontsize=10)
        ax.grid(alpha=.25, lw=.5)
    axes[0].legend(frameon=False, fontsize=7.6, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_cascade_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ensure_dirs()
    for f in (fig1_roc, figS1_nested_ci, fig2_calibration, fig3_dca,
              fig4_increments, fig5_learner, fig5_cascade):
        f()
        print("ok", f.__name__)