"""Merchant risk tiering: WoE scorecard vs gradient boosting, A-E tiers, expected loss.

Run after generate_data.py:

    python src/risk_model.py

Outputs:
    data/scored_snapshots.csv   every snapshot with PD, tier and expected loss (input to the policy engine)
    reports/risk_tiering.md     model comparison, tier table and expected-vs-realized loss
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from features import FEATURES_CATEGORICAL, FEATURES_NUMERIC, HORIZON_DAYS, SNAPSHOT_DAYS, build_snapshots

ROOT = Path(__file__).resolve().parent.parent
OOT_SNAPSHOTS = [240, 270]          # held out: later in time than anything the models train on
# Train only on snapshots whose 90-day outcome is fully observed by the first test snapshot.
# Standing on day 240, a snapshot from day 210 still has 60 days of its outcome in the future.
TRAIN_SNAPSHOTS = [t for t in SNAPSHOT_DAYS if t + HORIZON_DAYS <= min(OOT_SNAPSHOTS)]
N_BINS = 8
# Two models, two jobs: the scorecard sets tiers (transparent, explainable to merchants and
# auditors); gradient boosting is an early-warning overlay that flags merchants the scorecard
# rates A-C but that look like tier E to the boosted model.
WATCHLIST_TIERS = ["A", "B", "C"]
LGD = 1.0                           # before reserves: nothing is recovered once the merchant is gone
TIERS = ["A", "B", "C", "D", "E"]
TIER_SHARES = [0.40, 0.30, 0.15, 0.10, 0.05]   # share of training population, best to worst


# ---------- WoE scorecard ----------

def woe_fit(x: pd.Series, y: pd.Series, categorical: bool):
    """Return a mapping from bin to Weight of Evidence, plus the variable's Information Value."""
    if categorical:
        bins = x.astype(str)
        edges = None
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        bins = pd.cut(x, edges, include_lowest=True)
    tab = pd.crosstab(bins, y)
    good = (tab[0] + 0.5) / (tab[0].sum() + 0.5)    # +0.5 avoids log(0) in empty bins
    bad = (tab[1] + 0.5) / (tab[1].sum() + 0.5)
    woe = np.log(good / bad)
    iv = float(((good - bad) * woe).sum())
    return {"edges": edges, "woe": woe, "iv": iv}


def woe_transform(x: pd.Series, spec, categorical: bool):
    bins = x.astype(str) if categorical else pd.cut(x, spec["edges"], include_lowest=True)
    return bins.map(spec["woe"]).astype(float).fillna(0.0)


def fit_scorecard(train):
    cols = FEATURES_NUMERIC + FEATURES_CATEGORICAL
    specs = {c: woe_fit(train[c], train["target"], c in FEATURES_CATEGORICAL) for c in cols}
    X = pd.DataFrame({c: woe_transform(train[c], specs[c], c in FEATURES_CATEGORICAL) for c in cols})
    model = LogisticRegression(max_iter=1000).fit(X, train["target"])
    return specs, model


def predict_scorecard(df, specs, model):
    cols = list(specs)
    X = pd.DataFrame({c: woe_transform(df[c], specs[c], c in FEATURES_CATEGORICAL) for c in cols})
    return model.predict_proba(X)[:, 1]


# ---------- Gradient boosting ----------

def _gb_matrix(df):
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL].copy()
    X["segment"] = X["segment"].astype("category")
    return X


def fit_gb(train):
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100,
        categorical_features="from_dtype", random_state=42,
    )
    return model.fit(_gb_matrix(train), train["target"])


# ---------- Evaluation ----------

def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def assign_tiers(pd_values, cutoffs):
    return pd.cut(pd_values, [-np.inf, *cutoffs, np.inf], labels=TIERS)


def tier_table(df):
    g = df.groupby("tier", observed=False)
    return pd.DataFrame({
        "merchant_snapshots": g.size(),
        "share": g.size() / len(df),
        "predicted_pd": g["pd"].mean(),
        "observed_failure_rate": g["target"].mean(),
        "failures": g["target"].sum(),
        "expected_loss": g["expected_loss"].sum(),
        "realized_loss": g["realized_loss"].sum(),
    })


def main():
    merchants = pd.read_csv(ROOT / "data" / "merchants.csv")
    daily = pd.read_csv(ROOT / "data" / "daily.csv")
    chargebacks = pd.read_csv(ROOT / "data" / "chargebacks.csv")
    panel = build_snapshots(merchants, daily, chargebacks)

    is_train = panel["snapshot_day"].isin(TRAIN_SNAPSHOTS)
    is_oot = panel["snapshot_day"].isin(OOT_SNAPSHOTS)
    train, oot = panel[is_train], panel[is_oot]

    specs, scorecard = fit_scorecard(train)
    gb = fit_gb(train)

    results = []
    for name, predict in [
        ("WoE scorecard (logistic)", lambda d: predict_scorecard(d, specs, scorecard)),
        ("Gradient boosting", lambda d: gb.predict_proba(_gb_matrix(d))[:, 1]),
    ]:
        for split, d in [("train", train), ("out-of-time", oot)]:
            p = predict(d)
            results.append({"model": name, "sample": split,
                            "auc": roc_auc_score(d["target"], p), "ks": ks_stat(d["target"], p)})
    comparison = pd.DataFrame(results)

    panel["pd_scorecard"] = predict_scorecard(panel, specs, scorecard)
    panel["pd_gb"] = gb.predict_proba(_gb_matrix(panel))[:, 1]
    panel["pd"] = panel["pd_scorecard"]

    # Tier cut-offs come from the training period only, then stay fixed.
    cutoffs = np.quantile(panel.loc[is_train, "pd"], np.cumsum(TIER_SHARES)[:-1])
    panel["tier"] = assign_tiers(panel["pd"], cutoffs)

    # Early-warning overlay: gradient boosting PD above its own tier-E cut-off (fixed on training data).
    gb_alert_cutoff = np.quantile(panel.loc[is_train, "pd_gb"], 1 - TIER_SHARES[-1])
    panel["watchlist"] = panel["tier"].isin(WATCHLIST_TIERS) & (panel["pd_gb"] >= gb_alert_cutoff)

    # EAD calibration: like a credit conversion factor, scale the run-rate exposure estimate
    # so that, on training-period failures, estimated exposure matches realized loss.
    failed_train = panel[is_train & (panel["target"] == 1)]
    ead_factor = failed_train["realized_loss"].sum() / failed_train["ead"].sum()
    panel["ead_calibrated"] = panel["ead"] * ead_factor
    panel["expected_loss"] = panel["pd"] * panel["ead_calibrated"] * LGD

    out_csv = ROOT / "data" / "scored_snapshots.csv"
    panel.to_csv(out_csv, index=False)

    iv = pd.Series({c: s["iv"] for c, s in specs.items()}, name="information_value").sort_values(ascending=False)
    coef = pd.Series(scorecard.coef_[0], index=list(specs), name="coefficient")
    imp = permutation_importance(gb, _gb_matrix(oot), oot["target"], scoring="roc_auc",
                                 n_repeats=5, random_state=42)
    gb_importance = pd.Series(imp.importances_mean, index=_gb_matrix(oot).columns,
                              name="auc_drop_when_shuffled").sort_values(ascending=False)
    oot_tiers = tier_table(panel[is_oot])
    o = panel[is_oot]
    in_scope = o[o["tier"].isin(WATCHLIST_TIERS)]
    watch = {
        "in_scope": len(in_scope),
        "flagged": int(in_scope["watchlist"].sum()),
        "flagged_failure_rate": in_scope.loc[in_scope["watchlist"], "target"].mean(),
        "unflagged_failure_rate": in_scope.loc[~in_scope["watchlist"], "target"].mean(),
        "failures_in_scope": int(in_scope["target"].sum()),
        "failures_caught": int(in_scope.loc[in_scope["watchlist"], "target"].sum()),
    }
    failed_oot = panel[is_oot & (panel["target"] == 1)]
    ead_check = {
        "failures": len(failed_oot),
        "ead_factor": ead_factor,
        "sum_ead_calibrated": failed_oot["ead_calibrated"].sum(),
        "sum_realized_loss": failed_oot["realized_loss"].sum(),
        "realized_over_ead": failed_oot["realized_loss"].sum() / failed_oot["ead_calibrated"].sum(),
        "rank_correlation": failed_oot[["ead", "realized_loss"]].corr(method="spearman").iloc[0, 1],
    }

    write_report(train, oot, comparison, iv, coef, gb_importance, cutoffs, oot_tiers, ead_check, watch)
    print(comparison.round(3).to_string(index=False))
    print("\nOut-of-time tiers:")
    print(oot_tiers.round(4).to_string())
    print("\nEAD check:", {k: round(v, 3) for k, v in ead_check.items()})
    print("Watchlist:", {k: round(v, 4) for k, v in watch.items()})


def write_report(train, oot, comparison, iv, coef, gb_importance, cutoffs, oot_tiers, ead_check, watch):
    fmt_tiers = oot_tiers.copy()
    fmt_tiers["share"] = (fmt_tiers["share"] * 100).map("{:.1f}%".format)
    fmt_tiers["predicted_pd"] = (fmt_tiers["predicted_pd"] * 100).map("{:.2f}%".format)
    fmt_tiers["observed_failure_rate"] = (fmt_tiers["observed_failure_rate"] * 100).map("{:.2f}%".format)
    for c in ["expected_loss", "realized_loss"]:
        fmt_tiers[c] = fmt_tiers[c].map("${:,.0f}".format)

    lines = [
        "# Risk tiering results",
        "",
        "Generated by `src/risk_model.py`. All data is synthetic.",
        "",
        "## Setup",
        "",
        f"- Monthly point-in-time snapshots; target = merchant fails within the next {HORIZON_DAYS} days.",
        f"- Train: snapshots {TRAIN_SNAPSHOTS} "
        f"({len(train):,} merchant-snapshots, {int(train['target'].sum())} failures).",
        f"- Out-of-time test: snapshots {OOT_SNAPSHOTS} "
        f"({len(oot):,} merchant-snapshots, {int(oot['target'].sum())} failures).",
        f"- Snapshots between the two are left out: their {HORIZON_DAYS}-day outcomes would not yet be known "
        f"on day {min(OOT_SNAPSHOTS)}, when the model would go live.",
        "- Counts are merchant-snapshots, not unique merchants: one failure can appear in up to three "
        "consecutive snapshots.",
        "",
        "## Model comparison",
        "",
        comparison.round(3).to_markdown(index=False),
        "",
        "## Scorecard variables",
        "",
        "Information Value (IV) measures how well each variable separates failing from surviving merchants. "
        "Coefficients on WoE inputs should be negative (higher WoE = safer).",
        "",
        pd.concat([iv, coef], axis=1).round(3).to_markdown(),
        "",
        "## Gradient boosting: what drives it",
        "",
        "Permutation importance on the out-of-time sample: how much AUC drops when one feature is shuffled.",
        "",
        gb_importance.round(4).to_frame().to_markdown(),
        "",
        "## Out-of-time tiers",
        "",
        "Tiers are set by the **WoE scorecard** (primary model).",
        "",
        "Tier cut-offs are set on the training population "
        f"({', '.join(f'{s:.0%}' for s in TIER_SHARES)} from A to E) and applied unchanged to the test period. "
        f"PD cut-offs: {', '.join(f'{c:.2%}' for c in cutoffs)}.",
        "",
        fmt_tiers.to_markdown(),
        "",
        "Expected loss = PD x EAD x LGD, with LGD = 100% (no reserve held yet).",
        "",
        "## Early-warning overlay (gradient boosting)",
        "",
        f"Merchants the scorecard places in tiers {', '.join(WATCHLIST_TIERS)} are flagged for review when the "
        "gradient boosting PD is above its own tier-E cut-off. Out-of-time results:",
        "",
        f"- Merchant-snapshots in tiers {'-'.join([WATCHLIST_TIERS[0], WATCHLIST_TIERS[-1]])}: {watch['in_scope']:,}",
        f"- Flagged: {watch['flagged']:,} ({watch['flagged'] / watch['in_scope']:.1%})",
        f"- Failure rate, flagged: {watch['flagged_failure_rate']:.2%}; not flagged: {watch['unflagged_failure_rate']:.2%}",
        f"- Failures caught: {watch['failures_caught']} of {watch['failures_in_scope']} in these tiers",
        "",
        "## Exposure (EAD) check",
        "",
        "EAD is the run-rate exposure (undelivered sales likely to be disputed + sales still inside the normal "
        "dispute window), scaled by a factor fitted on training-period failures, similar to a credit conversion "
        "factor. The check below is on test-period failures only.",
        "",
        f"- Calibration factor (fitted on training failures): {ead_check['ead_factor']:.2f}",
        f"- Failures: {ead_check['failures']}",
        f"- Sum of calibrated EAD: ${ead_check['sum_ead_calibrated']:,.0f}",
        f"- Sum of realized loss: ${ead_check['sum_realized_loss']:,.0f}",
        f"- Realized / calibrated estimate: {ead_check['realized_over_ead']:.2f}",
        f"- Rank correlation (Spearman): {ead_check['rank_correlation']:.2f}",
        "",
    ]
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "risk_tiering.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
