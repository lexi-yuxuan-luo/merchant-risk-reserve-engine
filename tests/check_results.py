"""Checks that a fresh pipeline run still supports the conclusions in the README.

Exact numbers can differ slightly across machines (floating-point differences in the
linear algebra libraries can move a merchant across a tier boundary), so this checks
the findings with tolerances rather than requiring byte-identical reports.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def main():
    scored = pd.read_csv(ROOT / "data" / "scored_snapshots.csv")
    oot = scored[scored["snapshot_day"].isin([240, 270])]
    auc_scorecard = roc_auc_score(oot["target"], oot["pd_scorecard"])
    auc_gb = roc_auc_score(oot["target"], oot["pd_gb"])
    assert auc_scorecard > 0.70, f"scorecard out-of-time AUC fell to {auc_scorecard:.3f}"
    assert auc_gb > 0.75, f"gradient boosting out-of-time AUC fell to {auc_gb:.3f}"

    train = scored[scored["snapshot_day"].isin([60, 90, 120, 150])]
    rates = train.groupby("tier")["target"].mean()
    assert rates.is_monotonic_increasing, f"training tier failure rates not monotonic: {rates.round(4).to_dict()}"

    # keep_default_na=False: the control level "None" must stay a string, not become NaN
    res = pd.read_csv(ROOT / "data" / "policy_results.csv", keep_default_na=False)
    best = res.groupby("scheme")["total_cost"].min()
    no_controls = res.loc[(res["scheme"] == "Uniform") & (res["level_A"] == "None"), "total_cost"].iloc[0]
    assert best["Expected-loss tiers"] < best["PD tiers"] < best["Segment groups"] < best["Uniform"] < no_controls, \
        f"scheme ranking changed: {best.round(0).to_dict()}"
    assert best["Expected-loss tiers"] < 0.5 * best["Uniform"], "expected-loss tiering no longer halves the uniform cost"

    print(f"OK  AUC scorecard {auc_scorecard:.3f}, GB {auc_gb:.3f}; best total cost by scheme: "
          + ", ".join(f"{k} ${v:,.0f}" for k, v in best.sort_values().items()))


if __name__ == "__main__":
    main()
