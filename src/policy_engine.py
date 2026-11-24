"""Reserve and settlement-delay policy engine.

For each risk tier the platform chooses a control level. Controls reduce losses
(the platform holds merchant funds it can use to cover chargebacks if the merchant
fails) but cost revenue (merchants with funds locked up are more likely to leave).

Run after risk_model.py:

    python src/policy_engine.py

Outputs:
    reports/policy_engine.md       policy levels, frontier, recommended policy, comparisons
    reports/frontier.png           loss vs revenue frontier for each tiering scheme
    data/policy_results.csv        every evaluated policy
"""

from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import _matrices, realized_loss  # noqa: E402
from risk_model import TIER_SHARES, TRAIN_SNAPSHOTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Evaluation window: the out-of-time period. Tiers are read at day 240 and refreshed at day 270.
EVAL_START, EVAL_END = 240, 365
TIER_SNAPSHOTS = [240, 270]
TIERS = ["A", "B", "C", "D", "E"]

# Control levels: (rolling reserve rate, reserve hold days, extra settlement delay days).
LEVELS = {
    "None":       (0.00,   0, 0),
    "Light":      (0.05,  90, 0),
    "Medium":     (0.10,  90, 2),
    "Heavy":      (0.15, 120, 5),
    "Very heavy": (0.25, 180, 7),
}
LEVEL_NAMES = list(LEVELS)

SCHEMES = {
    # name: (group used for churn, group in force on the failure day)
    "Expected-loss tiers": ("el_tier", "el_tier_at_failure"),
    "PD tiers": ("tier", "tier_at_failure"),
    "Segment groups": ("segment_group", "segment_group"),
}
COLORS = {"Expected-loss tiers": "#2a78d6", "PD tiers": "#eb6834",
          "Segment groups": "#1baf7a", "Uniform": "#eda100"}
MARKERS = {"Expected-loss tiers": "o", "PD tiers": "s", "Segment groups": "^", "Uniform": "D"}

TAKE_RATE = 0.005            # platform net revenue per $ of volume
CHURN_SENSITIVITY = 0.4      # churn probability = 1 - exp(-sensitivity x funds locked / monthly volume)


def funds_locked_days(level):
    """Steady-state funds held, in days of sales: unsettled days + reserve rate x hold days."""
    rate, hold, delay = LEVELS[level]
    return delay + rate * hold


def churn_probability(level):
    locked_share_of_month = funds_locked_days(level) / 30
    return 1 - np.exp(-CHURN_SENSITIVITY * locked_share_of_month)


def held_at_failure(vol_row, fail_day, level):
    """Merchant funds the platform still holds on the failure day.

    Sales in the last `delay` days have not been paid out yet; of the sales paid out in the
    `hold` days before that, `rate` is still sitting in the rolling reserve.
    """
    rate, hold, delay = LEVELS[level]
    unsettled = vol_row[max(fail_day - delay, 0):fail_day].sum()
    reserve = rate * vol_row[max(fail_day - delay - hold, 0):max(fail_day - delay, 0)].sum()
    return unsettled + reserve


def load_inputs():
    merchants = pd.read_csv(ROOT / "data" / "merchants.csv")
    daily = pd.read_csv(ROOT / "data" / "daily.csv")
    chargebacks = pd.read_csv(ROOT / "data" / "chargebacks.csv")
    scored = pd.read_csv(ROOT / "data" / "scored_snapshots.csv")
    vol, _, _ = _matrices(merchants, daily, chargebacks)
    loss = realized_loss(merchants, chargebacks)
    return merchants, daily, chargebacks, scored, vol, loss


def segment_groups(merchants, vol, loss):
    """Model-free benchmark: rank segments by loss rate before the evaluation window, then
    split them into five ordered groups (same shape as tiers A-E)."""
    fail = merchants["fail_day"].to_numpy()
    before = (fail >= 0) & (fail < EVAL_START)
    seg_loss = pd.Series(np.where(before, loss.to_numpy(), 0.0)).groupby(merchants["segment"]).sum()
    seg_vol = pd.Series(vol[:, :EVAL_START].sum(axis=1)).groupby(merchants["segment"]).sum()
    ranked = (seg_loss / seg_vol).sort_values().index.tolist()      # safest first
    # 7 segments into 5 groups: 2, 1, 2, 1, 1 from safest to riskiest.
    sizes = [2, 1, 2, 1, 1]
    mapping, i = {}, 0
    for tier, n in zip(TIERS, sizes):
        for seg in ranked[i:i + n]:
            mapping[seg] = tier
        i += n
    return mapping, ranked


def el_rate_tiers(scored):
    """Tier by expected loss per $ of daily volume (PD x EAD / run-rate).

    Reserves are a share of volume, so the risk that matters is expected loss relative to
    volume: it combines how likely a merchant is to fail (PD) with how much exposure each
    $ of its sales creates (driven by delivery lag). Cut-offs use the same population shares
    as the PD tiers and are fixed on the training snapshots.
    """
    el_rate = np.divide(scored["expected_loss"], scored["daily_volume_30d"],
                        out=np.zeros(len(scored)), where=scored["daily_volume_30d"] > 0)
    train = el_rate[scored["snapshot_day"].isin(TRAIN_SNAPSHOTS).to_numpy()]
    cutoffs = np.quantile(train, np.cumsum(TIER_SHARES)[:-1])
    return pd.Series(pd.cut(el_rate, [-np.inf, *cutoffs, np.inf], labels=TIERS).astype(str),
                     index=scored.index)


def build_population(merchants, scored, vol, loss, seg_map):
    """One row per merchant alive at day 240: its group under each scheme, volume, and failure."""
    scored = scored.assign(el_tier=el_rate_tiers(scored))
    s240 = scored[scored["snapshot_day"] == TIER_SNAPSHOTS[0]].set_index("merchant_id")
    s270 = scored[scored["snapshot_day"] == TIER_SNAPSHOTS[1]].set_index("merchant_id")
    m = merchants.set_index("merchant_id").loc[s240.index]
    row_of = pd.Series(np.arange(len(merchants)), index=merchants["merchant_id"])

    pop = pd.DataFrame(index=s240.index)
    pop["row"] = row_of.loc[pop.index].to_numpy()
    pop["fail_day"] = m["fail_day"]
    pop["fails_in_window"] = (m["fail_day"] >= EVAL_START) & (m["fail_day"] < EVAL_END)
    pop["realized_loss"] = np.where(pop["fails_in_window"], loss.loc[pop.index], 0.0)
    pop["window_volume"] = vol[pop["row"], EVAL_START:EVAL_END].sum(axis=1)
    # Revenue at stake if the merchant leaves: what it would have processed in the window,
    # estimated from its run-rate at day 240 (we can't use actual future volume).
    pop["volume_at_stake"] = s240["daily_volume_30d"] * (EVAL_END - EVAL_START)
    # Tier in force on the failure day: refreshed at day 270 if the merchant was still alive.
    tier_240 = s240["tier"]
    tier_270 = s270["tier"].reindex(pop.index)
    pop["tier_at_failure"] = np.where((pop["fail_day"] >= TIER_SNAPSHOTS[1]) & tier_270.notna(),
                                      tier_270, tier_240)
    pop["tier"] = tier_240
    el_270 = s270["el_tier"].reindex(pop.index)
    pop["el_tier"] = s240["el_tier"]
    pop["el_tier_at_failure"] = np.where((pop["fail_day"] >= TIER_SNAPSHOTS[1]) & el_270.notna(),
                                         el_270, pop["el_tier"])
    pop["segment_group"] = m["segment"].map(seg_map)
    return pop


def evaluate(pop, vol, group_col_churn, group_col_failure, assignment):
    """Residual loss and revenue lost for one policy: a dict {group: level}."""
    churn = pop[group_col_churn].map(lambda g: churn_probability(assignment[g]))
    revenue_lost = float((churn * pop["volume_at_stake"] * TAKE_RATE).sum())

    failed = pop[pop["fails_in_window"]]
    recovered = 0.0
    for mid, r in failed.iterrows():
        held = held_at_failure(vol[r["row"]], int(r["fail_day"]), assignment[r[group_col_failure]])
        recovered += min(held, r["realized_loss"])
    gross = float(failed["realized_loss"].sum())
    return {"residual_loss": max(gross - recovered, 0.0), "revenue_lost": revenue_lost}


def monotone_assignments():
    """Every assignment where a worse tier never gets a lighter control level: 126 of them."""
    for combo in combinations_with_replacement(range(len(LEVEL_NAMES)), len(TIERS)):
        yield {t: LEVEL_NAMES[i] for t, i in zip(TIERS, combo)}


def pareto_front(df):
    """Policies not beaten on both residual loss and revenue lost by any other policy."""
    d = df.sort_values(["revenue_lost", "residual_loss"])
    best, keep = np.inf, []
    for idx, r in d.iterrows():
        if r["residual_loss"] < best - 1e-9:
            keep.append(idx)
            best = r["residual_loss"]
    return df.loc[keep].sort_values("revenue_lost")


def run_all(pop, vol):
    rows = []
    for scheme, (churn_col, fail_col) in SCHEMES.items():
        for a in monotone_assignments():
            rows.append({"scheme": scheme, **{f"level_{t}": a[t] for t in TIERS},
                         **evaluate(pop, vol, churn_col, fail_col, a)})
    for level in LEVEL_NAMES:
        a = {t: level for t in TIERS}
        rows.append({"scheme": "Uniform", **{f"level_{t}": level for t in TIERS},
                     **evaluate(pop, vol, "tier", "tier_at_failure", a)})
    res = pd.DataFrame(rows)
    res["total_cost"] = res["residual_loss"] + res["revenue_lost"]
    return res


def plot_frontier(res, baseline_loss, path):
    # Label each line where it is well separated from the others: (revenue lost in $k, text offset).
    label_at = {"Expected-loss tiers": (230, (6, -14)), "PD tiers": (180, (8, 8)),
                "Segment groups": (95, (10, 0)), "Uniform": (200, (10, 2))}
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for scheme in ["Uniform", "Segment groups", "PD tiers", "Expected-loss tiers"]:
        d = res[res["scheme"] == scheme]
        f = pareto_front(d) if scheme != "Uniform" else d.sort_values("revenue_lost")
        x, y = f["revenue_lost"].to_numpy() / 1e3, f["residual_loss"].to_numpy() / 1e3
        ax.plot(x, y, color=COLORS[scheme], linewidth=2, marker=MARKERS[scheme], markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=scheme,
                zorder=3 if scheme == "Expected-loss tiers" else 2)
        target_x, offset = label_at[scheme]
        i = int(np.argmin(np.abs(x - target_x)))
        ax.annotate(scheme, (x[i], y[i]), xytext=offset, textcoords="offset points", fontsize=9,
                    color="#3d3d3a", ha="right" if offset[0] < 0 else "left")
    ax.axhline(baseline_loss / 1e3, color="#9a9890", linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("No controls", (0, baseline_loss / 1e3), xytext=(4, 4), textcoords="offset points",
                fontsize=9, color="#6b6a64")
    ax.set_xlabel("Revenue lost to merchant attrition ($k)", color="#3d3d3a")
    ax.set_ylabel("Residual chargeback loss ($k)", color="#3d3d3a")
    ax.set_title("Loss vs attrition frontier, out-of-time period", loc="left", color="#1f1f1d", fontsize=12)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c9c7bd")
    ax.tick_params(colors="#6b6a64")
    ax.grid(color="#e8e6df", linewidth=0.8)
    ax.set_xlim(0, 1000)   # the heaviest uniform policies sit further right, at zero loss
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def churn_sensitivity_check(pop, vol, values=(0.2, 0.4, 0.8)):
    """Recommended expected-loss-tier policy under different churn sensitivities."""
    global CHURN_SENSITIVITY
    original, rows = CHURN_SENSITIVITY, []
    churn_col, fail_col = SCHEMES["Expected-loss tiers"]
    for v in values:
        CHURN_SENSITIVITY = v
        evals = [(a, evaluate(pop, vol, churn_col, fail_col, a)) for a in monotone_assignments()]
        a, r = min(evals, key=lambda e: e[1]["residual_loss"] + e[1]["revenue_lost"])
        rows.append({"churn_sensitivity": v, "churn_at_light": f"{churn_probability('Light'):.1%}",
                     **{t: a[t] for t in TIERS},
                     "residual_loss": f"${r['residual_loss']:,.0f}", "revenue_lost": f"${r['revenue_lost']:,.0f}"})
    CHURN_SENSITIVITY = original
    return pd.DataFrame(rows)


def main():
    merchants, _, _, scored, vol, loss = load_inputs()
    seg_map, ranked = segment_groups(merchants, vol, loss)
    pop = build_population(merchants, scored, vol, loss, seg_map)
    res = run_all(pop, vol)
    res.to_csv(ROOT / "data" / "policy_results.csv", index=False)

    baseline = res[(res["scheme"] == "Uniform") & (res["level_A"] == "None")].iloc[0]
    best = {s: res[res["scheme"] == s].sort_values("total_cost").iloc[0]
            for s in [*SCHEMES, "Uniform"]}

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    plot_frontier(res, baseline["residual_loss"], out / "frontier.png")
    sensitivity = churn_sensitivity_check(pop, vol)
    write_report(pop, res, baseline, best, ranked, seg_map, sensitivity, out / "policy_engine.md")

    print(f"Population: {len(pop):,} merchants; failures in window: {int(pop['fails_in_window'].sum())}; "
          f"gross loss ${pop['realized_loss'].sum():,.0f}")
    for s, r in best.items():
        print(f"{s:15s} best total cost ${r['total_cost']:,.0f}  (loss ${r['residual_loss']:,.0f}, "
              f"revenue ${r['revenue_lost']:,.0f})  {[r[f'level_{t}'] for t in TIERS]}")


def write_report(pop, res, baseline, best, ranked, seg_map, sensitivity, path):
    lv = pd.DataFrame([{"level": k, "reserve_rate": f"{v[0]:.0%}", "hold_days": v[1], "extra_delay_days": v[2],
                        "funds_locked_days_of_sales": round(funds_locked_days(k), 1),
                        "churn_probability": f"{churn_probability(k):.1%}"} for k, v in LEVELS.items()])

    def fmt(r):
        return {"scheme": r["scheme"], **{t: r[f"level_{t}"] for t in TIERS},
                "residual_loss": f"${r['residual_loss']:,.0f}", "revenue_lost": f"${r['revenue_lost']:,.0f}",
                "total_cost": f"${r['total_cost']:,.0f}"}

    best_tbl = pd.DataFrame([fmt(baseline)] + [fmt(r) for r in best.values()])
    best_tbl.iloc[0, 0] = "No controls"
    front = pareto_front(res[res["scheme"] == "Expected-loss tiers"])
    front_tbl = pd.DataFrame([fmt(r) for _, r in front.iterrows()]).drop(columns="scheme")

    # Like-for-like: at the model's recommended revenue cost, how much loss does each scheme leave?
    budget = best["Expected-loss tiers"]["revenue_lost"]
    at_budget = []
    for s in [*SCHEMES, "Uniform"]:
        d = res[(res["scheme"] == s) & (res["revenue_lost"] <= budget * 1.0001)]
        at_budget.append({"scheme": s, "lowest_residual_loss": f"${d['residual_loss'].min():,.0f}"})

    gross = pop["realized_loss"].sum()
    rec = best["Expected-loss tiers"]
    lines = [
        "# Reserve policy engine results",
        "",
        "Generated by `src/policy_engine.py`. All data is synthetic; every parameter is in `assumptions.md`.",
        "",
        "## Setup",
        "",
        f"- Population: {len(pop):,} merchants alive on day {EVAL_START}. Evaluation window: days "
        f"{EVAL_START}–{EVAL_END - 1}, the out-of-time period the risk models never trained on.",
        f"- Failures in the window: {int(pop['fails_in_window'].sum())}; gross loss with no controls: ${gross:,.0f}.",
        "- Tiers are read at day 240 and refreshed at day 270 for merchants still alive.",
        f"- Revenue lost = churn probability x merchant's run-rate volume over the window x {TAKE_RATE:.1%} net take rate.",
        "",
        "## Control levels",
        "",
        "A rolling reserve withholds a share of each payout for a hold period; an extra settlement delay holds "
        "the whole payout for a few days. On the failure day, the platform keeps whatever merchant funds it "
        "still holds and uses them to cover chargebacks.",
        "",
        lv.to_markdown(index=False),
        "",
        "## Policies compared",
        "",
        "Each scheme assigns a control level to five ordered groups, with worse groups never getting lighter "
        "controls (126 combinations).",
        "",
        "- **Expected-loss tiers**: merchants ranked by expected loss per $ of daily volume (PD × EAD ÷ run-rate), "
        "cut at the same population shares as the PD tiers. Reserves are a share of volume, so this is the "
        "risk that matters: it combines how likely a merchant is to fail with how much exposure each $ of "
        "its sales creates.",
        "- **PD tiers**: the scorecard's A–E tiers, ranked on probability of failure only.",
        "- **Segment groups**: a model-free benchmark. Segments are ranked by their loss rate before day 240 and split into five "
        f"groups ({', '.join(f'{seg} = {seg_map[seg]}' for seg in ranked)}).",
        "- **Uniform**: one level for everyone.",
        "",
        "![Frontier](frontier.png)",
        "",
        "## Lowest total cost by scheme",
        "",
        "Total cost = residual loss + revenue lost.",
        "",
        best_tbl.to_markdown(index=False),
        "",
        f"At the recommended policy's revenue cost (${budget:,.0f}), the lowest residual loss each scheme can reach:",
        "",
        pd.DataFrame(at_budget).to_markdown(index=False),
        "",
        "## Recommended policy (expected-loss tiers)",
        "",
        ", ".join(f"Tier {t}: {rec[f'level_{t}']}" for t in TIERS) + ".",
        "",
        f"- Residual loss ${rec['residual_loss']:,.0f} vs ${gross:,.0f} with no controls "
        f"({1 - rec['residual_loss'] / gross:.0%} reduction).",
        f"- Revenue lost to attrition: ${rec['revenue_lost']:,.0f}.",
        "",
        "Several neighbouring policies on the frontier have total costs within a few percent of this one, so the "
        "optimum is flat: the choice between them is a question of risk appetite rather than precision.",
        "",
        "## Sensitivity to the churn assumption",
        "",
        "Churn sensitivity is the least certain input. The recommended policy under lower and higher values:",
        "",
        sensitivity.to_markdown(index=False),
        "",
        "## Efficient frontier (expected-loss tiers)",
        "",
        "Policies that no other policy beats on both residual loss and revenue lost.",
        "",
        front_tbl.to_markdown(index=False),
        "",
        "## Limitations",
        "",
        "- Churn sensitivity and take rate are illustrative assumptions; the recommended policy moves if they change.",
        "- Churn is applied only to revenue: the loss side assumes merchants stay long enough to fail. In practice "
        "some risky merchants would leave, which would lower losses further.",
        "- Each merchant is assumed to have been under its tier's controls long enough for its reserve to be full.",
        f"- The window has only {int(pop['fails_in_window'].sum())} failures, so differences between nearby policies "
        "are noisy.",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
