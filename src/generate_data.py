"""Generate synthetic merchant, transaction and chargeback data.

Every parameter below is documented in assumptions.md. Run:

    python src/generate_data.py

Outputs (in data/):
    merchants.csv    one row per merchant: attributes + whether / when it failed
    daily.csv        one row per merchant per trading day: transaction count and volume
    chargebacks.csv  one row per chargeback: transaction day, arrival day, amount, reason
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_MERCHANTS = 5_000
TRADING_DAYS = 365
RUNOFF_DAYS = 180

# segment: (share, avg_ticket, delivery_lag_days, base_cb_rate, annual_failure_hazard)
SEGMENTS = {
    "retail_physical":  (0.30,  40,  3, 0.0030, 0.03),
    "digital_goods":    (0.15,  15,  0, 0.0060, 0.05),
    "subscription":     (0.15,  20, 30, 0.0050, 0.06),
    "travel":           (0.10, 300, 60, 0.0040, 0.05),
    "events_ticketing": (0.10,  80, 45, 0.0040, 0.06),
    "food_delivery":    (0.10,  25,  0, 0.0020, 0.04),
    "electronics":      (0.10, 250,  5, 0.0050, 0.04),
}

NEW_MERCHANT_MONTHS = 6
NEW_MERCHANT_UPLIFT = 1.5
BUST_OUT_SHARE = 0.5          # share of failing merchants that ramp volume before failing
BUST_OUT_DAYS = 30
BUST_OUT_MULTIPLIER = 3.0
DISTRESS_DAYS = 60            # failing merchants that don't bust out decline over this many days
DISTRESS_FLOOR = 0.4          # ... down to this share of normal volume
RISK_SIGMA = 1.0              # spread of hidden merchant quality (lognormal, mean fixed at 1)
NON_DELIVERY_DISPUTE_RATE = 0.25
NORMAL_DISPUTE_MEAN_DAYS = 30
NON_DELIVERY_DISPUTE_MEAN_DAYS = 20
MAX_DISPUTE_DAYS = 120


def make_merchants(rng: np.random.Generator) -> pd.DataFrame:
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s][0] for s in names])
    segment = rng.choice(names, size=N_MERCHANTS, p=shares / shares.sum())

    m = pd.DataFrame({
        "merchant_id": [f"M{i:05d}" for i in range(N_MERCHANTS)],
        "segment": segment,
        "tenure_months": np.clip(np.round(rng.lognormal(np.log(18), 0.9, N_MERCHANTS)), 1, 120).astype(int),
        "monthly_volume": np.round(rng.lognormal(np.log(20_000), 1.0, N_MERCHANTS), 2),
    })
    m["avg_ticket"] = m["segment"].map(lambda s: SEGMENTS[s][1])
    m["delivery_lag_days"] = m["segment"].map(lambda s: SEGMENTS[s][2])
    base_cb = m["segment"].map(lambda s: SEGMENTS[s][3])
    base_hazard = m["segment"].map(lambda s: SEGMENTS[s][4])

    # Hidden merchant quality: the model never sees this directly.
    # mu = -sigma^2 / 2 keeps the mean at 1, so segment baselines are the average merchant's values.
    risk = rng.lognormal(-RISK_SIGMA**2 / 2, RISK_SIGMA, N_MERCHANTS)
    risk = np.where(m["tenure_months"] < NEW_MERCHANT_MONTHS, risk * NEW_MERCHANT_UPLIFT, risk)
    m["hidden_risk_multiplier"] = risk
    m["hidden_cb_rate"] = np.minimum(base_cb * risk, 0.05)

    # Failure day: geometric draw from a constant daily hazard. Beyond the window = no failure.
    daily_hazard = np.minimum(base_hazard * risk / 365, 0.5)
    fail_day = rng.geometric(daily_hazard)
    m["failed"] = fail_day < TRADING_DAYS
    m["fail_day"] = np.where(m["failed"], fail_day, -1)
    m["bust_out"] = m["failed"] & (rng.random(N_MERCHANTS) < BUST_OUT_SHARE)
    return m


def simulate_merchant(row, rng: np.random.Generator):
    last_day = row.fail_day if row.failed else TRADING_DAYS
    days = np.arange(last_day)

    expected_txns = row.monthly_volume / 30 / row.avg_ticket
    lam = np.full(len(days), expected_txns)
    if row.bust_out:
        # Volume climbs linearly from 1x to BUST_OUT_MULTIPLIER over the last BUST_OUT_DAYS.
        days_into_ramp = days - (last_day - BUST_OUT_DAYS) + 1
        progress = np.clip(days_into_ramp / BUST_OUT_DAYS, 0, 1)
        lam *= 1 + (BUST_OUT_MULTIPLIER - 1) * progress
    elif row.failed:
        # Distress: the business slowly loses customers before it closes.
        days_into_decline = days - (last_day - DISTRESS_DAYS) + 1
        progress = np.clip(days_into_decline / DISTRESS_DAYS, 0, 1)
        lam *= 1 - (1 - DISTRESS_FLOOR) * progress

    txns = rng.poisson(lam)
    volume = np.round(txns * row.avg_ticket * rng.lognormal(0, 0.2, len(days)), 2)
    daily = pd.DataFrame({"merchant_id": row.merchant_id, "day": days,
                          "txn_count": txns, "volume": volume})

    ticket = np.divide(volume, txns, out=np.zeros_like(volume), where=txns > 0)
    delivered_by = days + row.delivery_lag_days
    undelivered = row.failed & (delivered_by >= last_day)

    # Normal disputes on delivered transactions.
    n_normal = rng.binomial(txns * ~undelivered, row.hidden_cb_rate)
    # Non-delivery disputes when the merchant fails before delivering.
    n_nondel = rng.binomial(txns * undelivered, NON_DELIVERY_DISPUTE_RATE)

    cbs = []
    for counts, reason in ((n_normal, "normal"), (n_nondel, "non_delivery")):
        txn_day = np.repeat(days, counts)
        if len(txn_day) == 0:
            continue
        if reason == "normal":
            lag = rng.gamma(2.0, NORMAL_DISPUTE_MEAN_DAYS / 2, len(txn_day))
            arrival = txn_day + row.delivery_lag_days + np.minimum(lag, MAX_DISPUTE_DAYS)
        else:
            lag = rng.gamma(2.0, NON_DELIVERY_DISPUTE_MEAN_DAYS / 2, len(txn_day))
            arrival = last_day + lag
        cbs.append(pd.DataFrame({
            "merchant_id": row.merchant_id,
            "txn_day": txn_day,
            "arrival_day": np.round(arrival).astype(int),
            "amount": np.repeat(ticket, counts),
            "reason": reason,
        }))
    cb = pd.concat(cbs) if cbs else None
    return daily, cb


def main():
    rng = np.random.default_rng(SEED)
    merchants = make_merchants(rng)

    daily_parts, cb_parts = [], []
    for row in merchants.itertuples(index=False):
        daily, cb = simulate_merchant(row, rng)
        daily_parts.append(daily)
        if cb is not None:
            cb_parts.append(cb)

    daily = pd.concat(daily_parts, ignore_index=True)
    chargebacks = pd.concat(cb_parts, ignore_index=True)
    chargebacks = chargebacks[chargebacks["arrival_day"] < TRADING_DAYS + RUNOFF_DAYS]

    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    # Hidden columns (prefixed "hidden_") are kept for checking the simulation, never as model features.
    merchants.to_csv(out / "merchants.csv", index=False)
    daily.to_csv(out / "daily.csv", index=False)
    chargebacks.to_csv(out / "chargebacks.csv", index=False)

    print_summary(merchants, daily, chargebacks)


def print_summary(merchants, daily, chargebacks):
    vol = daily.groupby("merchant_id")["volume"].sum()
    cb = chargebacks.groupby("merchant_id")["amount"].sum()
    seg = merchants.set_index("merchant_id")["segment"]
    by_seg = pd.DataFrame({
        "merchants": seg.value_counts(),
        "failure_rate": merchants.groupby("segment")["failed"].mean(),
        "volume_musd": vol.groupby(seg).sum() / 1e6,
        "cb_ratio_bps": cb.groupby(seg).sum() / vol.groupby(seg).sum() * 1e4,
    }).round(3)
    print(f"merchants: {len(merchants):,}  failed: {merchants['failed'].sum():,}")
    print(f"daily rows: {len(daily):,}  chargebacks: {len(chargebacks):,}")
    print(chargebacks["reason"].value_counts().to_string())
    print("\nBy segment (cb_ratio_bps = chargeback amount / volume, in basis points):")
    print(by_seg.to_string())


if __name__ == "__main__":
    main()
