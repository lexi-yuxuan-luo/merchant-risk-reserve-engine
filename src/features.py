"""Point-in-time features for merchant failure prediction.

A snapshot at day t asks: "Standing here on day t, using only what we know so far,
will this merchant fail in the next HORIZON_DAYS?"

    observation: days before t          ->  features
    performance: days t .. t+HORIZON-1  ->  target (failed or not)

Only transactions before t and chargebacks that have *arrived* before t are used,
so no future information leaks into the features.
"""

import numpy as np
import pandas as pd

from generate_data import NON_DELIVERY_DISPUTE_RATE, TRADING_DAYS, RUNOFF_DAYS

SNAPSHOT_DAYS = list(range(60, 271, 30))   # 60, 90, ..., 270
HORIZON_DAYS = 90
NORMAL_PIPELINE_DAYS = 30                  # sales still inside the normal dispute window, beyond delivery

FEATURES_NUMERIC = [
    "tenure_months_now", "delivery_lag_days", "avg_ticket",
    "daily_volume_30d", "volume_growth_7d_vs_90d", "volume_growth_30d_vs_90d",
    "cb_ratio_90d", "cb_count_90d",
]
FEATURES_CATEGORICAL = ["segment"]


def _matrices(merchants, daily, chargebacks):
    """Merchant x day arrays: sales volume by trading day, chargeback amount/count by arrival day."""
    idx = pd.Index(merchants["merchant_id"])
    n = len(idx)
    vol = np.zeros((n, TRADING_DAYS))
    rows = idx.get_indexer(daily["merchant_id"])
    np.add.at(vol, (rows, daily["day"].to_numpy()), daily["volume"].to_numpy())

    horizon = TRADING_DAYS + RUNOFF_DAYS
    cb_amt = np.zeros((n, horizon))
    cb_cnt = np.zeros((n, horizon))
    rows = idx.get_indexer(chargebacks["merchant_id"])
    arr = chargebacks["arrival_day"].to_numpy()
    np.add.at(cb_amt, (rows, arr), chargebacks["amount"].to_numpy())
    np.add.at(cb_cnt, (rows, arr), 1)
    return vol, cb_amt, cb_cnt


def realized_loss(merchants, chargebacks):
    """Loss the platform absorbs: every chargeback arriving on or after the failure day.

    Before failure the merchant pays its own chargebacks; after failure there is
    nobody to recover from (before any reserve is held).
    """
    fail_day = merchants.set_index("merchant_id")["fail_day"]
    cb = chargebacks.join(fail_day, on="merchant_id")
    lost = cb[(cb["fail_day"] >= 0) & (cb["arrival_day"] >= cb["fail_day"])]
    return lost.groupby("merchant_id")["amount"].sum().reindex(merchants["merchant_id"], fill_value=0.0)


def build_snapshots(merchants, daily, chargebacks):
    vol, cb_amt, cb_cnt = _matrices(merchants, daily, chargebacks)
    loss = realized_loss(merchants, chargebacks).to_numpy()
    fail_day = merchants["fail_day"].to_numpy()
    failed = merchants["failed"].to_numpy()

    parts = []
    for t in SNAPSHOT_DAYS:
        alive = ~failed | (fail_day >= t)
        window = min(90, t)
        v90 = vol[:, t - window:t].mean(axis=1)
        v30 = vol[:, t - 30:t].mean(axis=1)
        v7 = vol[:, t - 7:t].mean(axis=1)
        cb90_amt = cb_amt[:, t - window:t].sum(axis=1)

        snap = pd.DataFrame({
            "merchant_id": merchants["merchant_id"],
            "snapshot_day": t,
            "segment": merchants["segment"],
            "tenure_months_now": merchants["tenure_months"] + t / 30,
            "delivery_lag_days": merchants["delivery_lag_days"],
            "avg_ticket": merchants["avg_ticket"],
            "daily_volume_30d": v30,
            "volume_growth_7d_vs_90d": np.divide(v7, v90, out=np.ones_like(v7), where=v90 > 0),
            "volume_growth_30d_vs_90d": np.divide(v30, v90, out=np.ones_like(v30), where=v90 > 0),
            "cb_ratio_90d": np.divide(cb90_amt, v90 * window, out=np.zeros_like(v90), where=v90 > 0),
            "cb_count_90d": cb_cnt[:, t - window:t].sum(axis=1),
            "target": failed & (fail_day >= t) & (fail_day < t + HORIZON_DAYS),
            "realized_loss": loss,
        })
        parts.append(snap[alive])

    panel = pd.concat(parts, ignore_index=True)
    panel["target"] = panel["target"].astype(int)
    # Realized loss only counts when the failure falls inside this snapshot's horizon.
    panel["realized_loss"] = np.where(panel["target"] == 1, panel["realized_loss"], 0.0)
    panel["ead"] = exposure_at_default(panel)
    return panel


def exposure_at_default(panel):
    """Estimated loss if the merchant failed today, using the current 30-day sales run-rate.

    = undelivered sales x share that will be disputed
    + sales still inside the normal dispute window x observed chargeback ratio
    """
    undelivered = panel["daily_volume_30d"] * panel["delivery_lag_days"] * NON_DELIVERY_DISPUTE_RATE
    pipeline_days = panel["delivery_lag_days"] + NORMAL_PIPELINE_DAYS
    normal_tail = panel["daily_volume_30d"] * pipeline_days * panel["cb_ratio_90d"]
    return undelivered + normal_tail
