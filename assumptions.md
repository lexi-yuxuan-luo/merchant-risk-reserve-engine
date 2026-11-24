# Assumptions

Every number used to generate the synthetic data is listed here. All data in this project is synthetic.

**Source types**
- **Illustrative**: a value chosen to make the simulation behave plausibly. It is *not* an industry benchmark and should not be quoted as one.
- **Public**: based on published, publicly available information. The reference is given.

No value in this project is taken from, or calibrated to, any employer's data, rules or thresholds.

---

## 1. Simulation scope

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Number of merchants | 5,000 | Illustrative | Gives roughly 250 failures, enough to train and validate a tiering model; runs in seconds |
| Trading window | 365 days | Illustrative | One year of activity |
| Run-off window | +180 days | Illustrative | Chargebacks keep arriving after the trading window ends; the extra 180 days lets late ones land |
| Currency | USD | Illustrative | Single currency keeps the model simple |
| Random seed | 42 | — | Results are reproducible |

## 2. Merchant segments

The key driver of exposure is **delivery lag**: the number of days between the customer paying and the merchant delivering the goods or service. While a purchase is undelivered, the platform is exposed. If the merchant fails, those customers dispute the charge and the platform pays.

| Segment | Share | Avg ticket (USD) | Delivery lag (days) | Base chargeback rate | Annual failure hazard |
|---|---|---|---|---|---|
| retail_physical | 30% | 40 | 3 | 0.30% | 3% |
| digital_goods | 15% | 15 | 0 | 0.60% | 5% |
| subscription | 15% | 20 | 30 | 0.50% | 6% |
| travel | 10% | 300 | 60 | 0.40% | 5% |
| events_ticketing | 10% | 80 | 45 | 0.40% | 6% |
| food_delivery | 10% | 25 | 0 | 0.20% | 4% |
| electronics | 10% | 250 | 5 | 0.50% | 4% |

All values: **Illustrative.** The ordering reflects the general idea that future-delivery businesses (travel, ticketing, subscriptions) carry more exposure per dollar than instant-delivery ones. The exact numbers are not benchmarks.

Base chargeback rates are deliberately set below the levels at which card networks typically put merchants into monitoring programs. Check current Visa and Mastercard program documentation before quoting any threshold.

## 3. Merchant-level variation

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Tenure (months) | Lognormal, median ≈ 18, clipped to 1–120 | Illustrative | Most merchants are young; a long tail are established |
| Monthly volume (USD) | Lognormal, median ≈ 20,000 | Illustrative | SMB-sized merchants with a heavy tail |
| Risk multiplier | Lognormal(0, 0.5) | Illustrative | Unobserved merchant quality; scales both chargeback rate and failure hazard |
| New-merchant uplift | ×1.5 risk if tenure < 6 months | Illustrative | Young merchants have less track record and fail more often |
| Daily volume noise | Lognormal(0, 0.2) on average ticket | Illustrative | Day-to-day variation |

## 4. Merchant failure

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Failure timing | Constant daily hazard = annual hazard × risk multiplier ÷ 365 | Illustrative | Simplest model that ties failure to observable risk |
| Pre-failure volume ramp | 50% of failing merchants triple volume over the 30 days before failure | Illustrative | Represents "bust-out" behaviour: taking as much money as possible before disappearing. Gives the model a learnable warning signal |
| Activity after failure | No new transactions | Illustrative | |

## 5. Chargebacks

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Normal disputes | Binomial(transactions, base rate × risk multiplier) | Illustrative | |
| Undelivered at failure | 25% of customers whose purchase was not yet delivered dispute | Illustrative | Not every customer disputes; some don't notice or get refunded elsewhere |
| Normal dispute arrival | Delivery date + Gamma(mean ≈ 30 days), capped at 120 days | Illustrative; cap informed by public card network dispute timeframes | Disputes arrive weeks after purchase. Card networks generally allow cardholders a limited window to dispute, often cited as up to 120 days; verify against current network rules before quoting |
| Non-delivery dispute arrival | Failure date + Gamma(mean ≈ 20 days) | Illustrative | Customers notice soon after the merchant stops operating |

## 6. Not modelled (yet)

- Refunds (separate from chargebacks)
- Chargeback representment (merchant winning a dispute back)
- Fraud by third parties using stolen cards
- Multiple currencies, FX, cross-border fees

These are reasonable extensions and are listed so the scope is explicit.
