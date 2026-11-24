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

**Sense check against public benchmarks.** Base chargeback rates of 0.2%–0.6% sit within publicly reported industry averages and well below card network monitoring thresholds:

- Industry averages: around 0.26% of transactions across industries (Sift, Q3 2025, as reported by [chargeback.io](https://www.chargeback.io/de/blog/average-e-commerce-chargeback-rate)); 0.47% for US e-commerce and 0.52% for UK e-commerce ([Signifyd](https://www.signifyd.com/?p=53082)).
- Mastercard Excessive Chargeback Merchant program: 1.5% or more, with at least 100 chargebacks a month ([Chargebacks911](https://chargebacks911.com/mastercard-chargeback-threshold/)).
- Visa Acquirer Monitoring Program (VAMP), merchant "Excessive" threshold: 2.2%, reduced to 1.5% in AP, Canada, EU and US from 1 April 2026. The VAMP ratio counts fraud reports and disputes together ([Visa VAMP fact sheet](https://corporate.visa.com/content/dam/VCOM/corporate/visa-perspectives/security-and-trust/documents/visa-acquirer-monitoring-program-fact-sheet-2025.pdf)).

After the risk multiplier, a minority of high-risk merchants exceed 1%, which is the intended behaviour. Program rules change; check the networks' current documentation before quoting a threshold.

## 3. Merchant-level variation

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Tenure (months) | Lognormal, median ≈ 18, clipped to 1–120 | Illustrative | Most merchants are young; a long tail are established |
| Monthly volume (USD) | Lognormal, median ≈ 20,000 | Illustrative | SMB-sized merchants with a heavy tail |
| Risk multiplier | Lognormal with σ = 1.0 and mean fixed at 1 | Illustrative | Unobserved merchant quality; scales both chargeback rate and failure hazard. Mean of 1 means segment baselines describe the average merchant. A wide spread reflects that merchant risk is concentrated: most merchants are low-risk and a few are very risky. An earlier version used σ = 0.5, which made failures close to random: even a model that knew each merchant's true hazard only reached an out-of-time AUC of about 0.63 |
| New-merchant uplift | ×1.5 risk if tenure < 6 months | Illustrative | Young merchants have less track record and fail more often |
| Daily volume noise | Lognormal(0, 0.2) on average ticket | Illustrative | Day-to-day variation |

## 4. Merchant failure

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Failure timing | Constant daily hazard = annual hazard × risk multiplier ÷ 365 | Illustrative | Simplest model that ties failure to observable risk |
| Pre-failure volume ramp | 50% of failing merchants ramp volume linearly from 1x to 3x over the 30 days before failure | Illustrative | Represents "bust-out" behaviour: taking as much money as possible before disappearing. A gradual ramp is more realistic than a sudden jump and makes early detection harder. Gives the model a learnable warning signal |
| Pre-failure distress | The other 50% of failing merchants see volume decline linearly to 40% of normal over the 60 days before failure | Illustrative | Represents a business slowly losing customers before it closes. Distressed merchants tend to cause smaller losses than bust-outs, because they have fewer undelivered sales when they fail |
| Activity after failure | No new transactions | Illustrative | |

## 5. Chargebacks

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Normal disputes | Binomial(transactions, base rate × risk multiplier) | Illustrative | |
| Undelivered at failure | 25% of customers whose purchase was not yet delivered dispute | Illustrative; no public benchmark found | Not every customer disputes; some don't notice or get refunded elsewhere. With real data, calibrate from chargebacks on historical merchant failures |
| Normal dispute arrival | Delivery date + Gamma(mean ≈ 30 days), capped at 120 days | Illustrative; cap informed by public card network dispute timeframes | Disputes arrive weeks after purchase. Card networks generally allow cardholders a limited window to dispute, often cited as up to 120 days; verify against current network rules before quoting |
| Non-delivery dispute arrival | Failure date + Gamma(mean ≈ 20 days) | Illustrative | Customers notice soon after the merchant stops operating |

## 6. Risk tiering

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Snapshots | Every 30 days, from day 60 to day 270 | Design choice | Each snapshot uses only transactions before that day and chargebacks that have already arrived, so no future information leaks into features |
| Prediction horizon | 90 days | Design choice | Long enough to act on (raise a reserve, delay settlement), short enough to capture the pre-failure warning signals |
| Train / test split | Train on snapshots 60–150; test out-of-time on 240 and 270; 180 and 210 are left out | Design choice | Standing on day 240, when the model would go live, only snapshots up to day 150 have a fully observed 90-day outcome. Using 180 or 210 would train on outcomes not yet known, the same mistake as modelling on immature vintages |
| Primary model | WoE scorecard (logistic regression) sets tiers | Design choice | Transparent and widely used; easy to explain to merchants and auditors |
| Early-warning overlay | Gradient boosting flags tier A–C merchants whose boosted PD is above its own tier-E cut-off | Design choice | Uses the more accurate but less transparent model to catch risk the scorecard misses, by sending cases to review rather than changing tiers directly |
| Tier shares | A–E = 40%, 30%, 15%, 10%, 5% of the training population | Illustrative | Cut-offs are fixed on training data and applied unchanged to the test period |
| LGD | 100% | Illustrative | Before any reserve is held, nothing is recovered once a merchant has gone. Reserves are introduced in the policy engine |
| Exposure (EAD) | 30-day sales run-rate × (delivery lag × 25% + (delivery lag + 30 days) × observed chargeback ratio), scaled by a factor fitted on training-period failures | Illustrative | Undelivered sales that would be disputed, plus sales still inside the normal dispute window. The scaling factor plays the role of a credit conversion factor. In practice the 25% would be estimated from past merchant failures |

## 7. Reserve policy engine

| Parameter | Value | Type | Rationale |
|---|---|---|---|
| Evaluation window | Days 240–364, merchants alive on day 240 | Design choice | The out-of-time period: neither risk model trained on it |
| Tier refresh | Tiers read at day 240, refreshed at day 270 | Design choice | Mirrors a monthly re-tiering cycle |
| Control levels | None / Light / Medium / Heavy / Very heavy = reserve 0/5/10/15/25%, held 0/90/90/120/180 days, extra settlement delay 0/0/2/5/7 days | Illustrative | A ladder from no controls to very restrictive. Real terms are set per merchant contract |
| Loss recovery | On the failure day, the platform uses all merchant funds it still holds (unsettled sales + reserve balance) to cover chargebacks | Simplification | Ignores legal and operational friction in applying held funds |
| Steady state | Each merchant has been under its tier's controls long enough for the reserve to be full | Simplification | Avoids modelling the reserve build-up period |
| Net take rate | 0.5% of volume | Illustrative | Platform revenue after interchange and network fees. No public benchmark used |
| Churn | Probability = 1 − exp(−0.4 × funds locked ÷ monthly volume); e.g. 5.8% at Light, 50% at Very heavy | Illustrative; no public benchmark found | Merchants with more cash tied up are more likely to move to another provider. Tested at 0.2 and 0.8 in the report |
| Churn and losses | Churn reduces revenue only; merchants are assumed to stay long enough to fail | Conservative simplification | In practice some risky merchants would leave first, lowering losses |
| Revenue at stake | Merchant's 30-day run-rate at day 240 × window length × take rate | Design choice | Uses only information available on day 240 |
| Expected-loss tiers | Ranked by expected loss ÷ daily volume, cut at 40/30/15/10/5% on training snapshots | Design choice | Reserves are a share of volume, so expected loss per $ of volume is the relevant risk. It combines PD with exposure, which depends heavily on delivery lag |

## 8. Not modelled (yet)

- Refunds (separate from chargebacks)
- Chargeback representment (merchant winning a dispute back)
- Fraud by third parties using stolen cards
- Multiple currencies, FX, cross-border fees

These are reasonable extensions and are listed so the scope is explicit.
