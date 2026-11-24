# Merchant Risk Tiering & Reserve Policy Engine

> Work in progress.

Payment platforms settle funds to merchants before chargebacks and refunds arrive, sometimes months later. If a merchant fails in between, the platform absorbs the loss.

This project builds a merchant risk tiering model and a policy engine that sets rolling reserves, settlement delays and transaction limits per tier. It then quantifies the trade-off between **loss coverage** and **merchant attrition**.

All data is synthetic; every assumption is documented in [`assumptions.md`](assumptions.md).

## Quick start

![pipeline](https://github.com/lexi-yuxuan-luo/merchant-risk-reserve-engine/actions/workflows/pipeline.yml/badge.svg)

```bash
pip install -r requirements.txt
python src/generate_data.py   # writes data/merchants.csv, data/daily.csv, data/chargebacks.csv
python src/risk_model.py      # trains the models, writes data/scored_snapshots.csv and reports/risk_tiering.md
python src/policy_engine.py   # evaluates reserve policies, writes reports/policy_engine.md and reports/frontier.png
```

## Risk tiering

Merchants are scored at monthly point-in-time snapshots on their probability of failing within 90 days, using only information available at each snapshot. Models are validated out-of-time, on later snapshots they never saw.

- Two models, two jobs: a transparent WoE scorecard (logistic regression) sets the tiers; a gradient boosting model acts as an early-warning overlay, flagging merchants the scorecard rates well but the boosted model sees as high risk.
- Models train only on snapshots whose outcome is fully observed by the time the model would go live.
- Merchants are grouped into tiers A–E, with cut-offs fixed on the training period.
- Expected loss = PD × EAD × LGD. Exposure is driven by delivery lag and sales run-rate, so two merchants with the same PD can carry very different expected losses.

Full results: [`reports/risk_tiering.md`](reports/risk_tiering.md).

## Reserve policy engine

Each tier gets a control level: a rolling reserve, an extra settlement delay, or both. Controls cut losses, because the platform can use the merchant funds it still holds when a merchant fails. They also cost revenue, because merchants with cash tied up are more likely to leave. Every policy is evaluated on the out-of-time period and plotted as residual loss against revenue lost.

![Loss vs attrition frontier](reports/frontier.png)

- **Tiering on expected loss beats tiering on PD.** Reserves are a share of volume, so the risk that matters is expected loss per $ of volume. It combines how likely a merchant is to fail with how much exposure its sales create, which depends heavily on delivery lag. A merchant with a moderate PD but a 60-day delivery lag can be riskier than one with a higher PD and instant delivery.
- **Any risk-based tiering beats one-size-fits-all.** The best uniform policy costs more than twice as much as the best expected-loss tiering.
- **The recommendation is robust.** Across churn assumptions, tiers A and B get no controls and tier C a light reserve; only the strength of controls on D and E changes.

Full results: [`reports/policy_engine.md`](reports/policy_engine.md).

## Roadmap

- [x] Synthetic data generator
- [x] Merchant risk tiering (scorecard + gradient boosting) and expected loss
- [x] Reserve / settlement-delay policy engine and loss–attrition frontier
- [ ] Stress testing
- [ ] Streamlit demo
