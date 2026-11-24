# Merchant Risk Tiering & Reserve Policy Engine

> Work in progress.

Payment platforms settle funds to merchants before chargebacks and refunds arrive, sometimes months later. If a merchant fails in between, the platform absorbs the loss.

This project builds a merchant risk tiering model and a policy engine that sets rolling reserves, settlement delays and transaction limits per tier. It then quantifies the trade-off between **loss coverage** and **merchant attrition**.

All data is synthetic; every assumption is documented in [`assumptions.md`](assumptions.md).

## Quick start

```bash
pip install -r requirements.txt
python src/generate_data.py   # writes data/merchants.csv, data/daily.csv, data/chargebacks.csv
python src/risk_model.py      # trains the models, writes data/scored_snapshots.csv and reports/risk_tiering.md
```

## Risk tiering

Merchants are scored at monthly point-in-time snapshots on their probability of failing within 90 days, using only information available at each snapshot. Models are validated out-of-time, on later snapshots they never saw.

- A transparent WoE scorecard (logistic regression) and a gradient boosting model are compared. Gradient boosting is the champion; the scorecard is kept as an explainable benchmark.
- Merchants are grouped into tiers A–E, with cut-offs fixed on the training period.
- Expected loss = PD × EAD × LGD. Exposure is driven by delivery lag and sales run-rate, so two merchants with the same PD can carry very different expected losses.

Full results: [`reports/risk_tiering.md`](reports/risk_tiering.md).

## Roadmap

- [x] Synthetic data generator
- [x] Merchant risk tiering (scorecard + gradient boosting) and expected loss
- [ ] Reserve / settlement-delay policy engine and loss–attrition frontier
- [ ] Stress testing
- [ ] Streamlit demo
