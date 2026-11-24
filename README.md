# Merchant Risk Tiering & Reserve Policy Engine

> Work in progress.

Payment platforms settle funds to merchants before chargebacks and refunds arrive, sometimes months later. If a merchant fails in between, the platform absorbs the loss.

This project builds a merchant risk tiering model and a policy engine that sets rolling reserves, settlement delays and transaction limits per tier. It then quantifies the trade-off between **loss coverage** and **merchant attrition**.

All data is synthetic; every assumption is documented in [`assumptions.md`](assumptions.md).

## Quick start

```bash
pip install -r requirements.txt
python src/generate_data.py   # writes data/merchants.csv, data/daily.csv, data/chargebacks.csv
```

## Roadmap

- [x] Synthetic data generator
- [ ] Merchant risk tiering (scorecard + gradient boosting) and expected loss
- [ ] Reserve / settlement-delay policy engine and loss–attrition frontier
- [ ] Stress testing
- [ ] Streamlit demo
