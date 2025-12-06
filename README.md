# Bank Fraud Synthetic Dataset for Train-Test Split Experimentation Project

This project contains the code to explore train-test splits with different ratios and the subsequent impact on a model's accuracy with imbalanced and balanced datasets.

A synthetic dataset generator is created for a simple bank transactions fraud detection problem. It produces two CSVs with 1,000 rows each by default:

- data/processed/bank_transactions_balanced.csv (50% fraud)
- data/processed/bank_transactions_imbalanced.csv (5% fraud by default)

## Notebook usage (recommended)

```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

1. Open `notebooks/generate_synthetic_bank_fraud.ipynb` in your IDE (e.g., VS Code) and run all cells.
2. Optionally edit the parameters in the first code cell (`outdir`, `rows`, `seed`, `imbalanced_rate`).
3. The CSVs will be saved to `data/processed/` by default.

## Schema

Columns in the generated CSVs:

- transaction_id: String transaction ID (e.g., `T0000123`).
- customer_id: Integer customer identifier.
- device_id: String device identifier used for the transaction.
- transaction_datetime: ISO timestamp string of the transaction.
- amount: Transaction amount in the local currency.
- merchant_category: Categorical (grocery, electronics, restaurants, gas, travel, online_services, utilities, entertainment, healthcare).
- transaction_type: Categorical (pos, online, atm, transfer).
- channel: Categorical (in_person, mobile, web, ivr).
- card_present: 0/1 whether the card was physically present.
- is_international: 0/1 whether the transaction is international.
- device_trust_score: 0.0–1.0 trust score for device.
- ip_risk_score: 0–100 risk score for IP.
- account_age_days: Age of the account in days.
- txn_count_24h: Number of transactions in last 24 hours.
- avg_amount_30d: Average amount over last 30 days.
- time_since_last_txn_sec: Seconds since last transaction.
- same_device_as_last: 0/1 whether same device as previous transaction.
- is_fraud: Target label (0 = not fraudulent, 1 = fraudulent).

## Notes

- Balanced dataset uses 50% fraud rate.
- Imbalanced dataset uses `--imbalanced_rate` (default 5%).
- Feature distributions differ by label to make the problem learnable but not trivial.
