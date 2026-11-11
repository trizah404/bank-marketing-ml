# Bank Marketing: Term-Deposit Subscription Predictor

An end-to-end traditional ML system. It predicts whether a bank customer will subscribe to a term deposit **before** the bank calls them, so the call centre can prioritise who to call.

**Dataset:** UCI Bank Marketing (`bank-additional-full.csv`). 41,188 calls, 20 input columns, 11.3% "yes".

## Run order

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data       # 1. clean + stratified 80/20 split  -> data/processed/
python -m src.eda        # 2. EDA on TRAIN only               -> reports/figures/01-04, eda_summary.json
python -m src.train      # 3. leakage test, CV comparison, tuning -> reports/cv_results.json  (slowest step)
python -m src.evaluate   # 4. threshold, ONE test evaluation, temporal check -> models/model.joblib
python -m src.predict    # 5. score one example customer
```

Always run from the repo root, using `python -m src.<file>`, so the `src.` imports resolve.

## Repo map: read the code in this order

| File | Phase | Concept from the course |
|---|---|---|
| `src/config.py` | 0 | Every setting in one place: leaky columns, seed, business-cost assumptions |
| `src/data.py` | 1–2 | Remove duplicates and leaky columns → **split first** → lock the test set away |
| `src/eda.py` | 1 | EDA on the training split only; the `duration` leakage audit |
| `src/features.py` | 3–4 | Feature engineering + `ColumnTransformer` inside a `Pipeline` (no leakage inside CV) |
| `src/train.py` | 5–7 | Dummy baseline, stratified 5-fold CV, train-vs-CV gap (bias/variance), `RandomizedSearchCV` |
| `src/evaluate.py` | 8 | Threshold chosen from out-of-fold training predictions, test set used once, PR curve, permutation importance |
| `src/predict.py` | 9 | Load the bundle and score raw input (the API will wrap this) |

## Key decisions (the "why")

1. **`duration` is dropped.** It is the call length, which is only known after the call has ended. `train.py` measures exactly how much it inflates PR-AUC.
2. **The metric is PR-AUC, not accuracy.** Always predicting "no" already scores 88.7% accuracy and catches nobody.
3. **"unknown" is kept as its own category.** Missingness carries signal here.
4. **`pdays = 999` becomes a flag plus NaN.** 999 is a code meaning "never contacted", not a number of days.
5. **Imbalance is handled with the threshold, not resampling.** Probabilities stay calibrated, and the threshold maximises expected profit under assumed costs (€5 per call, €50 per subscriber; both are in `config.py`).
6. **Temporal robustness check.** The data is time-ordered, so `evaluate.py` also trains on the past 80% and tests on the future 20%, to show how the model holds up under drift.

## Results

_Fill this in from `reports/cv_results.json` and `reports/test_results.json` after running._

## Deployment

To do: FastAPI `/predict` → Docker → hosting.
