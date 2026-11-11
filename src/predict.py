"""
Load the saved bundle and score ONE raw customer.
This is the function your API (FastAPI) will call during deployment.

    python -m src.predict
"""
import joblib
import pandas as pd

from src.config import MODEL_DIR

_bundle = None


def load_bundle():
    global _bundle
    if _bundle is None:                     # load once, reuse
        _bundle = joblib.load(MODEL_DIR / "model.joblib")
    return _bundle


def predict_one(customer: dict) -> dict:
    b = load_bundle()
    if "duration" in customer:
        raise ValueError("'duration' is not allowed: it is only known after the call.")
    missing = set(b["input_columns"]) - set(customer)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    X = pd.DataFrame([customer])[b["input_columns"]]
    p = float(b["pipeline"].predict_proba(X)[0, 1])
    return {"probability": round(p, 4), "call": p >= b["threshold"],
            "threshold": b["threshold"], "model": b["model_name"]}


if __name__ == "__main__":
    example = {
        "age": 62, "job": "retired", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
        "month": "oct", "day_of_week": "tue", "campaign": 1, "pdays": 6, "previous": 1,
        "poutcome": "success", "emp.var.rate": -3.4, "cons.price.idx": 92.431,
        "cons.conf.idx": -26.9, "euribor3m": 0.754, "nr.employed": 5017.5,
    }
    print(predict_one(example))
