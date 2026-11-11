"""
Phase 1: Exploratory Data Analysis.

Done on the TRAINING split only. Even "just looking" at the test set can
influence your decisions (which features to build, which outliers to drop),
which is a quiet form of leakage.

Outputs: reports/eda_summary.json + 4 figures in reports/figures/.
"""
import json

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import (FIG_DIR, REPORT_DIR, TARGET, TEST_SIZE, RANDOM_STATE)
from src.data import load_raw, clean
from src.plotting import style, rate_bars


def training_rows_with_duration() -> pd.DataFrame:
    """Rebuild the training rows *including* `duration`, only so we can audit
    the leakage. Same dedupe + same seed => exactly the same rows as data.py."""
    raw = load_raw().drop_duplicates().reset_index(drop=True)
    cleaned = clean(load_raw())
    train_idx, _ = train_test_split(cleaned.index, test_size=TEST_SIZE,
                                    stratify=cleaned[TARGET],
                                    random_state=RANDOM_STATE)
    df = raw.loc[train_idx].copy()
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def main():
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = training_rows_with_duration()
    base = df[TARGET].mean()

    summary = {
        "train_rows": len(df),
        "yes_rate": round(base, 4),
        "dummy_accuracy_if_always_no": round(1 - base, 4),
        "unknown_counts": {c: int((df[c] == "unknown").sum())
                           for c in df.select_dtypes(exclude="number").columns
                           if (df[c] == "unknown").any()},
        "pdays_999_share": round((df["pdays"] == 999).mean(), 4),
        "default_yes_count": int((df["default"] == "yes").sum()),
    }

    # 1) LEAKAGE AUDIT: duration vs outcome
    q = pd.qcut(df["duration"], 5)
    r = df.groupby(q, observed=True)[TARGET].mean()
    labels = [f"{int(i.left)+1}-{int(i.right)}s" for i in r.index]
    labels[0] = f"0-{int(r.index[0].right)}s"
    fig, ax = plt.subplots(figsize=(7, 4))
    rate_bars(ax, labels, r.values,
              "Call duration predicts the outcome, but is only known after the call",
              "call duration (quintiles)", base)
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_duration_leakage.png"); plt.close(fig)
    summary["yes_rate_by_duration_quintile"] = dict(zip(labels, r.round(3).tolist()))

    # 2) Age is U-shaped -> justifies age bands
    bands = pd.cut(df["age"], [0, 25, 35, 50, 60, 200],
                   labels=["<25", "25-34", "35-49", "50-59", "60+"])
    r = df.groupby(bands, observed=True)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    rate_bars(ax, list(r.index.astype(str)), r.values,
              "Young and older customers subscribe far more often", "age band", base)
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_age_u_shape.png"); plt.close(fig)
    summary["yes_rate_by_age_band"] = r.round(3).to_dict()

    # 3) Previous campaign outcome
    r = df.groupby("poutcome")[TARGET].mean().reindex(["nonexistent", "failure", "success"])
    fig, ax = plt.subplots(figsize=(7, 4))
    rate_bars(ax, ["never contacted", "previous: failure", "previous: success"], r.values,
              "A past success is the strongest single signal", "previous campaign outcome", base)
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_poutcome.png"); plt.close(fig)
    summary["yes_rate_by_poutcome"] = r.round(3).to_dict()

    # 4) Month (also a hint that the data is time-ordered)
    order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    r = df.groupby("month")[TARGET].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    rate_bars(ax, order, r.values, "Subscription rate swings strongly by month",
              "month of last contact", base)
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_month.png"); plt.close(fig)
    summary["yes_rate_by_month"] = r.round(3).to_dict()

    (REPORT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
