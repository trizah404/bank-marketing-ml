"""
Phase 8: pick the decision threshold, then evaluate on the test set ONCE.

Order matters:
  1. Threshold is chosen from OUT-OF-FOLD predictions on the TRAINING set
     (cross_val_predict). Choosing it on the test set = tuning on test = leakage.
  2. Only then is the test set opened, scored once, and reported.
  3. Extra: a time-ordered robustness check (see temporal_check).

Saves the deployable bundle: models/model.joblib = {pipeline, threshold, ...}
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import cross_val_predict

from src.config import (COST_PER_CALL, FIG_DIR, INPUT_COLUMNS, MODEL_DIR,
                        REPORT_DIR, TARGET, VALUE_PER_SUBSCRIBER)
from src.data import clean, load_raw, load_split
from src.plotting import INK2, MUTED, SERIES, style
from src.train import CV


def profit(y_true, proba, t):
    """Each call costs COST_PER_CALL; each subscriber we call earns VALUE.
    We only call customers with probability >= t."""
    called = proba >= t
    tp = np.sum(called & (y_true == 1))
    return tp * VALUE_PER_SUBSCRIBER - called.sum() * COST_PER_CALL


def choose_threshold(pipe, X, y):
    """Scan thresholds on out-of-fold probabilities; keep the most profitable.
    Theory check: with calibrated probabilities, calling is worth it when
    p * VALUE > COST, i.e. p > COST/VALUE = 0.10."""
    oof = cross_val_predict(clone(pipe), X, y, cv=CV, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
    profits = np.array([profit(y.values, oof, t) for t in grid])
    f1s = np.array([f1_score(y, oof >= t) for t in grid])
    best_t = float(grid[profits.argmax()])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, profits / 1000, color=SERIES[0], lw=2)
    ax.axvline(best_t, color=MUTED, ls="--", lw=1)
    ax.text(best_t + 0.01, profits.max() / 1000, f"best threshold = {best_t:.2f}",
            color=INK2, va="top")
    ax.set_title("Expected campaign profit vs decision threshold (training, out-of-fold)")
    ax.set_xlabel("call the customer if P(subscribe) ≥ threshold")
    ax.set_ylabel("profit (thousand EUR)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_threshold_profit.png"); plt.close(fig)

    return best_t, {
        "profit_best_threshold": best_t,
        "f1_best_threshold": float(grid[f1s.argmax()]),
        "theoretical_threshold_cost_over_value": COST_PER_CALL / VALUE_PER_SUBSCRIBER,
        "oof_pr_auc": float(average_precision_score(y, oof)),
    }


def metrics_at(y, proba, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": t, "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "confusion": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "profit_eur": float(profit(np.asarray(y), proba, t)),
        "profit_if_call_everyone_eur": float(profit(np.asarray(y), proba, 0.0)),
        "calls_made": int(pred.sum()), "calls_if_everyone": int(len(y)),
    }


def plot_test(y, proba, t, cm):
    # PR curve
    p, r, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(r, p, color=SERIES[0], lw=2, label="model")
    ax.axhline(y.mean(), color=MUTED, ls="--", lw=1, label=f"random ({y.mean():.2f})")
    ax.set_title("Precision–recall curve (test set)")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_pr_curve_test.png"); plt.close(fig)

    # Confusion matrix (rows = ACTUAL, like sklearn)
    m = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(m, cmap="Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "#0b0b0b", fontsize=12)
    ax.set_xticks([0, 1], ["pred: no", "pred: yes"]); ax.set_yticks([0, 1], ["actual: no", "actual: yes"])
    ax.set_title(f"Confusion matrix @ threshold {t:.2f}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "07_confusion_test.png"); plt.close(fig)


def importance(pipe, X, y):
    """Permutation importance: shuffle one RAW column, measure how much PR-AUC
    drops. Model-agnostic, and works on the original column names."""
    res = permutation_importance(pipe, X, y, scoring="average_precision",
                                 n_repeats=5, random_state=0, n_jobs=-1)
    order = res.importances_mean.argsort()[::-1][:10]
    names = [X.columns[i] for i in order][::-1]
    vals = res.importances_mean[order][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, vals, color=SERIES[0], height=0.6, zorder=2)
    ax.set_title("Top 10 features (drop in PR-AUC when shuffled)")
    ax.set_xlabel("PR-AUC drop"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "08_feature_importance.png"); plt.close(fig)
    return {X.columns[i]: float(res.importances_mean[i]) for i in order}


def temporal_check(pipe):
    """
    The raw file is ordered by time (May 2008 -> Nov 2010) and the economic
    columns (euribor3m, nr.employed...) change over time. A random split lets
    the model train on months that sit *between* test months. In real life you
    always predict the FUTURE. So: train on the first 80% of rows, test on the
    last 20%. This is a robustness check only - it is NOT used to pick anything.
    """
    df = clean(load_raw())
    cut = int(len(df) * 0.8)
    past, future = df.iloc[:cut], df.iloc[cut:]
    m = clone(pipe).fit(past[INPUT_COLUMNS], past[TARGET])
    proba = m.predict_proba(future[INPUT_COLUMNS])[:, 1]
    return {"past_yes_rate": float(past[TARGET].mean()),
            "future_yes_rate": float(future[TARGET].mean()),
            "future_pr_auc": float(average_precision_score(future[TARGET], proba)),
            "future_roc_auc": float(roc_auc_score(future[TARGET], proba))}


def main():
    style()
    bundle = joblib.load(MODEL_DIR / "selected_unthresholded.joblib")
    pipe, name = bundle["pipeline"], bundle["name"]
    X_tr, y_tr = load_split("train")

    print("== Choosing threshold on training data (out-of-fold) ==")
    t, thr_info = choose_threshold(pipe, X_tr, y_tr)
    print(json.dumps(thr_info, indent=2))

    # ---------------- THE TEST SET IS OPENED HERE, ONCE ----------------
    X_te, y_te = load_split("test")
    proba = pipe.predict_proba(X_te)[:, 1]
    test = metrics_at(y_te, proba, t)
    plot_test(y_te, proba, t, test["confusion"])
    print("\n== TEST SET (reported once) ==\n" + json.dumps(test, indent=2))

    imp = importance(pipe, X_te, y_te)
    print("\n== Temporal robustness check ==")
    temporal = temporal_check(pipe)
    print(json.dumps(temporal, indent=2))

    joblib.dump({"pipeline": pipe, "threshold": t, "model_name": name,
                 "input_columns": list(X_tr.columns)}, MODEL_DIR / "model.joblib")
    (REPORT_DIR / "test_results.json").write_text(json.dumps({
        "model": name, "threshold_selection": thr_info, "test": test,
        "permutation_importance": imp, "temporal_check": temporal}, indent=2))
    print(f"\nSaved deployable bundle -> {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
