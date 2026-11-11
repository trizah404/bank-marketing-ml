"""
Phases 5-7: baselines -> fair model comparison -> hyperparameter tuning.

Everything here uses ONLY the training split, via stratified 5-fold CV.
The test set is not loaded in this file at all - that's the guarantee.

Primary metric: PR-AUC (average precision).
  * Accuracy is useless here: "always say no" already scores 88.7%.
  * PR-AUC measures how well the model RANKS the rare 'yes' customers above
    the 'no' customers, across all thresholds. Random guessing scores ~0.11
    (the yes rate), so that's the floor to beat.
"""
import json
import time

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate)
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_FOLDS, MODEL_DIR, RANDOM_STATE, REPORT_DIR, TARGET
from src.data import load_split
from src.features import make_pipeline

# Same folds for every model -> an apples-to-apples comparison.
CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1@0.5": "f1"}


def candidate_models():
    """Four model families, each with a reason to be here."""
    return {
        # sanity floor: always predicts the majority class
        "dummy (always 'no')": DummyClassifier(strategy="most_frequent"),
        # Lecture 6: linear, interpretable, well-calibrated probabilities
        "logistic regression": LogisticRegression(max_iter=2000, solver="liblinear"),
        # unrestricted tree: expected to OVERFIT (watch the train/CV gap)
        "decision tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        # many trees averaged -> lower variance than one tree
        "random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        # trees built sequentially, each fixing the last one's errors
        "gradient boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


# Hyperparameter search spaces (prefix `model__` = the step name in the pipeline)
SEARCH_SPACES = {
    "logistic regression": {
        "model__C": loguniform(1e-3, 1e2),          # inverse regularisation strength
        "model__penalty": ["l1", "l2"],             # Lasso vs Ridge
    },
    "decision tree": {
        "model__max_depth": randint(2, 15),
        "model__min_samples_leaf": randint(5, 200),
    },
    "random forest": {
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_leaf": randint(1, 50),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "gradient boosting": {
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_iter": randint(100, 600),
        "model__max_leaf_nodes": randint(8, 64),
        "model__min_samples_leaf": randint(20, 300),
        "model__l2_regularization": loguniform(1e-3, 10),
    },
}


def compare_models(X, y):
    """Phase 5-6: every model, same folds, same metrics, mean +/- std."""
    results = {}
    for name, model in candidate_models().items():
        t = time.time()
        cv = cross_validate(make_pipeline(model), X, y, cv=CV, scoring=SCORING,
                            return_train_score=True, n_jobs=1)
        results[name] = {
            "cv_pr_auc_mean": cv["test_pr_auc"].mean(),
            "cv_pr_auc_std": cv["test_pr_auc"].std(),
            "train_pr_auc_mean": cv["train_pr_auc"].mean(),
            "cv_roc_auc_mean": cv["test_roc_auc"].mean(),
            "cv_f1_at_0.5_mean": cv["test_f1@0.5"].mean(),
            "seconds": round(time.time() - t, 1),
        }
        r = results[name]
        print(f"{name:22s} PR-AUC {r['cv_pr_auc_mean']:.3f} ± {r['cv_pr_auc_std']:.3f}"
              f"  (train {r['train_pr_auc_mean']:.3f})  ROC-AUC {r['cv_roc_auc_mean']:.3f}"
              f"  F1@0.5 {r['cv_f1_at_0.5_mean']:.3f}")
    return results


def tune(name, X, y, n_iter=30):
    """Phase 7: RandomizedSearchCV. Random search > grid search when only a few
    hyperparameters really matter (Bergstra & Bengio, 2012)."""
    search = RandomizedSearchCV(
        make_pipeline(candidate_models()[name]), SEARCH_SPACES[name],
        n_iter=n_iter, scoring="average_precision", cv=CV,
        random_state=RANDOM_STATE, n_jobs=-1, return_train_score=True)
    search.fit(X, y)
    i = search.best_index_
    out = {
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "cv_pr_auc_mean": search.cv_results_["mean_test_score"][i],
        "cv_pr_auc_std": search.cv_results_["std_test_score"][i],
        "train_pr_auc_mean": search.cv_results_["mean_train_score"][i],
    }
    print(f"tuned {name:16s} PR-AUC {out['cv_pr_auc_mean']:.3f} ± {out['cv_pr_auc_std']:.3f}"
          f"  params {out['best_params']}")
    return search.best_estimator_, out


def leakage_experiment():
    """
    How much does `duration` inflate the score? Same logistic regression,
    same folds, with and without the leaky column. This number is the
    evidence for dropping it.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src.eda import training_rows_with_duration
    from src.features import build_preprocessor

    df = training_rows_with_duration()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    prep = build_preprocessor()
    with_dur = Pipeline([
        ("both", ColumnTransformer([
            ("rest", prep, X.columns.drop("duration").tolist()),
            ("dur", StandardScaler(), ["duration"]),
        ])),
        ("model", LogisticRegression(max_iter=2000, solver="liblinear")),
    ])
    without = make_pipeline(LogisticRegression(max_iter=2000, solver="liblinear"))
    a = cross_validate(with_dur, X, y, cv=CV, scoring="average_precision")["test_score"].mean()
    b = cross_validate(without, X.drop(columns="duration"), y, cv=CV,
                       scoring="average_precision")["test_score"].mean()
    print(f"LEAKAGE: PR-AUC with duration {a:.3f}  vs without {b:.3f}")
    return {"pr_auc_with_duration": a, "pr_auc_without_duration": b}


def main():
    X, y = load_split("train")
    print(f"training rows: {len(X):,}\n\n== Leakage experiment ==")
    leak = leakage_experiment()

    print("\n== Phase 5-6: baseline comparison (default settings) ==")
    baseline = compare_models(X, y)

    # Tune the top-2 real models by CV PR-AUC (chosen from data, not by hand)
    ranked = sorted((n for n in baseline if n in SEARCH_SPACES),
                    key=lambda n: baseline[n]["cv_pr_auc_mean"], reverse=True)
    top2 = ranked[:2]
    print(f"\n== Phase 7: tuning top 2: {top2} ==")
    tuned, best_name, best_est = {}, None, None
    for name in top2:
        est, info = tune(name, X, y)
        tuned[name] = info
        if best_name is None or info["cv_pr_auc_mean"] > tuned[best_name]["cv_pr_auc_mean"]:
            best_name, best_est = name, est

    print(f"\nSELECTED: {best_name}")
    MODEL_DIR.mkdir(exist_ok=True)
    # best_estimator_ is already refit on ALL training data by RandomizedSearchCV
    joblib.dump({"pipeline": best_est, "name": best_name},
                MODEL_DIR / "selected_unthresholded.joblib")

    to_float = lambda d: {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                          for k, v in d.items()}
    (REPORT_DIR / "cv_results.json").write_text(json.dumps({
        "leakage_experiment": to_float(leak),
        "baseline_comparison": {k: to_float(v) for k, v in baseline.items()},
        "tuned": {k: to_float(v) for k, v in tuned.items()},
        "selected_model": best_name,
    }, indent=2))


if __name__ == "__main__":
    main()
