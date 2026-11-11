"""
Phase 3-4: feature engineering + preprocessing, packaged as ONE sklearn Pipeline.

Why a Pipeline?
  * Inside cross-validation, sklearn re-fits the whole pipeline on each
    training fold. The scaler/encoder therefore never sees the validation
    fold -> no leakage, automatically.
  * The saved model file contains preprocessing + model together, so the API
    can accept RAW customer data. (Mistake #5: deploying a model without its
    preprocessing.)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds domain-driven features. It has nothing to LEARN (no fit statistics),
    so it can't leak, but it lives in the pipeline so the API applies it too.

    New features and the reason for each:
      prev_contacted : pdays == 999 means "never contacted before". 999 is a
                       code, not a number of days - a linear model would read
                       it as "very long ago". We pull it out into a 0/1 flag.
      pdays          : 999 replaced by NaN, then imputed (median) downstream,
                       so the real day counts keep their meaning.
      campaign_log   : calls-this-campaign is heavily right-skewed (1..56).
                       log1p compresses the tail so a few extreme customers
                       don't dominate the weights.
      age_band       : subscription rate vs age is U-shaped (students and
                       retirees subscribe more). A straight-line model can't
                       bend (Part 1 of Lecture 7: bias!), so we give it bands.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["prev_contacted"] = (X["pdays"] != 999).astype(int)
        X["pdays"] = X["pdays"].replace(999, np.nan)
        X["campaign_log"] = np.log1p(X["campaign"])
        X["age_band"] = pd.cut(
            X["age"], bins=[0, 25, 35, 50, 60, 200],
            labels=["<25", "25-34", "35-49", "50-59", "60+"]
        ).astype(str)
        return X


NUMERIC = [
    "age", "campaign_log", "pdays", "previous", "prev_contacted",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome", "age_band",
]


def build_preprocessor() -> Pipeline:
    """
    numeric     -> impute median (only pdays has gaps) -> standardise
    categorical -> one-hot encode. 'unknown' is KEPT as its own category:
                   in this data "unknown" is informative (e.g. default=unknown
                   customers subscribe less), so deleting those rows would
                   throw away signal.
    handle_unknown='ignore': if the API receives a category never seen in
    training, it becomes all-zeros instead of crashing.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    columns = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])
    return Pipeline([("engineer", FeatureEngineer()), ("columns", columns)])


def make_pipeline(model) -> Pipeline:
    """preprocessing + any sklearn model = one object to fit/save/serve."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])
