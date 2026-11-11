"""
Central configuration. Every other file imports from here, so a value is
changed in ONE place only.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "bank-additional-full.csv"
SPLIT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42          # fixed seed -> every run gives identical splits/results
TEST_SIZE = 0.20           # 80/20 holdout (Lecture 7)
CV_FOLDS = 5               # stratified 5-fold cross-validation

TARGET = "y"

# ---- LEAKAGE GUARD -------------------------------------------------------
# `duration` = length of the sales call in seconds. It is only known AFTER the
# call ends, i.e. after the outcome is already decided. Using it would be the
# "feature that secretly contains the answer" leakage from Lecture 7.
LEAKY_COLUMNS = ["duration"]

# Raw input columns the model is allowed to see (what we know BEFORE calling).
NUMERIC_RAW = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_RAW = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
INPUT_COLUMNS = NUMERIC_RAW + CATEGORICAL_RAW

# ---- BUSINESS COST ASSUMPTIONS (used to pick the decision threshold) -----
# These are ASSUMPTIONS for illustration; a real bank would supply them.
COST_PER_CALL = 5.0        # agent time per call (EUR)
VALUE_PER_SUBSCRIBER = 50.0  # profit from one term-deposit subscription (EUR)
