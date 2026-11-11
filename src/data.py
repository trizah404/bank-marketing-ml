"""
Phase 1-2: load, clean, and split the data.

The ORDER here is the whole lesson of Lecture 7:
    load -> remove duplicates & leaky columns -> SPLIT -> lock test set away
Nothing that *learns* from the data (scaling, encoding, imputation) happens
here. That all lives inside the Pipeline, which is fit on training data only.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (RAW_DATA, SPLIT_DIR, TARGET, LEAKY_COLUMNS,
                        INPUT_COLUMNS, TEST_SIZE, RANDOM_STATE)


def load_raw() -> pd.DataFrame:
    """Read the CSV (it uses ';' as separator)."""
    return pd.read_csv(RAW_DATA, sep=";")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning that does NOT learn anything from the data, so it is safe to do
    before splitting:
      1. drop exact duplicate rows (otherwise a row could land in both train
         and test -> leakage type 3 from Lecture 7)
      2. drop the leaky `duration` column
      3. turn the label 'yes'/'no' into 1/0
    """
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=LEAKY_COLUMNS)
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def split(df: pd.DataFrame):
    """
    Stratified 80/20 split. `stratify=y` keeps the 11% 'yes' rate identical in
    train and test, so a lucky/unlucky split can't distort the test score.
    """
    X = df[INPUT_COLUMNS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            stratify=y, random_state=RANDOM_STATE)


def make_and_save_splits():
    """Create the split ONCE and save it, so every later script uses the
    exact same test set (and nobody re-splits and accidentally peeks)."""
    df = clean(load_raw())
    X_train, X_test, y_train, y_test = split(df)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(SPLIT_DIR / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(SPLIT_DIR / "test.csv", index=False)
    print(f"train: {len(X_train):,} rows  (yes rate {y_train.mean():.3%})")
    print(f"test : {len(X_test):,} rows  (yes rate {y_test.mean():.3%})  <- LOCKED")


def load_split(name: str):
    """name = 'train' or 'test'. Returns X, y."""
    df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
    return df[INPUT_COLUMNS], df[TARGET]


if __name__ == "__main__":
    make_and_save_splits()
