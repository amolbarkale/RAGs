
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple

def load_and_prepare_data(
    path_pattern: str = "../data/*.csv",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    1. Globs all CSVs matching path_pattern
    2. Concatenates them into one DataFrame
    3. Standardizes to columns ['text','label']
    4. Shuffles and splits into train/test
    """
    # 1) Read & concatenate
    files = glob.glob(path_pattern)
    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)

    # 2) Standardize column names
    #    – if you have 'headline' + 'short_description', merge them
    if {"headline", "short_description"}.issubset(df.columns):
        df["text"] = df["headline"].fillna("") + " " + df["short_description"].fillna("")
    #    – make sure we have `text` & `label`
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Each CSV must have at least `text` and `label` columns")

    # 3) Drop any rows with missing text/label
    df = df[["text", "label"]].dropna().reset_index(drop=True)

    # 4) Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 5) Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
