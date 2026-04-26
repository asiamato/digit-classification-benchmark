import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(csv_path="data/digits.csv"):
    """Load the UCI digits dataset. Falls back to CSV if sklearn is unavailable."""
    try:
        digits = load_digits()
        X, y = digits.data, digits.target
        images = digits.images
        feat_names = [f"pixel_{i:02d}" for i in range(X.shape[1])]
    except Exception:
        df = pd.read_csv(csv_path, header=None)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(int)
        images = X.reshape(-1, 8, 8)
        feat_names = [f"pixel_{i:02d}" for i in range(X.shape[1])]

    return X, y, images, feat_names


def split_data(X, y, test_size=0.20, val_size=0.10, seed=42):
    """Stratified train/val/test split."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=seed, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def dataset_summary(X, y):
    labels, counts = np.unique(y, return_counts=True)
    return pd.DataFrame({
        "digit": labels,
        "count": counts,
        "proportion": np.round(counts / len(y), 4),
    })
