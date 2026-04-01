import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data(csv_path="data/digits.csv"):
    # Load UCI Digits (8x8 images, 64 features)
    try:
        digits = load_digits()
        X, y   = digits.data, digits.target
        images = digits.images
        feature_names = [f"pixel_{i:02d}" for i in range(X.shape[1])]
    except Exception:
        df = pd.read_csv(csv_path)
        X  = df.iloc[:, :-1].values.astype(float)
        y  = df.iloc[:,  -1].values.astype(int)
        images = X.reshape(-1, 8, 8)
        feature_names = list(df.columns[:-1])
    
    return X, y, images, feature_names

def split_data(X, y, test_size=0.20, val_size=0.10, random_state=42):
    # Stratified split to preserve class distribution
    # Bishop (2006)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_val, random_state=random_state, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def dataset_summary(X, y):
    labels, counts = np.unique(y, return_counts=True)
    return pd.DataFrame({
        "digit":     labels,
        "count":     counts,
        "proportion": counts / len(y),
    })
