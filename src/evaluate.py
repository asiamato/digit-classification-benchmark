import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Optional, List

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    cross_val_score,
    learning_curve,
)
from sklearn.calibration import calibration_curve

# Plot defaults for academic style
sns.set_theme(
    style="whitegrid",
    context="notebook",
    rc={"figure.dpi": 150},
)

def _multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calculates the Brier Score for multi-class classification problems."""
    n_classes = y_proba.shape[1]
    y_true_one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - y_true_one_hot)**2, axis=1)))

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """Evaluates a model computing standard and probabilistic metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "model":           model_name,
        "accuracy":        accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_test, y_pred, average="macro",    zero_division=0),
        "f1_macro":        f1_score(y_test, y_pred, average="macro",        zero_division=0),
        "f1_weighted":     f1_score(y_test, y_pred, average="weighted",     zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            metrics["log_loss"] = log_loss(y_test, y_proba)
            metrics["roc_auc"]  = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted",
            )
            metrics["brier_score"] = _multiclass_brier_score(y_test, y_proba)
        except Exception:
            metrics["log_loss"] = np.nan
            metrics["roc_auc"]  = np.nan
            metrics["brier_score"] = np.nan
            
    return metrics

def comparative_evaluation(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray, cv: int = 5) -> pd.DataFrame:
    """Trains and evaluates multiple candidate models."""
    records = []
    for name, pipe in models.items():
        logging.info(f"  Training {name} ...")
        pipe.fit(X_train, y_train)
        row  = evaluate_model(pipe, X_test, y_test, model_name=name)
        
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        row["cv_mean"]  = cv_scores.mean()
        row["cv_std"]   = cv_scores.std()
        row["overfit"]  = row["cv_mean"] - row["accuracy"]
        records.append(row)
    
    return pd.DataFrame(records).sort_values("accuracy", ascending=False)

def plot_confusion_matrix(model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                          labels: Optional[List[str]] = None, title: str = "Confusion Matrix", 
                          save_path: Optional[str] = None):
    """Plots both absolute and normalized confusion matrices."""
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} (Counts)")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{title} (Normalized / Recall)")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

def plot_learning_curves(model: Any, X: np.ndarray, y: np.ndarray, 
                         title: str = "Learning Curve", cv: int = 5, save_path: Optional[str] = None):
    """Generates learning curves to assess underfitting or overfitting."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, scoring="accuracy", n_jobs=-1, random_state=42,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes_abs, train_scores.mean(axis=1), "o-", label="Training score")
    ax.plot(train_sizes_abs, val_scores.mean(axis=1),   "s-", label="Validation score")
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

def plot_model_comparison(results_df: pd.DataFrame, metric: str = "accuracy", save_path: Optional[str] = None):
    """Creates a horizontal bar chart comparing models based on a specific metric."""
    df = results_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["model"], df[metric])
    ax.set_title(f"Model Comparison - {metric.replace('_', ' ').title()}")
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, 
                 model_a: str = "A", model_b: str = "B") -> Dict[str, Any]:
    """
    Performs McNemar's statistical test to compare two classifiers.
    Formula: χ² = (|b − c| − 1)² / (b + c)
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False, "summary": "Identical errors."}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "chi2":        chi2,
        "p_value":     p_value,
        "significant": p_value < 0.05,
        "summary":     f"McNemar p-value = {p_value:.4f} ({model_a} vs {model_b})",
    }

def plot_calibration_curves(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, 
                            n_bins: int = 10, save_path: Optional[str] = None):
    """Plots calibration curves (reliability diagrams) for probabilistic models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        try:
            y_proba = model.predict_proba(X_test)
            classes = np.unique(y_test)
            fracs, preds = [], []
            for c in classes:
                y_bin = (y_test == c).astype(int)
                frac, pred = calibration_curve(y_bin, y_proba[:, c], n_bins=n_bins)
                fracs.append(frac)
                preds.append(pred)
            ax.plot(np.mean(preds, axis=0), np.mean(fracs, axis=0), "o-", label=name)
        except Exception:
            continue
            
    ax.set_title("Calibration Curves (Reliability Diagram)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

def save_results_csv(df: pd.DataFrame, path: str) -> None:
    """Saves the evaluation results to a CSV file."""
    df.to_csv(path, index=False)