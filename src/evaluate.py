import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, log_loss, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", context="notebook", rc={"figure.dpi": 150})


def _brier_multiclass(y_true, y_proba):
    """Brier score extended to K classes (mean squared error on probability vectors)."""
    n_classes = y_proba.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    metrics = {
        "model":           model_name,
        "accuracy":        accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted":     f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # probabilistic metrics (only if the model supports predict_proba)
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            metrics["log_loss"]    = log_loss(y_test, y_proba)
            metrics["roc_auc"]     = roc_auc_score(y_test, y_proba,
                                                   multi_class="ovr", average="weighted")
            metrics["brier_score"] = _brier_multiclass(y_test, y_proba)
        except Exception:
            metrics["log_loss"]    = np.nan
            metrics["roc_auc"]     = np.nan
            metrics["brier_score"] = np.nan

    return metrics


def comparative_evaluation(models, X_train, y_train, X_test, y_test, cv=5):
    """Train each model, evaluate on test set, and run cross-validation."""
    rows = []
    for name, pipe in models.items():
        print(f"  Training {name}...")
        pipe.fit(X_train, y_train)
        row = evaluate_model(pipe, X_test, y_test, model_name=name)

        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=cv, scoring="accuracy", n_jobs=-1)
        row["cv_mean"]  = cv_scores.mean()
        row["cv_std"]   = cv_scores.std()
        row["overfit"]  = row["cv_mean"] - row["accuracy"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test, labels=None,
                          title="Confusion Matrix", save_path=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title(f"{title} — Counts")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title(f"{title} — Recall (normalised)")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_learning_curves(model, X, y, title="Learning Curve", cv=5, save_path=None):
    sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_sc, val_sc = learning_curve(
        model, X, y, train_sizes=sizes, cv=cv,
        scoring="accuracy", n_jobs=-1, random_state=42,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes, train_sc.mean(axis=1), "o-", label="Train")
    ax.plot(train_sizes, val_sc.mean(axis=1), "s-", label="Validation")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_model_comparison(results_df, metric="accuracy", save_path=None):
    df = results_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["model"], df[metric])
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def mcnemar_test(y_true, y_pred_a, y_pred_b, model_a="A", model_b="B"):
    """
    McNemar's test on paired predictions.
    Returns chi2 statistic, p-value, and whether p < 0.05.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    # b = cases where A is right but B is wrong, c = opposite
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False,
                "summary": "Models make identical errors."}

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)

    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2_stat, df=1)

    return {
        "chi2": chi2_stat,
        "p_value": p,
        "significant": p < 0.05,
        "summary": f"McNemar p={p:.4f} ({model_a} vs {model_b})",
    }


def plot_calibration_curves(models, X_test, y_test, n_bins=10, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        try:
            y_proba = model.predict_proba(X_test)
            classes = np.unique(y_test)
            all_frac, all_pred = [], []
            for c in classes:
                y_bin = (y_test == c).astype(int)
                frac, pred = calibration_curve(y_bin, y_proba[:, c], n_bins=n_bins)
                all_frac.append(frac)
                all_pred.append(pred)
            ax.plot(np.mean(all_pred, axis=0), np.mean(all_frac, axis=0),
                    "o-", label=name)
        except Exception:
            continue

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="lower right", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def save_results_csv(df, path):
    df.to_csv(path, index=False)