import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_data, split_data, dataset_summary
from src.models import (
    build_baseline_pipeline,
    build_candidate_models,
    tune_svm,
    tune_mlp,
    build_ensemble,
)
from src.evaluate import (
    comparative_evaluation,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_model_comparison,
    mcnemar_test,
    plot_calibration_curves,
    save_results_csv,
)

warnings.filterwarnings("ignore")

def main(no_tune=False):
    os.makedirs("results", exist_ok=True)
    np.random.seed(42)

    print("Loading data...")
    X, y, images, feat_names = load_data()
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Merge train and val since we'll use CV or just train on everything before testing
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    labels = [str(d) for d in range(10)]

    print("Fitting baseline model (PCA + RBF-SVM)...")
    baseline = build_baseline_pipeline()
    baseline.fit(X_tv, y_tv)
    print(f"Baseline test accuracy: {baseline.score(X_test, y_test):.4f}")

    print("Training and evaluating candidate models...")
    models = build_candidate_models()
    results = comparative_evaluation(models, X_tv, y_tv, X_test, y_test)
    print(f"Results:\n{results.to_string()}")

    save_results_csv(results, "results/metrics_summary.csv")
    plot_model_comparison(results, "accuracy", "results/comparison.png")

    svm_tuned = None
    mlp_tuned = None
    
    if not no_tune:
        print("Running SVM grid search (this might take a bit)...")
        svm_tuned = tune_svm(X_tv, y_tv).best_estimator_
        print("Running MLP randomized search...")
        mlp_tuned = tune_mlp(X_tv, y_tv).best_estimator_

    print("Building soft-voting ensemble...")
    ensemble = build_ensemble(svm_tuned, mlp_tuned)
    ensemble.fit(X_tv, y_tv)
    print(f"Ensemble test accuracy: {ensemble.score(X_test, y_test):.4f}")

    best_name = results.iloc[0]["model"]
    y_pred_best = models[best_name].predict(X_test)
    y_pred_ens = ensemble.predict(X_test)

    mcn = mcnemar_test(y_test, y_pred_best, y_pred_ens, best_name, "Ensemble")
    print(f"McNemar: {mcn['summary']}")

    print("Generating plots...")
    plot_confusion_matrix(ensemble, X_test, y_test, labels,
                          "Ensemble", "results/ensemble_cm.png")
    plot_learning_curves(ensemble, X_tv, y_tv,
                         "Ensemble — Learning Curve", cv=5,
                         save_path="results/learning_curve.png")

    proba_models = {n: m for n, m in models.items() if hasattr(m, "predict_proba")}
    plot_calibration_curves(proba_models, X_test, y_test,
                            save_path="results/calibration.png")

    print("Done. Results saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digit classification benchmark")
    parser.add_argument("--no-tune", action="store_true",
                        help="skip hyperparameter tuning for a faster run")
    args = parser.parse_args()
    main(args.no_tune)