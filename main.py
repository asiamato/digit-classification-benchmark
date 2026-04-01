import os
import time
import argparse
import warnings
import logging
import random
import numpy as np
import pandas as pd
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

# Standard logger configuration for scientific research
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def set_global_seed(seed: int = 42) -> None:
    """Sets the global seeds to ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Global seed set to {seed}")

def main(no_tune: bool = False) -> None:
    """Main execution pipeline for the digit recognition research."""
    # Initial setup
    os.makedirs("results", exist_ok=True)
    set_global_seed(42)
    
    logging.info("Loading dataset...")
    X, y, images, feature_names = load_data()
    logging.info(f"Dataset summary:\n{dataset_summary(X, y).to_string(index=False)}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    labels = [str(d) for d in range(10)]

    # 1. Baseline Model
    # p(C_k | x) = p(x | C_k) p(C_k) / p(x)
    logging.info("Training Baseline Model (PCA + SVM)...")
    baseline = build_baseline_pipeline()
    baseline.fit(X_trainval, y_trainval)
    logging.info(f"Baseline accuracy: {baseline.score(X_test, y_test):.4f}")
    
    # 2. Comparative Evaluation
    logging.info("Performing comparative evaluation of candidate models...")
    models = build_candidate_models()
    results = comparative_evaluation(models, X_trainval, y_trainval, X_test, y_test)
    logging.info(f"Comparative results:\n{results.to_string()}")
    
    save_results_csv(results, "results/metrics_summary.csv")
    plot_model_comparison(results, "accuracy", "results/comparison.png")
    
    # 3. Hyperparameter Tuning
    svm_tuned = mlp_tuned = None
    if not no_tune:
        logging.info("Tuning SVM hyperparameters...")
        svm_tuned = tune_svm(X_trainval, y_trainval).best_estimator_
        logging.info("Tuning MLP hyperparameters...")
        mlp_tuned = tune_mlp(X_trainval, y_trainval).best_estimator_
        
    # 4. Ensemble Voting
    logging.info("Building and training Soft-Voting Ensemble...")
    ensemble = build_ensemble(svm_tuned, mlp_tuned)
    ensemble.fit(X_trainval, y_trainval)
    logging.info(f"Ensemble accuracy: {ensemble.score(X_test, y_test):.4f}")
    
    # 5. Statistical Significance (McNemar's test)
    # χ² = (|b − c| − 1)² / (b + c)
    best_single_model_name = results.iloc[0]["model"]
    y_pred_best = models[best_single_model_name].predict(X_test)
    y_pred_ens  = ensemble.predict(X_test)
    
    mcn = mcnemar_test(y_test, y_pred_best, y_pred_ens, best_single_model_name, "Ensemble")
    logging.info(f"McNemar's Test Result: {mcn['summary']}")
    
    # 6. Final Plots and Visualizations
    logging.info("Generating final evaluation plots...")
    plot_confusion_matrix(ensemble, X_test, y_test, labels, "Ensemble", "results/ensemble_cm.png")
    plot_learning_curves(ensemble, X_trainval, y_trainval, "Learning Curve", cv=5, save_path="results/learning_curve.png")
    
    proba_models = {n: m for n, m in models.items() if hasattr(m, "predict_proba")}
    plot_calibration_curves(proba_models, X_test, y_test, save_path="results/calibration.png")
    
    logging.info("Pipeline completed successfully. Results saved in the 'results/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digit Recognition Research Pipeline")
    parser.add_argument("--no-tune", action="store_true", help="Skip the hyperparameter tuning phase for a faster run")
    args = parser.parse_args()
    main(args.no_tune)