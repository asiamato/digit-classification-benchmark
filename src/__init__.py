# Digit Recognition Research Package
# Based on Bishop (2006) and Goodfellow et al. (2016)

from src.data_loader import load_data, split_data, dataset_summary
from src.models import (
    build_baseline_pipeline,
    build_candidate_models,
    tune_svm,
    tune_mlp,
    build_ensemble,
)
from src.evaluate import (
    evaluate_model,
    comparative_evaluation,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_model_comparison,
    plot_calibration_curves,
    mcnemar_test,
    save_results_csv,
)
