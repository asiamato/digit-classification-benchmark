# digit-classification-benchmark

A comparative analysis of ML classifiers on the UCI Handwritten Digits dataset, with statistical testing and calibration evaluation.

This project benchmarks different classification strategies — from generative models (Naive Bayes, QDA) to discriminative ones (SVM, MLP) and ensemble methods — on the classic 8×8 handwritten digits dataset. The focus goes beyond raw accuracy: we also look at **probabilistic calibration**, **dimensionality reduction via PCA**, and whether differences between models are actually **statistically significant** (McNemar's test).

## Highlights

* **PCA preprocessing**: 30 components retain >99% of variance, cutting training time without losing spatial information.
* **9 classifiers compared**: Kernel SVM, MLP, KNN, Random Forest, Gradient Boosting, LDA, QDA, Logistic Regression, Gaussian Naive Bayes.
* **Soft-Voting Ensemble**: Combines the best-performing models; achieves **97.22%** test accuracy.
* **Statistical testing**: McNemar's test checks whether the ensemble genuinely outperforms the best single model or if the difference is just noise.
* **Calibration analysis**: Log Loss and Brier Score reveal that high-accuracy models (like Random Forest) can be poorly calibrated, while simpler models (Logistic Regression) produce more reliable probability estimates.

## Results

| Model                | Accuracy | Precision | Recall | F1     | Log Loss | ROC AUC | Brier Score | CV Mean | Overfit |
|:---------------------|----------|-----------|--------|--------|----------|---------|-------------|---------|---------|
| **SVM (RBF kernel)** | **0.972**| 0.973     | 0.972  | 0.972  | 0.123    | **0.999**| **0.042**  | 0.981   | 0.009   |
| K-Nearest Neighbours | 0.969    | 0.971     | 0.969  | 0.969  | 0.357    | 0.995   | 0.050      | 0.974   | 0.005   |
| MLP Neural Network   | 0.969    | 0.970     | 0.969  | 0.969  | 0.134    | 0.997   | 0.049      | 0.969   | 0.000   |
| QDA                  | 0.956    | 0.957     | 0.956  | 0.956  | 0.886    | 0.987   | 0.083      | 0.958   | 0.002   |
| Logistic Regression  | 0.956    | 0.956     | 0.955  | 0.955  | 0.147    | 0.998   | 0.074      | 0.958   | 0.003   |
| LDA (Fisher)         | 0.953    | 0.953     | 0.952  | 0.952  | 0.199    | 0.998   | 0.077      | 0.955   | 0.002   |
| Random Forest        | 0.953    | 0.955     | 0.953  | 0.953  | 0.411    | 0.998   | 0.157      | 0.963   | 0.010   |
| Gradient Boosting    | 0.933    | 0.934     | 0.933  | 0.933  | 0.353    | 0.997   | 0.106      | 0.935   | 0.002   |
| Gaussian Naive Bayes | 0.911    | 0.915     | 0.911  | 0.912  | 0.814    | 0.981   | 0.149      | 0.877   | -0.034  |

Precision, Recall, and F1 are macro-averaged. CV Mean is the 5-fold cross-validation accuracy on the training set. Overfit = CV Mean − Test Accuracy. Brier Score measures calibration (lower is better).

## Background

The classification task follows a Bayesian decision framework — each model approximates the posterior

$$p(\mathcal{C}_k \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathcal{C}_k)\, p(\mathcal{C}_k)}{p(\mathbf{x})}$$

either directly (discriminative models) or through the class-conditional likelihood (generative models).

To compare classifiers we use McNemar's test on the contingency table of their errors:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

where *b* and *c* are the discordant error counts.

## Project Structure

```
├── src/               # Core modules (data loading, models, evaluation)
├── main.py            # Full pipeline: train → evaluate → plot
├── notebooks/
│   └── analysis.ipynb # Interactive walkthrough with visualisations
├── data/
│   └── digits.csv     # Fallback dataset (usually fetched via sklearn)
└── results/           # Generated metrics CSV + plots
```

## Getting Started

```bash
git clone https://github.com/yourusername/digit-classification-benchmark.git
cd digit-classification-benchmark
pip install -r requirements.txt
```

Run the full pipeline (with hyperparameter tuning):

```bash
python main.py
```

Or skip tuning for a quick run:

```bash
python main.py --no-tune
```

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*.