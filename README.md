# digit-classification-benchmark
A rigorous comparative analysis of Machine Learning classifiers and ensemble strategies on the UCI Handwritten Digits dataset, featuring statistical significance testing and probabilistic calibration.


This repository contains a comprehensive comparative analysis of machine learning classifiers applied to the UCI Optical Recognition of Handwritten Digits dataset. The project evaluates different classification strategies — from basic generative models to modern ensemble methods — following the theoretical frameworks of Bishop (2006) and Goodfellow et al. (2016).

Beyond standard accuracy metrics, this study emphasizes **probabilistic calibration**, **computational efficiency (PCA)**, and **statistical significance testing** to rigorously evaluate model performance in a research context.

## Key Highlights

* **Dimensionality Reduction**: Implemented PCA (30 components) retaining >99% of the variance, significantly reducing training time while preserving critical spatial features.
* **Algorithm Benchmarking**: Trained and evaluated 9 distinct model architectures, including Kernel SVMs, Multi-Layer Perceptrons (MLP), Random Forests, and Gradient Boosting.
* **Ensemble Learning**: Built a custom Soft-Voting Ensemble combining the top-performing models (SVM and MLP), achieving the highest overall accuracy (**97.78%**).
* **Statistical Rigor**: Applied **McNemar's Test** to compare the predictive marginal homogeneity of the best single model versus the ensemble, providing statistical context for model selection in production environments.
* **Confidence & Calibration**: Evaluated model certainty using Log Loss and Multiclass Brier Score, revealing that while tree-based models (like Random Forest) achieve high accuracy, they suffer from overconfidence compared to highly calibrated models like Logistic Regression and SVM.

## Comparative Model Performance

The following table summarizes the test-set performance of all evaluated models, sorted by Accuracy. 

| Model                | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | Log Loss | ROC AUC | Brier Score | CV Mean | Overfit |
|:---------------------|---------:|------------------:|---------------:|-----------:|---------:|--------:|------------:|--------:|--------:|
| **SVM (RBF kernel)** | **0.9722** | 0.9732 | 0.9721 | 0.9720 | 0.1227 | **0.9993** | **0.0422** | 0.9812 | 0.0090 |
| K-Nearest Neighbours | 0.9694 | 0.9706 | 0.9691 | 0.9689 | 0.3573 | 0.9950 | 0.0500 | 0.9742 | 0.0048 |
| MLP Neural Network   | 0.9694 | 0.9704 | 0.9690 | 0.9688 | 0.1343 | 0.9968 | 0.0487 | 0.9694 |-0.0001 |
| QDA                  | 0.9556 | 0.9565 | 0.9555 | 0.9555 | 0.8859 | 0.9866 | 0.0825 | 0.9575 | 0.0020 |
| Logistic Regression  | 0.9556 | 0.9558 | 0.9552 | 0.9553 | 0.1472 | 0.9979 | 0.0741 | 0.9582 | 0.0027 |
| LDA (Fisher)         | 0.9528 | 0.9526 | 0.9523 | 0.9523 | 0.1990 | 0.9977 | 0.0767 | 0.9548 | 0.0020 |
| Random Forest        | 0.9528 | 0.9549 | 0.9525 | 0.9527 | 0.4112 | 0.9979 | 0.1572 | 0.9631 | 0.0103 |
| Gradient Boosting    | 0.9194 | 0.9210 | 0.9189 | 0.9188 | 0.4955 | 0.9945 | 0.1376 | 0.9186 |-0.0009 |
| Gaussian Naive Bayes | 0.9111 | 0.9150 | 0.9108 | 0.9116 | 0.8140 | 0.9813 | 0.1486 | 0.8775 |-0.0336 |

*(Note: CV Mean represents the 5-fold cross-validation accuracy on the training set. Overfit is the difference between CV Mean and Test Accuracy. Brier Score measures probabilistic calibration, where closer to 0 is better).*

## Mathematical Framework

The classification task is grounded in Bayesian decision theory, aiming to accurately model the posterior probability of class $\mathcal{C}_k$ given the input feature vector $\mathbf{x}$:

$$p(\mathcal{C}_k \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathcal{C}_k)\, p(\mathcal{C}_k)}{p(\mathbf{x})}$$

Statistical significance between the top models is assessed using **McNemar's test** on paired nominal data, utilizing the standard chi-squared statistic:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

*(Where $b$ and $c$ represent the discordant error pairs between two classifiers).*

## Project Structure

* `src/`: Core Python modules for data processing, model building, and evaluation matrices.
* `main.py`: The main automated pipeline script for training, tuning, and testing.
* `notebooks/analysis.ipynb`: An interactive Jupyter Notebook documenting the exploratory data analysis, visual storytelling, and research conclusions.
* `data/`: Local storage for the dataset (automatically fetched via `scikit-learn` if empty).
* `results/`: Output directory for generated CSV metrics and high-resolution plots (Confusion Matrices, Calibration Curves, Learning Curves).

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/digit-classification-benchmark.git](https://github.com/yourusername/digit-classification-benchmark.git)
   cd digit-classification-benchmark
   
2. Install the required dependencies:
   
   ```bash 
   pip install -r requirements.txt
   Run the full research pipeline (includes cross-validated hyperparameter tuning):

   ```bash
   python main.py


   Optional: Skip the hyperparameter tuning phase for a faster baseline run:

   ```bash
   python main.py --no-tune

## References
Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management.