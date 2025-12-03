# 📘 House Price Prediction — Version 2  
### *End-to-End Machine Learning Pipeline with Linear Models & Custom Gradient Descent*

---

## 1. Introduction

This project builds a complete machine learning pipeline to predict median house prices in California.  
Version 2 introduces major improvements over Version 1, including modular source code, engineered features, multiple linear models, systematic cross-validation, hyperparameter tuning, and a custom Gradient Descent Regressor implemented from scratch.

**TODO:** Add a short motivation (why this project matters).

---

## 2. Dataset

- **Source:** California Housing dataset (`sklearn.datasets.fetch_california_housing`)
- **Samples:** ~20,000
- **Features:** 8 numerical predictors
- **Target:** `median_house_value`
- **Type:** Tabular, numerical regression dataset

Preprocessing steps and feature transformations are implemented in  
`src/feature_engineering.py`.

**TODO:** Add summary statistics if desired.

---

## 3. Project Structure

project/
│
├── notebooks/
│ ├── 01-feature-engineering.ipynb
│ ├── 02-experimental-models.ipynb
│ ├── 03-gradient-descent.ipynb
│ ├── 04-cross-validation.ipynb
│ └── 05-training-pipeline-demo.ipynb
│
├── src/
│ ├── feature_engineering.py
│ ├── gradient_descent.py
│ ├── hyperparameter_tuning.py
│ ├── evaluation.py
│ ├── training_pipeline.py
│ └── utils.py
│
├── report.md
├── README.md
└── requirements.txt


Version 2 emphasizes clean separation of responsibilities, reproducibility, and reusability across notebooks and scripts.

---

## 4. Methodology

### 4.1 Feature Engineering

Feature transformations include:

- Handling missing values  
- Scaling using `StandardScaler`  
- Ratio-based transformations (if applicable)  
- Ensuring consistent preprocessing during training and inference  

All functions are modularized inside `feature_engineering.py`.

---

### 4.2 Models Implemented

| Model | Implementation | Purpose |
|-------|----------------|----------|
| OLS (Linear Regression) | scikit-learn | Baseline model |
| Ridge Regression | scikit-learn | L2 regularization for stability |
| Lasso Regression | scikit-learn | L1 regularization for sparsity |
| Gradient Descent Regressor | Custom implementation | Manual optimization, educational comparison |

The custom Gradient Descent model supports:

- configurable learning rate  
- iterations  
- convergence tracking  
- MSE loss optimization  

---

### 4.3 Training Pipeline

A full training workflow is implemented in `training_pipeline.py`, including:

- data loading  
- feature engineering  
- splitting into train/validation/test  
- model training  
- evaluation  
- prediction  

This encapsulates the entire workflow into a reproducible pipeline.

---

## 5. Cross-Validation

Cross-validation is performed in `04-cross-validation.ipynb` using a 5-fold setup to measure model stability and variance.

Metrics include:

- Mean R² across folds  
- Standard deviation of R²  
- Fold-level scores for model comparison  

> **Note:**  
> In industry, hyperparameter tuning typically performs cross-validation internally (nested CV).  
> In this project, CV and tuning are separated intentionally for educational clarity and modularity.

---

## 6. Hyperparameter Tuning

The notebook and `hyperparameter_tuning.py` explore:

- a predefined grid of α values for Ridge and Lasso  
- evaluation using validation splits  
- selection of hyperparameters based on highest validation R²  

**TODO:** Add best α values if you want them documented.

---

## 7. Results

### 7.1 Cross-Validation Results

**TODO:** Insert table of CV results, including mean and std of R² for all models.

Example:

| Model | Mean R² | Std R² |
|-------|---------|--------|
| OLS | ... | ... |
| Ridge | ... | ... |
| Lasso | ... | ... |
| Gradient Descent | ... | ... |

---

### 7.2 Final Test Performance

After selecting hyperparameters, models are retrained on the full training set and evaluated once on the test set to avoid leakage.

**TODO:** Insert final scores.

---

## 8. Custom Gradient Descent Regressor

Notebook `03-gradient-descent.ipynb` documents:

- update rule derivations  
- cost function visualization  
- convergence tracking  
- comparison vs. scikit-learn OLS  

This model demonstrates understanding of optimization fundamentals and how manual training differs from closed-form solutions.

---

## 9. Limitations

- Only linear models are explored  
- Limited hyperparameter search space  
- No tree-based or neural models  
- Not optimized for deployment or production  
- California Housing dataset may not generalize beyond domain  

---

## 10. Future Work

- Add ElasticNet  
- Expand hyperparameter search  
- Implement nested cross-validation  
- Introduce Random Forest or Gradient Boosting models  
- Add UI or API for prediction demos  
- Explore polynomial or interaction features  

---

## 11. Reproducibility

The entire workflow is traceable through:

- modular source files  
- sequential notebooks  
- deterministic random seeds  
- a reproducible training pipeline script  

---

# End of Report
