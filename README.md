<div align="center">

# 📡 Fast Radio Burst Repeater Classification

### Machine learning classification of repeating and non-repeating Fast Radio Bursts using CHIME/FRB catalogue features

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Task-Binary%20Classification-2E8B57)
![Best Model](https://img.shields.io/badge/Best%20Model-LightGBM-7A1FA2)
![Best F1](https://img.shields.io/badge/Test%20F1-0.9457-brightgreen)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.9910-blue)

</div>

---

## 📌 Project Overview

Fast Radio Bursts (FRBs) are short-duration, high-energy radio signals originating from distant astronomical sources. Some FRB sources are observed only once, while others produce multiple detectable bursts and are classified as **repeaters**.

This project develops an end-to-end machine learning pipeline for classifying FRB events as:

- **Non-repeating FRB — `0`**
- **Repeating FRB — `1`**

The target variable is derived from the `repeater_name` field in the dataset. To prevent target leakage, `repeater_name` and the generated `Target` column are excluded from the model features.

The pipeline includes data inspection, feature selection, median imputation, feature-set comparison, Optuna-based hyperparameter optimization, evaluation of five ensemble classifiers, cross-validation, statistical significance testing, and SHAP-based model interpretation.

---

## 🎯 Project Objectives

- Build a reliable binary classifier for repeating and non-repeating FRB events.
- Use physically meaningful dispersion, spectral, temporal, flux, exposure, and sky-position features.
- Compare a core 17-feature set with an expanded 22-feature set.
- Optimize multiple tree-based classifiers using Optuna.
- Evaluate performance using metrics suited to an imbalanced classification problem.
- Statistically compare the competing models.
- Explain the best model using feature importance and SHAP values.
- Generate event-level repeater probabilities for further analysis.

---

## ✨ Key Features

- Automatic binary-target creation from catalogue metadata
- Explicit target-leakage prevention
- Missing-value analysis and median imputation
- Stratified training and testing split
- Core versus expanded feature-set comparison
- Optuna hyperparameter optimization
- Five optimized ensemble-learning classifiers
- Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and PR-AUC evaluation
- Confusion matrices and ROC-curve comparison
- Ten-fold stratified cross-validation
- Friedman and pairwise Wilcoxon statistical tests
- Global and local SHAP explanations
- CSV export of model results, predictions, and feature importance

---

## 📊 Dataset Summary

The notebook uses the dataset file:

```text
chimefrbcat2_dataset.csv
```

| Property | Value |
|---|---:|
| Total records | 5,045 |
| Original columns | 60 |
| Duplicate records removed | 0 |
| Selected model features | 22 |
| Non-repeating events | 3,763 |
| Repeating events | 1,282 |
| Training records | 4,036 |
| Testing records | 1,009 |
| Train-test split | 80:20 |

### Class Distribution

| Class | Count | Proportion |
|---|---:|---:|
| Non-repeating | 3,763 | 74.59% |
| Repeating | 1,282 | 25.41% |

Because the classes are imbalanced, the project reports **balanced accuracy, F1-score, ROC-AUC, and PR-AUC** in addition to standard accuracy.

---

## 🧪 Target Definition

The target is created using the following rule:

```python
df["Target"] = df["repeater_name"].notna().astype(int)
```

| Target | Meaning |
|---:|---|
| `0` | Non-repeating FRB |
| `1` | Repeating FRB |

The columns `repeater_name` and `Target` are not used as predictors.

---

## 🔭 Selected Features

### Core 17 Features

| Feature | General role |
|---|---|
| `dm_fitb` | Fitted dispersion measure |
| `dm_exc_ne2001` | Excess dispersion measure based on NE2001 |
| `dm_exc_ymw16` | Excess dispersion measure based on YMW16 |
| `snr_fitb` | Fitted signal-to-noise ratio |
| `bonsai_snr` | BONSAI pipeline signal-to-noise ratio |
| `bonsai_dm` | BONSAI pipeline dispersion measure |
| `fluence` | Integrated burst fluence |
| `flux` | Burst flux |
| `scat_time` | Scattering timescale |
| `peak_freq` | Peak observing frequency |
| `high_freq` | Upper frequency boundary |
| `low_freq` | Lower frequency boundary |
| `sp_idx` | Spectral index |
| `sp_run` | Spectral running |
| `bc_width` | Burst-component width |
| `gb` | Galactic latitude |
| `gl` | Galactic longitude |

### Additional 5 Features

| Feature | General role |
|---|---|
| `exp_up` | Upper exposure-related value |
| `exp_low` | Lower exposure-related value |
| `mjd_inf` | Infinite-frequency arrival time in MJD |
| `ra` | Right ascension |
| `dec` | Declination |

All 22 requested features are available in the notebook dataset.

---

## ⚙️ Data Preprocessing

The preprocessing workflow is designed to reduce data leakage and retain the original physical scales of the variables.

1. Load and inspect the dataset.
2. Check missing values and duplicate records.
3. Generate the binary target.
4. Select the 22 physically meaningful predictors.
5. Convert selected columns to numeric values.
6. Remove features that are completely missing or constant.
7. Perform an 80:20 stratified train-test split.
8. Fit a median imputer only on the training data.
9. Transform the testing data using the fitted training imputer.
10. Train tree-based classifiers without feature standardization.

No missing values remain in the final training and testing matrices.

---

## 🧩 Feature-Set Comparison

A fixed LightGBM configuration is first used to compare the core and expanded feature sets.

| Feature Set | Features | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Core requested features | 17 | 0.9653 | 0.9458 | 0.9547 | 0.9062 | 0.9299 | 0.9886 | 0.9783 |
| **Core + additional features** | **22** | **0.9762** | **0.9609** | **0.9754** | **0.9297** | **0.9520** | **0.9920** | **0.9859** |

The expanded 22-feature set is therefore used for the optimized model comparison.

---

## 🤖 Models Evaluated

The following classifiers are trained and optimized:

1. Random Forest
2. Extra Trees
3. XGBoost
4. LightGBM
5. CatBoost

Hyperparameter optimization is performed using **Optuna**, with mean cross-validated **F1-score** as the optimization objective.

---

## 🏆 Test-Set Results

Models are ranked by F1-score.

| Rank | Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **LightGBM** | **0.9732** | **0.9550** | 0.9751 | **0.9180** | **0.9457** | **0.9910** | 0.9830 |
| 2 | CatBoost | 0.9693 | 0.9459 | 0.9787 | 0.8984 | 0.9369 | 0.9901 | **0.9834** |
| 3 | XGBoost | 0.9633 | 0.9355 | 0.9740 | 0.8789 | 0.9240 | 0.9861 | 0.9719 |
| 4 | Extra Trees | 0.9613 | 0.9277 | **0.9865** | 0.8594 | 0.9186 | 0.9873 | 0.9707 |
| 5 | Random Forest | 0.9574 | 0.9289 | 0.9571 | 0.8711 | 0.9121 | 0.9894 | 0.9753 |

### Best Model: LightGBM

LightGBM achieves the strongest overall test-set performance:

- **Accuracy:** 97.32%
- **Balanced accuracy:** 95.50%
- **Precision:** 97.51%
- **Recall:** 91.80%
- **F1-score:** 94.57%
- **ROC-AUC:** 99.10%
- **PR-AUC:** 98.30%

### Best-Model Confusion Matrix

|  | Predicted Non-repeating | Predicted Repeating |
|---|---:|---:|
| Actual Non-repeating | 747 | 6 |
| Actual Repeating | 21 | 235 |

The model produces only **6 false positives** and **21 false negatives** on the 1,009-record test set.

---

## 🔁 Ten-Fold Cross-Validation

| Model | Mean F1 | Standard Deviation |
|---|---:|---:|
| Random Forest | 0.8899 | 0.0200 |
| Extra Trees | 0.8795 | 0.0332 |
| XGBoost | 0.8876 | 0.0220 |
| **LightGBM** | **0.9184** | **0.0147** |
| CatBoost | 0.9117 | 0.0201 |

LightGBM obtains both the highest mean F1-score and the lowest variation across folds.

---

## 📐 Statistical Model Comparison

A Friedman test is used to determine whether the five classifiers produce significantly different cross-validation results.

| Test | Statistic | p-value | Result |
|---|---:|---:|---|
| Friedman test | 25.84 | 0.0000341 | Significant difference exists |

Because the Friedman test is significant at `α = 0.05`, pairwise Wilcoxon signed-rank tests are also performed.

The notebook finds statistically significant differences between LightGBM and:

- Random Forest
- Extra Trees
- XGBoost

The difference between **LightGBM and CatBoost is not statistically significant** at the 0.05 level.

---

## 🔍 Model Explainability

The best LightGBM model is interpreted using SHAP.

The notebook generates:

- SHAP beeswarm plot
- SHAP dependence plot
- SHAP force plot
- SHAP waterfall plot
- Mean absolute SHAP feature-importance table
- Built-in LightGBM feature-importance chart

### Leading SHAP Features

| Rank | Feature | Mean Absolute SHAP Importance |
|---:|---|---:|
| 1 | `sp_run` | 1.1488 |
| 2 | `sp_idx` | 1.0407 |
| 3 | `dm_exc_ne2001` | 0.9046 |
| 4 | `bc_width` | 0.9020 |
| 5 | `dm_exc_ymw16` | 0.8579 |
| 6 | `dm_fitb` | 0.7041 |
| 7 | `gl` | 0.6835 |
| 8 | `bonsai_dm` | 0.6138 |
| 9 | `dec` | 0.6052 |
| 10 | `scat_time` | 0.4480 |

These results indicate that spectral behaviour, excess dispersion measure, burst width, fitted dispersion measure, and sky location contribute strongly to the model's classifications.

---

## 🔄 Project Workflow

```text
CHIME/FRB Catalogue Dataset
              │
              ▼
Data Inspection and Missing-Value Analysis
              │
              ▼
Binary Target Creation
              │
              ▼
Leakage Prevention
              │
              ▼
Physical Feature Selection
              │
              ▼
Stratified Train-Test Split
              │
              ▼
Training-Only Median Imputation
              │
              ▼
Core vs Expanded Feature Comparison
              │
              ▼
Optuna Hyperparameter Optimization
              │
              ▼
Five-Model Training and Evaluation
              │
              ▼
Cross-Validation and Statistical Tests
              │
              ▼
LightGBM Selection
              │
              ▼
SHAP Interpretation and Event Predictions
```

---

## 🛠️ Technology Stack

- Python
- Jupyter Notebook / Google Colab
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Optuna
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- SHAP
- SciPy

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/frb-repeater-classification.git
cd frb-repeater-classification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn optuna xgboost lightgbm catboost shap scipy jupyter
```

Alternatively, create a `requirements.txt` file and run:

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place the dataset in the repository root or inside a `data` directory.

The notebook currently loads:

```python
df = pd.read_csv("chimefrbcat2_dataset.csv")
```

When storing the dataset inside `data/`, change it to:

```python
df = pd.read_csv("data/chimefrbcat2_dataset.csv")
```

### 5. Run the Notebook

```bash
jupyter notebook
```

Open the notebook and execute the cells in order.

---

## 📁 Recommended Repository Structure

```text
frb-repeater-classification/
│
├── README.md
├── FRB_Repeater_Classification.ipynb
├── requirements.txt
│
├── data/
│   └── chimefrbcat2_dataset.csv
│
├── results/
│   ├── Feature_Set_Comparison.csv
│   ├── Model_Comparison.csv
│   ├── Ranked_Models.csv
│   ├── CrossValidation_Statistics.csv
│   ├── Wilcoxon_Test.csv
│   ├── FRB_Event_Predictions.csv
│   ├── Best_Model_Feature_Importance.csv
│   └── SHAP_Feature_Importance.csv
│
└── figures/
    ├── confusion_matrices/
    ├── roc_curves/
    ├── model_comparisons/
    └── shap_plots/
```

For a cleaner repository, rename:

```text
Untitled116_updated_FRB (1).ipynb
```

to:

```text
FRB_Repeater_Classification.ipynb
```

---

## 📦 Generated Output Files

| File | Description |
|---|---|
| `Feature_Set_Comparison.csv` | Comparison of core and expanded feature sets |
| `Model_Comparison.csv` | Test metrics for all classifiers |
| `Ranked_Models.csv` | Models ranked by F1-score |
| `FRB_Event_Predictions.csv` | Event-level predictions and repeater probabilities |
| `Best_Model_Feature_Importance.csv` | Native feature importance from the selected model |
| `CrossValidation_Statistics.csv` | Mean and standard deviation of cross-validation F1-scores |
| `Wilcoxon_Test.csv` | Pairwise Wilcoxon signed-rank test results |
| `SHAP_Feature_Importance.csv` | Mean absolute SHAP importance values |

---

## ⚠️ Limitations

- The target is inferred from the availability of `repeater_name` in the catalogue.
- A source classified as non-repeating may repeat in future observations.
- The dataset is imbalanced toward non-repeating events.
- Missing measurements are replaced using median imputation.
- The reported results are based on one stratified hold-out split.
- Hyperparameter searches use 10 Optuna trials per model and may be improved with larger search budgets.
- Catalogue selection effects and observational biases may influence the learned patterns.
- The model should support scientific analysis rather than replace expert astronomical validation.

---

## 🔮 Future Improvements

- Increase the number of Optuna optimization trials.
- Use repeated or nested stratified cross-validation.
- Group events by astronomical source to reduce source-level leakage.
- Explore calibrated classification probabilities.
- Apply imbalance-aware methods and threshold optimization.
- Add uncertainty estimates for individual classifications.
- Compare feature-selection and dimensionality-reduction methods.
- Evaluate neural-network and stacking-based ensemble approaches.
- Package the trained model in an interactive Streamlit or Flask application.
- Validate performance on an independent FRB catalogue.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit the changes with a clear message.
5. Push the branch to your fork.
6. Open a pull request.

---

## 📄 License

Add a suitable open-source license before publishing the repository. The MIT License is a common option for educational machine learning projects.

---

## ⭐ Support

If you find this project useful, consider starring the repository.

