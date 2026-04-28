# 522_course_project
Code repository for the CSC 522 Automated Learning and Data Analysis course project

## Project Overview

This project focuses on **malicious URL classification**, aiming to develop a machine learning model capable of identifying different types of malicious websites including defacement, malware, phishing, and spam attacks. The dataset contains over 30,000 URLs with 87 engineered features capturing various URL characteristics such as character composition, token distribution, entropy measures, and structural patterns.

**Project Status: Complete** ✅ - All phases implemented with automated testing.

## Project Pipeline

The project is structured into four main phases:

1. **Data Preprocessing**: Load and combine multi-class datasets, handle missing values (represented as -1, NaN, and infinite values), drop low-variance features, normalize numerical features with StandardScaler, and perform stratified train/test split.

2. **Feature Selection & Analysis**: Conduct exploratory data analysis (EDA) to understand feature distributions and correlations, apply multiple feature selection methods (filter-based, wrapper-based, and embedded approaches), and identify the most discriminative features for malicious URL classification.

3. **Model Training & Hyperparameter Tuning**: Train multiple machine learning classifiers (KNN, Decision Tree, Random Forest) using Optuna for automated hyperparameter optimization. Evaluate models on validation set with comprehensive metrics including accuracy, F1-score, ROC-AUC, and confusion matrices.

4. **Evaluation & Reporting**: Evaluate models using appropriate multi-class metrics (accuracy, precision, recall, F1-score, confusion matrix), compare different approaches, and generate comprehensive performance reports.

## Project Structure

```
├── data/                          # Raw datasets and processed outputs
│   ├── raw/                      # Copy of original input CSVs (preserved)
│   └── processed/                # Cleaned data used by modeling pipeline
│       ├── All_processed.csv     # Fully processed dataset (scaled, imputed)
│       ├── train.csv             # Training set (80% of data)
│       └── test.csv              # Test set (20% of data)
├── src/                          # Source code modules
│   ├── preprocessing/            # Data loading and cleaning
│   ├── feature_selection/        # Feature analysis and selection
│   └── modeling/                 # Model training and tuning
├── output/                       # Generated reports and results
└── documentation/                # Project proposal and specifications
```

## Modeling Module

The `src/modeling/` directory contains components for training and evaluating machine learning models:

- `hyperparameter_tuner.py`: Optuna-based hyperparameter optimization for KNN, Decision Tree, and Random Forest classifiers
- `run_model_training.py`: Main script that loads selected features, performs hyperparameter tuning, trains models, and evaluates performance
- `model_trainer.py`: Placeholder for additional model training utilities

Models are trained on consensus-selected features from the feature selection phase. Hyperparameter tuning uses Optuna with weighted F1-score as the optimization objective. Performance is evaluated on a validation set with metrics including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

1. Install dependencies: `pip install -r requirements.txt`
2. Run preprocessing: `python -m src.preprocessing.run_preprocessing`
3. Confirm output at `data/processed/` (All_processed.csv, train.csv, test.csv)
4. Run feature selection: `python -m src.feature_selection.run_feature_selection`
5. Train and evaluate models: `python -m src.modeling.run_model_training`

## Testing

This repository uses `pytest` for automated tests. Run from the project root:

```bash
pip install -r requirements.txt
pytest -q
```

For targeted execution:

```bash
pytest -q tests/test_preprocessing.py
pytest -q tests/test_feature_selector.py
pytest -q tests/test_modeling.py
```

## Target Values

`{'benign': 0, 'Defacement': 1, 'malware': 2, 'phishing': 3, 'spam': 4}`
