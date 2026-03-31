# 522_course_project
Code repository for the CSC 522 Automated Learning and Data Analysis course project

## Project Overview

This project focuses on **malicious URL classification**, aiming to develop a machine learning model capable of identifying different types of malicious websites including defacement, malware, phishing, and spam attacks. The dataset contains over 30,000 URLs with 87 engineered features capturing various URL characteristics such as character composition, token distribution, entropy measures, and structural patterns.

## Project Pipeline

The project is structured into four main phases:

1. **Data Preprocessing**: Load and combine multi-class datasets, handle missing values (represented as -1 and NaN), normalize numerical features, and prepare data for analysis.

2. **Feature Selection & Analysis**: Conduct exploratory data analysis (EDA) to understand feature distributions and correlations, apply multiple feature selection methods (filter-based, wrapper-based, and embedded approaches), and identify the most discriminative features for malicious URL classification.

3. **Model Training & Development**: Build and train multiple machine learning classifiers (e.g., Random Forest, SVM, Gradient Boosting), tune hyperparameters, and perform cross-validation to ensure robust model performance.

4. **Evaluation & Reporting**: Evaluate models using appropriate multi-class metrics (accuracy, precision, recall, F1-score, confusion matrix), compare different approaches, and generate comprehensive performance reports.

## Project Structure

```
├── data/                          # Raw datasets (CSV files)
├── src/                          # Source code modules
│   ├── preprocessing/            # Data loading and cleaning
│   ├── feature_selection/        # Feature analysis and selection
│   ├── modeling/                 # Model training and tuning
│   └── evaluation/               # Performance metrics
├── notebooks/                    # Jupyter notebooks for exploration
├── output/                       # Generated reports and results
└── documentation/                # Project proposal and specifications
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Start with preprocessing notebooks to explore and clean the data
3. Run feature selection analysis to identify important features
4. Train and evaluate models using the selected features
