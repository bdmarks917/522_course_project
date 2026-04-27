import os
from pathlib import Path
import pandas as pd
import pytest
import optuna

import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / 'src'))

from modeling.hyperparameter_tuner import ObjectiveKnn, ObjectiveDt, ObjectiveRf


def load_train_data():
    base = repo_root
    csv_path = base / 'output' / 'feature_selection' / 'train_consensus_features.csv'
    if not csv_path.exists():
        pytest.skip('Consensus features not available. Run feature selection first.')
    return pd.read_csv(csv_path)


def test_objective_knn_callable():
    """Test that ObjectiveKnn can be called with a trial."""
    df = load_train_data()
    target = 'URL_Type_obf_Type'
    subsample = df.sample(n=min(500, len(df)), random_state=42)
    X = subsample.drop(columns=[target])
    y = subsample[target]

    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    objective = ObjectiveKnn(X_train, X_val, y_train, y_val)

    # Create a mock trial
    study = optuna.create_study()
    trial = study.ask()

    # Call the objective
    score = objective(trial)
    assert isinstance(score, float)
    assert 0 <= score <= 1  # F1 score


def test_objective_dt_callable():
    """Test that ObjectiveDt can be called with a trial."""
    df = load_train_data()
    target = 'URL_Type_obf_Type'
    subsample = df.sample(n=min(500, len(df)), random_state=42)
    X = subsample.drop(columns=[target])
    y = subsample[target]

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    objective = ObjectiveDt(X_train, X_val, y_train, y_val)

    study = optuna.create_study()
    trial = study.ask()

    score = objective(trial)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_objective_rf_callable():
    """Test that ObjectiveRf can be called with a trial."""
    df = load_train_data()
    target = 'URL_Type_obf_Type'
    subsample = df.sample(n=min(500, len(df)), random_state=42)
    X = subsample.drop(columns=[target])
    y = subsample[target]

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    objective = ObjectiveRf(X_train, X_val, y_train, y_val)

    study = optuna.create_study()
    trial = study.ask()

    score = objective(trial)
    assert isinstance(score, float)
    assert 0 <= score <= 1