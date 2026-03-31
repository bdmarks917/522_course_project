import os
from pathlib import Path
import pandas as pd
import pytest

# Ensure repository root is on path for module imports
import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / 'src'))

from feature_selection.feature_selector import FeatureSelector


def load_train_data():
    base = repo_root
    csv_path = base / 'data' / 'processed' / 'train.csv'
    if not csv_path.exists():
        pytest.skip('Processed train.csv is not available. Run preprocessing first.')
    return pd.read_csv(csv_path)


def test_select_by_correlation_filter_reduces_features():
    df = load_train_data()
    target = 'URL_Type_obf_Type'
    X = df.drop(columns=[target])
    y = df[target]

    selector = FeatureSelector(random_state=42)
    selected = selector.select_by_correlation_filter(X, y, corr_threshold=0.95)

    assert isinstance(selected, list)
    assert 0 < len(selected) < X.shape[1]
    assert all(feat in X.columns for feat in selected)


def test_select_comprehensive_default_consensus():
    df = load_train_data()
    target = 'URL_Type_obf_Type'
    subsample = df.sample(n=min(2000, len(df)), random_state=42)
    X = subsample.drop(columns=[target])
    y = subsample[target]

    selector = FeatureSelector(random_state=42)
    results = selector.select_comprehensive(X, y, n_features=20)

    assert 'consensus' in results
    assert isinstance(results['consensus'], list)
    assert len(results['consensus']) > 0
    assert results['consensus'][0] in X.columns


def test_feature_selection_summary_has_methods():
    df = load_train_data()
    target = 'URL_Type_obf_Type'
    subsample = df.sample(n=min(2000, len(df)), random_state=42)
    X = subsample.drop(columns=[target])
    y = subsample[target]

    selector = FeatureSelector(random_state=42)
    selector.select_comprehensive(X, y, n_features=20)
    summary_df = selector.get_selection_summary()

    assert 'method' in summary_df.columns
    assert 'features_selected' in summary_df.columns
    assert len(summary_df) >= 4
