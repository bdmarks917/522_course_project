import os
from pathlib import Path
import pandas as pd
import pytest

import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / 'src'))

from preprocessing.run_preprocessing import main as preprocessing_main


def test_preprocessing_creates_processed_files(tmp_path):
    """Run pipeline and verify expected processed files are created."""
    # Run pipeline (it writes to data/processed by default)
    data_dir = repo_root / 'data' / 'processed'
    # Remove existing outputs to get clean test-run behavior (if safe)
    train_file = data_dir / 'train.csv'
    test_file = data_dir / 'test.csv'
    all_file = data_dir / 'All_processed.csv'

    # Operation:
    processing_result = preprocessing_main()

    assert processing_result == data_dir
    assert train_file.exists()
    assert test_file.exists()
    assert all_file.exists()

    # Check shape and target column
    train_df = pd.read_csv(train_file)
    assert 'URL_Type_obf_Type' in train_df.columns
    assert len(train_df) > 0
