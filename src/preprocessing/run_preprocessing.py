"""Complete preprocessing pipeline for malicious URL classification.

This script handles the entire preprocessing workflow:
1. Data loading and low-variance feature removal
2. Missing value imputation
3. Feature scaling
4. Stratified train/test split

Usage:
    python -m src.preprocessing.run_preprocessing
    or from project root: python -c "from src.preprocessing.run_preprocessing import main; main()"
"""

from pathlib import Path
import shutil
import pandas as pd

from .data_loader import load_all_csv, save_csv
from .data_cleaner import (
    drop_low_variance,
    encode_target,
    summarize_column_variance,
    impute_missing_values,
    scale_features,
    stratified_train_test_split
)


def main():
    """Run the complete preprocessing pipeline."""
    project_root = Path(__file__).parent.parent.parent.resolve()
    data_root = project_root / "data"
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Ensure a raw copy is available and preserve original purity.
    source_all = raw_dir / "All.csv"
    if not source_all.exists():
        # Fallback to main data directory
        fallback = data_root / "All.csv"
        if not fallback.exists():
            raise FileNotFoundError(f"Source file not found in {raw_dir} or {data_root}")
        shutil.copy2(fallback, source_all)

    print(f"Loaded raw dataset: {source_all}")
    df = load_all_csv(str(source_all))
    print(f"Initial shape: {df.shape}")

    print("Top 20 columns with highest identical-value fraction:")
    print(summarize_column_variance(df).head(20).to_string(index=False))

    # Step 1: Drop low-variance features
    df, dropped = drop_low_variance(df, threshold=0.95)
    print(f"Dropped columns (>=95% same): {dropped}")
    print(f"After drop shape: {df.shape}")

    # Step 2: Encode target column with benign->0 if present
    df = encode_target(df, target_col="URL_Type_obf_Type", benign_label="benign")

    # Step 3: Impute missing values (-1, NaN, and infinite values)
    df = impute_missing_values(df, target_col="URL_Type_obf_Type", strategy="median")

    # Step 4: Scale numerical features
    df, scaler = scale_features(df, target_col="URL_Type_obf_Type")

    # Step 5: Save the fully processed dataset
    processed_all = processed_dir / "All_processed.csv"
    save_csv(df, str(processed_all), index=False)

    # Step 6: Perform stratified train/test split
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df, target_col="URL_Type_obf_Type", test_size=0.2
    )

    # Save train/test splits
    train_data = X_train.copy()
    train_data["URL_Type_obf_Type"] = y_train
    test_data = X_test.copy()
    test_data["URL_Type_obf_Type"] = y_test

    save_csv(train_data, str(processed_dir / "train.csv"), index=False)
    save_csv(test_data, str(processed_dir / "test.csv"), index=False)

    print(f"\n✅ Preprocessing complete!")
    print(f"📁 Processed dataset: {processed_all}")
    print(f"📁 Train set: {processed_dir / 'train.csv'}")
    print(f"📁 Test set: {processed_dir / 'test.csv'}")

    return processed_dir


if __name__ == "__main__":
    main()