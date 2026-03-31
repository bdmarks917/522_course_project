"""Handle missing values, outliers, and data normalization."""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def drop_low_variance(df: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
	"""Drop columns where a single value occurs in at least threshold fraction of rows.

	Parameters:
		df: Source dataframe.
		threshold: Fraction of identical values at or above which the column is dropped.

	Returns:
		A tuple of (reduced dataframe, dropped_columns_list).
	"""

	if not 0 < threshold <= 1:
		raise ValueError("threshold must be between 0 and 1")

	nrows = len(df)
	drop_cols = []
	for col in df.columns:
		top_freq = df[col].value_counts(dropna=False, normalize=True).iloc[0]
		if top_freq >= threshold:
			drop_cols.append(col)

	reduced = df.drop(columns=drop_cols)
	return reduced, drop_cols


def summarize_column_variance(df: pd.DataFrame) -> pd.DataFrame:
	"""Return a DataFrame with the largest value frequency per column."""
	summary = []
	for col in df.columns:
		top = df[col].value_counts(dropna=False, normalize=True).iloc[0]
		summary.append({"column": col, "top_value_fraction": top})
	return pd.DataFrame(summary).sort_values("top_value_fraction", ascending=False)


def encode_target(df: pd.DataFrame, target_col: str = "URL_Type_obf_Type", benign_label: str = "benign") -> pd.DataFrame:
	"""Encode the target column to integer labels with benign as 0 when available."""

	if target_col not in df.columns:
		raise ValueError(f"target_col '{target_col}' not found in DataFrame")

	# preserve explicit mapping for benign if present, then others in sorted order.
	unique_vals = pd.Series(df[target_col].dropna().unique())
	labels = []
	if benign_label in unique_vals.values:
		labels.append(benign_label)
	labels += sorted([x for x in unique_vals.values if x != benign_label])

	mapping = {v: i for i, v in enumerate(labels)}

	# drop unknown values from map if any; they should be encoded as -1 with warning.
	encoded = df[target_col].map(mapping)
	if encoded.isna().any():
		missing = df[target_col][encoded.isna()].unique().tolist()
		print(f"Warning: target_col contains unmapped labels {missing}; they are set as -1")
		encoded = encoded.fillna(-1).astype(int)

	result = df.copy()
	result[target_col] = encoded
	result[f"{target_col}_mapping"] = str(mapping)
	return result


def impute_missing_values(df: pd.DataFrame, target_col: str = "URL_Type_obf_Type", strategy: str = "median") -> pd.DataFrame:
	"""Impute missing values (-1 and NaN) in numerical features using specified strategy.

	Parameters:
		df: Input dataframe
		target_col: Target column name (will not be imputed)
		strategy: Imputation strategy - 'median' (recommended for robustness) or 'mean'

	Returns:
		DataFrame with missing values imputed
	"""
	if strategy not in ['median', 'mean']:
		raise ValueError("strategy must be 'median' or 'mean'")

	result = df.copy()

	# Identify numerical columns (exclude target and mapping columns)
	exclude_cols = [target_col, f"{target_col}_mapping"]
	numerical_cols = [col for col in result.columns if col not in exclude_cols and result[col].dtype in ['int64', 'float64']]

	print(f"Imputing missing values in {len(numerical_cols)} numerical features using {strategy}")

	for col in numerical_cols:
		# Check for -1 values (common missing indicator in this dataset), NaN, and infinite values
		missing_mask = (result[col] == -1) | result[col].isna() | np.isinf(result[col])

		if missing_mask.any():
			# Use only finite, non-missing values for calculating imputation value
			valid_mask = ~(result[col].isna() | np.isinf(result[col]) | (result[col] == -1))
			valid_values = result.loc[valid_mask, col]

			if len(valid_values) == 0:
				# If no valid values, use 0 as fallback
				fill_value = 0.0
			else:
				if strategy == 'median':
					fill_value = valid_values.median()
				else:  # mean
					fill_value = valid_values.mean()

			result.loc[missing_mask, col] = fill_value
			print(f"  {col}: imputed {missing_mask.sum()} missing/infinite values with {fill_value:.4f}")

	return result


def scale_features(df: pd.DataFrame, target_col: str = "URL_Type_obf_Type", scaler=None) -> Tuple[pd.DataFrame, Any]:
	"""Scale numerical features using StandardScaler.

	Parameters:
		df: Input dataframe
		target_col: Target column name (will not be scaled)
		scaler: Pre-fitted scaler (optional, for applying same scaling to new data)

	Returns:
		Tuple of (scaled dataframe, fitted scaler)
	"""
	result = df.copy()

	# Identify numerical columns to scale (exclude target and mapping columns)
	exclude_cols = [target_col, f"{target_col}_mapping"]
	scale_cols = [col for col in result.columns if col not in exclude_cols and result[col].dtype in ['int64', 'float64']]

	print(f"Scaling {len(scale_cols)} numerical features with StandardScaler")

	if scaler is None:
		scaler = StandardScaler()

	result[scale_cols] = scaler.fit_transform(result[scale_cols])

	return result, scaler


def stratified_train_test_split(df: pd.DataFrame, target_col: str = "URL_Type_obf_Type",
							   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Perform stratified train/test split maintaining class proportions.

	Parameters:
		df: Input dataframe
		target_col: Target column name
		test_size: Proportion of data for test set
		random_state: Random seed for reproducibility

	Returns:
		Tuple of (X_train, X_test, y_train, y_test)
	"""
	if target_col not in df.columns:
		raise ValueError(f"target_col '{target_col}' not found in DataFrame")

	# Separate features and target
	X = df.drop([target_col, f"{target_col}_mapping"], axis=1)
	y = df[target_col]

	print(f"Stratified train/test split: {test_size:.1%} test size")
	print(f"Original class distribution: {y.value_counts().sort_index()}")

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, stratify=y, random_state=random_state
	)

	print(f"Train set: {len(X_train)} samples")
	print(f"Test set: {len(X_test)} samples")
	print(f"Train class distribution: {y_train.value_counts().sort_index()}")
	print(f"Test class distribution: {y_test.value_counts().sort_index()}")

	return X_train, X_test, y_train, y_test
