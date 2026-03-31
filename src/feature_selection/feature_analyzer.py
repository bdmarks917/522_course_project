"""Analyze feature importance and correlation."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_correlations(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95) -> Dict[str, List[str]]:
	"""Analyze feature correlations and identify highly correlated pairs.

	Parameters:
		X: Feature matrix
		y: Target vector
		threshold: Correlation threshold for identifying redundant features

	Returns:
		Dictionary with correlation analysis results
	"""
	# Correlation matrix
	corr_matrix = X.corr()

	# Find highly correlated pairs
	high_corr_pairs = []
	for i in range(len(corr_matrix.columns)):
		for j in range(i+1, len(corr_matrix.columns)):
			if abs(corr_matrix.iloc[i, j]) > threshold:
				high_corr_pairs.append((
					corr_matrix.columns[i],
					corr_matrix.columns[j],
					corr_matrix.iloc[i, j]
				))

	# Correlation with target (for numerical features)
	target_corr = {}
	for col in X.columns:
		if X[col].dtype in ['int64', 'float64']:
			corr = X[col].corr(y)
			target_corr[col] = abs(corr)  # Use absolute correlation

	return {
		'correlation_matrix': corr_matrix,
		'high_corr_pairs': high_corr_pairs,
		'target_correlations': pd.Series(target_corr).sort_values(ascending=False)
	}


def calculate_feature_importance_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
	"""Calculate multiple feature importance scores using different methods.

	Parameters:
		X: Feature matrix
		y: Target vector

	Returns:
		DataFrame with feature scores from different methods
	"""
	feature_names = X.columns.tolist()
	scores_df = pd.DataFrame(index=feature_names)

	# Method 1: Mutual Information
	try:
		mi_scores = mutual_info_classif(X, y, random_state=42)
		scores_df['mutual_info'] = mi_scores
	except Exception as e:
		print(f"Mutual information failed: {e}")
		scores_df['mutual_info'] = 0

	# Method 2: Chi-squared (need non-negative features)
	try:
		# Scale to [0,1] for chi-squared
		scaler = MinMaxScaler()
		X_scaled = scaler.fit_transform(X)
		chi_scores, _ = chi2(X_scaled, y)
		scores_df['chi_squared'] = chi_scores
	except Exception as e:
		print(f"Chi-squared failed: {e}")
		scores_df['chi_squared'] = 0

	# Method 3: ANOVA F-test
	try:
		f_scores, _ = f_classif(X, y)
		scores_df['f_test'] = f_scores
	except Exception as e:
		print(f"F-test failed: {e}")
		scores_df['f_test'] = 0

	# Method 4: Random Forest feature importance
	try:
		rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
		rf.fit(X, y)
		scores_df['rf_importance'] = rf.feature_importances_
	except Exception as e:
		print(f"Random Forest importance failed: {e}")
		scores_df['rf_importance'] = 0

	# Calculate average rank across methods
	rank_cols = ['mutual_info', 'chi_squared', 'f_test', 'rf_importance']
	for col in rank_cols:
		scores_df[f'{col}_rank'] = scores_df[col].rank(ascending=False)

	scores_df['avg_rank'] = scores_df[[f'{col}_rank' for col in rank_cols]].mean(axis=1)
	scores_df['avg_score'] = scores_df[rank_cols].mean(axis=1)

	return scores_df.sort_values('avg_rank')


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, top_n: int = 10,
							  save_path: str = None) -> None:
	"""Plot distributions of top features by importance.

	Parameters:
		X: Feature matrix
		y: Target vector
		top_n: Number of top features to plot
		save_path: Path to save plots (optional)
	"""
	# Get feature importance scores
	importance_df = calculate_feature_importance_scores(X, y)
	top_features = importance_df.head(top_n).index.tolist()

	# Create subplots
	n_cols = 3
	n_rows = (len(top_features) + n_cols - 1) // n_cols

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
	if n_rows == 1:
		axes = axes.reshape(1, -1)

	for i, feature in enumerate(top_features):
		row, col = i // n_cols, i % n_cols
		ax = axes[row, col]

		# Plot distribution by class
		for class_val in sorted(y.unique()):
			mask = y == class_val
			sns.histplot(X.loc[mask, feature], alpha=0.5, label=f'Class {class_val}', ax=ax)

		ax.set_title(f'{feature} (Rank: {i+1})')
		ax.legend()
		ax.set_xlabel(feature)
		ax.set_ylabel('Count')

	# Hide empty subplots
	for i in range(len(top_features), n_rows * n_cols):
		row, col = i // n_cols, i % n_cols
		axes[row, col].set_visible(False)

	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Feature distribution plots saved to: {save_path}")

	plt.show()
