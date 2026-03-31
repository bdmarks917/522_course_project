"""Select optimal features using various methods (filter, wrapper, embedded)."""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
	SelectKBest, SelectPercentile, RFE, RFECV,
	mutual_info_classif, chi2, f_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from .feature_analyzer import calculate_feature_importance_scores, analyze_correlations


class FeatureSelector:
	"""Comprehensive feature selection using multiple strategies."""

	def __init__(self, random_state: int = 42):
		self.random_state = random_state
		self.selected_features = {}
		self.selection_results = {}

	def select_by_correlation_filter(self, X: pd.DataFrame, y: pd.Series,
								   corr_threshold: float = 0.95) -> List[str]:
		"""Filter method: Remove highly correlated features.

		Parameters:
			X: Feature matrix
			y: Target vector
			corr_threshold: Correlation threshold for feature removal

		Returns:
			List of selected feature names
		"""
		corr_analysis = analyze_correlations(X, y, corr_threshold)

		# Start with all features
		selected = list(X.columns)

		# Remove features that are highly correlated with others
		# Keep the one with higher correlation to target
		high_corr_pairs = corr_analysis['high_corr_pairs']
		target_corr = corr_analysis['target_correlations']

		to_remove = set()
		for feat1, feat2, corr_val in high_corr_pairs:
			if feat1 in selected and feat2 in selected:
				# Keep the feature with higher correlation to target
				corr1 = target_corr.get(feat1, 0)
				corr2 = target_corr.get(feat2, 0)
				if corr1 >= corr2:
					to_remove.add(feat2)
				else:
					to_remove.add(feat1)

		selected = [f for f in selected if f not in to_remove]

		self.selected_features['correlation_filter'] = selected
		self.selection_results['correlation_filter'] = {
			'method': 'Correlation Filter',
			'threshold': corr_threshold,
			'removed_pairs': len(high_corr_pairs),
			'features_removed': len(to_remove),
			'features_selected': len(selected)
		}

		return selected

	def select_by_univariate_tests(self, X: pd.DataFrame, y: pd.Series,
								 k: int = 50, method: str = 'mutual_info') -> List[str]:
		"""Filter method: Select top k features using univariate statistical tests.

		Parameters:
			X: Feature matrix
			y: Target vector
			k: Number of features to select
			method: Selection method ('mutual_info', 'chi2', 'f_test')

		Returns:
			List of selected feature names
		"""
		method_map = {
			'mutual_info': mutual_info_classif,
			'chi2': chi2,
			'f_test': f_classif
		}

		if method not in method_map:
			raise ValueError(f"Method must be one of {list(method_map.keys())}")

		# For chi-squared, ensure non-negative values
		if method == 'chi2':
			scaler = MinMaxScaler()
			X_scaled = pd.DataFrame(
				scaler.fit_transform(X),
				columns=X.columns,
				index=X.index
			)
		else:
			X_scaled = X

		# Select top k features
		selector = SelectKBest(score_func=method_map[method], k=min(k, len(X.columns)))
		selector.fit(X_scaled, y)

		# Get selected feature names
		selected_mask = selector.get_support()
		selected = X.columns[selected_mask].tolist()

		self.selected_features['univariate_filter'] = selected
		self.selection_results['univariate_filter'] = {
			'method': f'Univariate Filter ({method})',
			'k': k,
			'features_selected': len(selected),
			'scores': dict(zip(X.columns, selector.scores_))
		}

		return selected

	def select_by_rfe(self, X: pd.DataFrame, y: pd.Series,
					 estimator=None, n_features: int = 30,
					 cv: int = 3) -> List[str]:
		"""Wrapper method: Recursive Feature Elimination with cross-validation.

		Parameters:
			X: Feature matrix
			y: Target vector
			estimator: Base estimator (default: RandomForest)
			n_features: Number of features to select
			cv: Number of CV folds

		Returns:
			List of selected feature names
		"""
		if estimator is None:
			estimator = RandomForestClassifier(
				n_estimators=20,
				random_state=self.random_state,
				n_jobs=1,
				max_depth=5
			)

		# Use standard RFE (non-CV) for faster execution
		rfe = RFE(
			estimator=estimator,
			n_features_to_select=min(n_features, X.shape[1]),
			step=max(1, int(X.shape[1] * 0.15)),
		)
		rfe.fit(X, y)
		selected_mask = rfe.support_
		selected = X.columns[selected_mask].tolist()

		# Use CV score or fallback to importance if available
		try:
			cv_folds = max(2, min(5, cv))
			cv_splitter = StratifiedKFold(cv_folds, shuffle=True, random_state=self.random_state)
			scores = []
			for train_idx, test_idx in cv_splitter.split(X, y):
				estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
				scores.append(estimator.score(X.iloc[test_idx], y.iloc[test_idx]))
			cv_scores = np.mean(scores)
		except Exception:
			cv_scores = None

		# If we got more than requested, take top n_features by importance
		if len(selected) > n_features:
			# Get feature importances from the fitted estimator
			if hasattr(estimator, 'feature_importances_'):
				importances = dict(zip(X.columns[selected_mask], estimator.feature_importances_))
				selected = sorted(importances.keys(), key=lambda x: importances[x], reverse=True)[:n_features]
			else:
				# Fallback: take first n_features
				selected = selected[:n_features]

		self.selected_features['rfe'] = selected
		self.selection_results['rfe'] = {
			'method': 'Recursive Feature Elimination (RFE)',
			'cv_folds': cv_folds,
			'final_features': len(selected),
			'cv_score': float(cv_scores) if cv_scores is not None else None
		}

		return selected

	def select_by_embedded_method(self, X: pd.DataFrame, y: pd.Series,
								n_features: int = 30) -> List[str]:
		"""Embedded method: Feature importance from tree-based model.

		Parameters:
			X: Feature matrix
			y: Target vector
			n_features: Number of features to select

		Returns:
			List of selected feature names
		"""
		# Train Random Forest to get feature importances
		rf = RandomForestClassifier(
			n_estimators=200,
			random_state=self.random_state,
			n_jobs=-1
		)
		rf.fit(X, y)

		# Get feature importances
		importances = dict(zip(X.columns, rf.feature_importances_))

		# Select top n_features
		selected = sorted(importances.keys(), key=lambda x: importances[x], reverse=True)[:n_features]

		self.selected_features['embedded'] = selected
		self.selection_results['embedded'] = {
			'method': 'Embedded (Random Forest Importance)',
			'features_selected': len(selected),
			'top_importances': {feat: importances[feat] for feat in selected[:5]}
		}

		return selected

	def select_comprehensive(self, X: pd.DataFrame, y: pd.Series,
						   n_features: int = 30) -> Dict[str, List[str]]:
		"""Run comprehensive feature selection using multiple methods.

		Parameters:
			X: Feature matrix
			y: Target vector
			n_features: Target number of features to select

		Returns:
			Dictionary with selected features from different methods
		"""
		print("🔍 Running comprehensive feature selection...")

		# Method 1: Correlation filter
		print("  📊 Correlation filter...")
		corr_features = self.select_by_correlation_filter(X, y)
		print(f"    Selected {len(corr_features)} features after correlation filtering")

		# Method 2: Univariate filter
		print("  📈 Univariate statistical tests...")
		univariate_features = self.select_by_univariate_tests(
			X[corr_features], y, k=n_features*2, method='mutual_info'
		)
		print(f"    Selected {len(univariate_features)} features by mutual information")

		# Method 3: RFE
		print("  🔄 Recursive Feature Elimination...")
		rfe_features = self.select_by_rfe(X[univariate_features], y, n_features=n_features)
		print(f"    Selected {len(rfe_features)} features by RFE")

		# Method 4: Embedded method
		print("  🌳 Embedded feature importance...")
		embedded_features = self.select_by_embedded_method(X, y, n_features=n_features)
		print(f"    Selected {len(embedded_features)} features by embedded method")

		# Consensus selection: features that appear in multiple methods
		all_selected = {
			'correlation_filter': set(corr_features),
			'univariate_filter': set(univariate_features),
			'rfe': set(rfe_features),
			'embedded': set(embedded_features)
		}

		# Find intersection (features selected by at least 2 methods)
		consensus = set.intersection(*all_selected.values())
		if not consensus:
			# If no consensus, take union of top methods
			consensus = set(rfe_features + embedded_features)

		consensus_features = sorted(list(consensus))

		self.selected_features['consensus'] = consensus_features
		self.selection_results['consensus'] = {
			'method': 'Consensus (intersection of methods)',
			'features_selected': len(consensus_features),
			'method_agreement': len(consensus)
		}

		print(f"  ✅ Consensus selection: {len(consensus_features)} features")

		return {
			'correlation_filter': corr_features,
			'univariate_filter': univariate_features,
			'rfe': rfe_features,
			'embedded': embedded_features,
			'consensus': consensus_features
		}

	def get_selection_summary(self) -> pd.DataFrame:
		"""Get summary of all feature selection results."""
		summary_data = []
		for method, results in self.selection_results.items():
			summary_data.append({
				'method': results.get('method', method),
				'features_selected': results.get('features_selected', 0),
				'details': str(results)
			})

		return pd.DataFrame(summary_data)
"""Select optimal features using various methods (filter, wrapper, embedded)."""
