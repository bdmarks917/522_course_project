"""Run comprehensive feature selection pipeline."""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from feature_selection.feature_selector import FeatureSelector
from feature_selection.feature_analyzer import analyze_correlations, calculate_feature_importance_scores
from preprocessing.data_loader import save_csv
import pandas as pd


def main():
    """Main feature selection pipeline."""
    print("🚀 Starting Feature Selection Pipeline")
    print("=" * 50)

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "processed"
    output_dir = project_root / "output" / "feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    print("📂 Loading processed training data...")
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    print(f"  Loaded {len(train_df)} training samples with {len(train_df.columns)-1} features")

    # Separate features and target
    target_col = 'URL_Type_obf_Type'
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print(f"  Features: {X_train.shape[1]}, Target classes: {y_train.nunique()}")

    # Initialize feature selector
    selector = FeatureSelector(random_state=42)

    # Run comprehensive feature selection
    print("\n🔍 Running comprehensive feature selection...")
    selection_results = selector.select_comprehensive(X_train, y_train, n_features=30)

    # Save selection results
    print("\n💾 Saving feature selection results...")

    # Save selected features for each method
    for method, features in selection_results.items():
        features_df = pd.DataFrame({'feature': features})
        features_df.to_csv(output_dir / f"selected_features_{method}.csv", index=False)
        print(f"  Saved {len(features)} features for {method}")

    # Save selection summary
    summary_df = selector.get_selection_summary()
    summary_df.to_csv(output_dir / "feature_selection_summary.csv", index=False)

    # Create feature selection report
    print("\n📊 Generating feature selection report...")

    report = {
        'dataset_info': {
            'total_samples': len(train_df),
            'total_features': X_train.shape[1],
            'target_classes': int(y_train.nunique()),
            'class_distribution': y_train.value_counts().to_dict()
        },
        'selection_methods': {}
    }

    for method, features in selection_results.items():
        report['selection_methods'][method] = {
            'features_selected': len(features),
            'feature_names': features,
            'details': selector.selection_results.get(method, {})
        }

    # Save report as JSON
    with open(output_dir / "feature_selection_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Run feature analysis on consensus features
    print("\n📈 Analyzing consensus features...")

    consensus_features = selection_results.get('consensus', [])
    if consensus_features:
        X_consensus = X_train[consensus_features]

        # Analyze correlations
        corr_analysis = analyze_correlations(X_consensus, y_train)
        corr_df = pd.DataFrame({
            'feature1': [p[0] for p in corr_analysis['high_corr_pairs']],
            'feature2': [p[1] for p in corr_analysis['high_corr_pairs']],
            'correlation': [p[2] for p in corr_analysis['high_corr_pairs']]
        })
        corr_df.to_csv(output_dir / "consensus_correlations.csv", index=False)

        # Calculate importance scores
        importance_scores = calculate_feature_importance_scores(X_consensus, y_train)
        importance_df = importance_scores.reset_index().rename(columns={'index': 'feature'})
        importance_df = importance_df[['feature', 'mutual_info', 'chi_squared', 'f_test', 'rf_importance']]
        importance_df.to_csv(output_dir / "consensus_importance_scores.csv", index=False)

        # Create final dataset with consensus features
        final_train_df = X_consensus.copy()
        final_train_df[target_col] = y_train
        save_csv(final_train_df, output_dir / "train_consensus_features.csv")

        print(f"  Created final training set with {len(consensus_features)} consensus features")
    else:
        print("  ⚠️  No consensus features found, using RFE features as fallback")
        rfe_features = selection_results.get('rfe', [])
        if rfe_features:
            X_rfe = X_train[rfe_features]
            final_train_df = X_rfe.copy()
            final_train_df[target_col] = y_train
            save_csv(final_train_df, output_dir / "train_rfe_features.csv")

    # Load and process test data with selected features
    print("\n🔄 Processing test data with selected features...")
    test_path = data_dir / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Apply consensus features to test data
        if consensus_features:
            X_test_consensus = X_test[consensus_features]
            test_consensus_df = X_test_consensus.copy()
            test_consensus_df[target_col] = y_test
            save_csv(test_consensus_df, output_dir / "test_consensus_features.csv")
            print(f"  Processed test data with {len(consensus_features)} consensus features")
        elif rfe_features:
            X_test_rfe = X_test[rfe_features]
            test_rfe_df = X_test_rfe.copy()
            test_rfe_df[target_col] = y_test
            save_csv(test_rfe_df, output_dir / "test_rfe_features.csv")
            print(f"  Processed test data with {len(rfe_features)} RFE features")

    print("\n✅ Feature selection pipeline completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("\n📋 Summary:")
    for method, features in selection_results.items():
        print(f"  {method}: {len(features)} features")

    # Print top consensus features
    if consensus_features:
        print(f"\n🎯 Top {min(10, len(consensus_features))} consensus features:")
        for i, feature in enumerate(consensus_features[:10], 1):
            print(f"  {i}. {feature}")


if __name__ == "__main__":
    main()