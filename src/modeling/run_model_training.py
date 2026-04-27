"""Run model training and hyperparameter tuning pipeline."""

from pathlib import Path
from sklearn import model_selection, pipeline, ensemble, neighbors, tree, metrics
import pandas as pd
import optuna
from . import hyperparameter_tuner as tuner

# Load processed data
print(" Loading processed training data...")
project_root = Path(__file__).parent.parent.parent
train_path = project_root / "output" / "feature_selection" / "train_consensus_features.csv"
if not train_path.exists():
    raise FileNotFoundError(f"Training data not found at {train_path}")

df = pd.read_csv(train_path)
print(f"  Loaded {len(df)} training samples with {len(df.columns)-1} features")

# Create dataset splits for evaluating models against each other
# 60/20/20 - training/val/test
# The test set has already been set aside, so we only need to split the training data into training and validation sets
X = df.drop("URL_Type_obf_Type", axis=1)
y = df["URL_Type_obf_Type"]
X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.25, stratify=y, random_state=1234)

# Tune hyperparameters using Optuna
knn_study = optuna.create_study(direction='maximize')
knn_study.optimize(tuner.ObjectiveKnn(X_train, X_val, y_train, y_val), timeout=300, show_progress_bar=True)
print("Best KNN hyperparameters:", knn_study.best_params)

dt_study = optuna.create_study(direction='maximize')
dt_study.optimize(tuner.ObjectiveDt(X_train, X_val, y_train, y_val), timeout=300, show_progress_bar=True)
print("Best Decision Tree hyperparameters:", dt_study.best_params)

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(tuner.ObjectiveRf(X_train, X_val, y_train, y_val), timeout=300, show_progress_bar=True)
print("Best Random Forest hyperparameters:", rf_study.best_params)

# Create models with best hyperparameters
knn_model = neighbors.KNeighborsClassifier(
    n_neighbors=knn_study.best_params['n_neighbors'],
    weights=knn_study.best_params['weights'],
    metric=knn_study.best_params['metric']
)

decision_tree_model = tree.DecisionTreeClassifier(
    criterion=dt_study.best_params['criterion'],
    ccp_alpha=dt_study.best_params['ccp_alpha'],
    max_depth=dt_study.best_params['max_depth'],
    random_state=1234
)

random_forest_model = ensemble.RandomForestClassifier(
    n_estimators=rf_study.best_params['n_estimators'],
    max_depth=rf_study.best_params['max_depth'],
    min_samples_leaf=rf_study.best_params['min_samples_leaf'],
    max_features=rf_study.best_params['max_features'],
    bootstrap=rf_study.best_params['bootstrap'],
    random_state=1234
)

# Define models
models = {
    "KNN": knn_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

# Validation Evaluation
performance_results = {}
val_predictions = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_val_pred = model.predict(X_val)
    val_predictions[name] = y_val_pred
    
    # Probabilities (for ROC AUC, etc.)
    y_val_proba = model.predict_proba(X_val)
    
    # Metrics
    performance_results[name] = {
        "accuracy": metrics.accuracy_score(y_val, y_val_pred),
        "precision": metrics.precision_score(y_val, y_val_pred, zero_division=0, average='weighted'),
        "recall": metrics.recall_score(y_val, y_val_pred, zero_division=0, average='weighted'),
        "f1": metrics.f1_score(y_val, y_val_pred, zero_division=0, average='weighted'),
        "roc_auc": metrics.roc_auc_score(y_val, y_val_proba, average='weighted', multi_class='ovr'),
        "average_precision": metrics.average_precision_score(y_val, y_val_proba, average='weighted')
    }

# Results table
performance_df = pd.DataFrame(performance_results).T
print("\nValidation Performance:")
print(performance_df)

# Detailed Reports
for name in models:
    print(f"\n{name} Classification Report:")
    print(metrics.classification_report(y_val, val_predictions[name]))
    
    print(f"{name} Confusion Matrix:")
    print(metrics.confusion_matrix(y_val, val_predictions[name]))


# # Illustrate results
# plot_performances([
#     ('KNN', knn_cv_predict_results),
#     ('Decision Tree', decision_tree_cv_predict_results),
#     ('Random Forest', random_forest_cv_predict_results)
# ])