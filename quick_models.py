"""Quick preliminary model training and visualization"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('data/processed/train.csv')
test = pd.read_csv('data/processed/test.csv')

# Selected features
selected_features = [
    'ArgUrlRatio', 'Arguments_LongestWordLength', 'CharacterContinuityRate',
    'Domain_LongestWordLength', 'Entropy_Domain', 'NumberRate_Extension',
    'NumberRate_FileName', 'NumberRate_URL', 'NumberofDotsinURL',
    'SymbolCount_Domain', 'SymbolCount_FileName', 'argPathRatio',
    'avgdomaintokenlen', 'avgpathtokenlen', 'domainUrlRatio', 'domainlength',
    'longdomaintokenlen', 'spcharUrl'
]

# Prepare data
X_train = train[selected_features]
y_train = train['URL_Type_obf_Type']
X_test = test[selected_features]
y_test = test['URL_Type_obf_Type']

# Train models
models = {
    'K-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=1234)
}

predictions = {}
f1_scores = {}

print("Training models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    # Calculate F1 scores (macro average for multi-class)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_scores[name] = {'macro': f1_macro, 'weighted': f1_weighted}
    
    print(f"\n{name}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

# Create confusion matrix heatmaps
class_labels = ['Spam', 'Phishing', 'Malware', 'Defacement']

for name, y_pred in predictions.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_labels, yticklabels=class_labels, cbar=True)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    
    filename = f"output/confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filename}")

# Create F1 score comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
models_list = list(f1_scores.keys())
macro_scores = [f1_scores[m]['macro'] for m in models_list]
weighted_scores = [f1_scores[m]['weighted'] for m in models_list]

x = np.arange(len(models_list))
width = 0.35

ax.bar(x - width/2, macro_scores, width, label='F1 (Macro)', color='steelblue')
ax.bar(x + width/2, weighted_scores, width, label='F1 (Weighted)', color='coral')

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Model Comparison - F1 Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(macro_scores):
    ax.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
for i, v in enumerate(weighted_scores):
    ax.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('output/f1_scores_comparison.png', dpi=300, bbox_inches='tight')
print("F1 scores chart saved to output/f1_scores_comparison.png")

print("\nPreliminary model evaluation complete!")
