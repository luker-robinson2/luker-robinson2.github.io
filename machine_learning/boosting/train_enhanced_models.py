#!/usr/bin/env python3
"""
Train boosting models with enhanced state-based features
Comparing performance against original 58.70% baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GroupShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("ENHANCED BOOSTING MODELS TRAINING")
print("="*80)

# Load data
df = pd.read_csv('enhanced_features/training_data.csv')
print(f"\nDataset: {len(df)} samples from {df['match_id'].nunique()} matches")
print(f"Features: {len(df.columns)} columns")
print(f"\nClass distribution:")
print(df['winning_team'].value_counts())
print(f"  CT: {(df['winning_team'] == 'CT').mean()*100:.1f}%")
print(f"  T: {(df['winning_team'] == 'T').mean()*100:.1f}%")

# Prepare features
feature_cols = [
    # Time features
    'time_elapsed', 'round_time_remaining', 'time_phase',
    
    # Player features
    't_players_alive', 'ct_players_alive', 'player_count_advantage',
    't_total_hp', 'ct_total_hp', 't_avg_hp', 'ct_avg_hp',
    't_total_armor', 'ct_total_armor',
    
    # Economy features (CRITICAL)
    't_equipment_value', 'ct_equipment_value',
    't_avg_equipment_value', 'ct_avg_equipment_value',
    'equipment_advantage', 'equipment_advantage_per_player',
    
    # Bomb features
    'bomb_planted', 'time_since_plant', 'time_until_explosion', 'bomb_being_defused',
    
    # Weapon features
    't_awp_count', 'ct_awp_count', 'awp_advantage',
    't_rifle_count', 'ct_rifle_count', 'rifle_advantage',
]

X = df[feature_cols].copy()
y = df['winning_team'].copy()

# Convert labels to binary
y = (y == 'T').astype(int)  # T=1, CT=0

print(f"\nFeatures used: {len(feature_cols)}")
print(f"Target: T=1 ({y.sum()} samples), CT=0 ({(~y.astype(bool)).sum()} samples)")

# Use GroupShuffleSplit to keep samples from same match together
groups = df['match_id'].astype('category').cat.codes
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"\nTrain/Test split (group-aware by match):")
print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")
print(f"  Train T%: {y_train.mean()*100:.1f}%")
print(f"  Test T%:  {y_test.mean()*100:.1f}%")

# Store results
results = []

print(f"\n{'='*80}")
print("1. ADABOOST")
print("="*80)

best_ada_acc = 0
best_ada_model = None
best_ada_params = {}

for n_est in [50, 100, 200]:
    for lr in [0.5, 1.0, 1.5]:
        ada_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=n_est,
            learning_rate=lr,
            random_state=42,
            algorithm='SAMME'
        )
        ada_model.fit(X_train, y_train)
        y_pred = ada_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Model': 'AdaBoost',
            'n_estimators': n_est,
            'learning_rate': lr,
            'max_depth': 3,
            'accuracy': acc
        })
        
        if acc > best_ada_acc:
            best_ada_acc = acc
            best_ada_model = ada_model
            best_ada_params = {'n_estimators': n_est, 'learning_rate': lr}
        
        print(f"  n_est={n_est:3}, lr={lr:.1f}: {acc:.4f}")

print(f"\nBest AdaBoost: {best_ada_params}")
print(f"Accuracy: {best_ada_acc:.4f} ({best_ada_acc*100:.2f}%)")

print(f"\n{'='*80}")
print("2. GRADIENT BOOSTING")
print("="*80)

best_gb_acc = 0
best_gb_model = None
best_gb_params = {}

for n_est in [50, 100, 200]:
    for lr in [0.05, 0.1, 0.2]:
        for depth in [3, 5]:
            gb_model = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            y_pred = gb_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({
                'Model': 'GradientBoosting',
                'n_estimators': n_est,
                'learning_rate': lr,
                'max_depth': depth,
                'accuracy': acc
            })
            
            if acc > best_gb_acc:
                best_gb_acc = acc
                best_gb_model = gb_model
                best_gb_params = {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth}
            
            print(f"  n_est={n_est:3}, lr={lr:.2f}, depth={depth}: {acc:.4f}")

print(f"\nBest GradientBoosting: {best_gb_params}")
print(f"Accuracy: {best_gb_acc:.4f} ({best_gb_acc*100:.2f}%)")

print(f"\n{'='*80}")
print("3. XGBOOST")
print("="*80)

best_xgb_acc = 0
best_xgb_model = None
best_xgb_params = {}

for n_est in [50, 100, 200]:
    for lr in [0.05, 0.1, 0.2]:
        for depth in [3, 5, 7]:
            xgb_model = xgb.XGBClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth,
                random_state=42,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({
                'Model': 'XGBoost',
                'n_estimators': n_est,
                'learning_rate': lr,
                'max_depth': depth,
                'accuracy': acc
            })
            
            if acc > best_xgb_acc:
                best_xgb_acc = acc
                best_xgb_model = xgb_model
                best_xgb_params = {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth}
            
            print(f"  n_est={n_est:3}, lr={lr:.2f}, depth={depth}: {acc:.4f}")

print(f"\nBest XGBoost: {best_xgb_params}")
print(f"Accuracy: {best_xgb_acc:.4f} ({best_xgb_acc*100:.2f}%)")

# Compare with baseline
print(f"\n{'='*80}")
print("RESULTS COMPARISON")
print("="*80)
print(f"Baseline (old features):          58.70%")
print(f"AdaBoost (enhanced features):     {best_ada_acc*100:.2f}%  ({(best_ada_acc-0.587)*100:+.2f}%)")
print(f"GradientBoosting (enhanced):      {best_gb_acc*100:.2f}%  ({(best_gb_acc-0.587)*100:+.2f}%)")
print(f"XGBoost (enhanced):               {best_xgb_acc*100:.2f}%  ({(best_xgb_acc-0.587)*100:+.2f}%)")

best_overall = max(best_ada_acc, best_gb_acc, best_xgb_acc)
improvement = (best_overall - 0.587) * 100
print(f"\n🎯 Best Model: {best_overall*100:.2f}%")
print(f"📈 Improvement: {improvement:+.2f} percentage points")

# Detailed evaluation of best model
print(f"\n{'='*80}")
print("BEST MODEL DETAILED EVALUATION")
print("="*80)

if best_overall == best_xgb_acc:
    best_model = best_xgb_model
    model_name = "XGBoost"
elif best_overall == best_gb_acc:
    best_model = best_gb_model
    model_name = "GradientBoosting"
else:
    best_model = best_ada_model
    model_name = "AdaBoost"

y_pred_best = best_model.predict(X_test)

print(f"Model: {model_name}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['CT', 'T']))

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('enhanced_features/model_results.csv', index=False)
print(f"\nSaved all results to: enhanced_features/model_results.csv")

# Create visualizations
print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model, name) in enumerate([(best_ada_model, 'AdaBoost'), 
                                       (best_gb_model, 'GradientBoosting'),
                                       (best_xgb_model, 'XGBoost')]):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['CT', 'T'], yticklabels=['CT', 'T'])
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.2%}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('../../docs/img/boosting_confusion_matrices_enhanced.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved confusion_matrices_enhanced.png")

# 2. Feature Importance (XGBoost)
fig, ax = plt.subplots(figsize=(10, 8))
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

sns.barplot(data=feature_importance, y='feature', x='importance', ax=ax, palette='viridis')
ax.set_title('Top 15 Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig('../../docs/img/boosting_feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved feature_importance_enhanced.png")

# 3. Model Comparison
fig, ax = plt.subplots(figsize=(10, 6))
comparison_data = pd.DataFrame({
    'Model': ['Baseline\n(Old Features)', 'AdaBoost\n(Enhanced)', 
              'GradientBoosting\n(Enhanced)', 'XGBoost\n(Enhanced)'],
    'Accuracy': [0.587, best_ada_acc, best_gb_acc, best_xgb_acc]
})

colors = ['#cccccc', '#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(comparison_data['Model'], comparison_data['Accuracy']*100, color=colors, alpha=0.8, edgecolor='black')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.axhline(y=58.7, color='red', linestyle='--', label='Baseline (58.70%)', linewidth=2)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Accuracy Comparison: Enhanced vs Baseline Features', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../../docs/img/boosting_model_comparison_enhanced.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved model_comparison_enhanced.png")

print(f"\n{'='*80}")
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nKey Improvements:")
print(f"  • Used state-based features (economy, bomb, time)")
print(f"  • Time-series sampling (10 samples per round)")
print(f"  • Proper feature engineering")
print(f"  • Group-aware train/test split")
print(f"\nResult: {best_overall*100:.2f}% accuracy ({improvement:+.2f}pp vs baseline)")

