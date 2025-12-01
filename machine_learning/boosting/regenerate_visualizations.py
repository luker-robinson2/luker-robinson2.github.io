#!/usr/bin/env python3
"""
Quick script to regenerate visualizations with updated results
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

print("Generating updated visualizations...")

# Updated results from full dataset
results = {
    'AdaBoost': 0.6890,
    'Gradient Boosting': 0.6654,
    'XGBoost': 0.6716
}

baseline = 0.587

# 1. Performance Comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = list(results.keys())
accuracies = list(results.values())
improvements = [acc - baseline for acc in accuracies]

colors = ['#FFD700', '#3498db', '#2ecc71']  # Gold for best (AdaBoost)
bars = ax.bar(models, [acc*100 for acc in accuracies], color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f}%\n(+{imp*100:.2f}pp)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Baseline line
ax.axhline(y=baseline*100, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline*100:.2f}%', alpha=0.7)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Boosting Model Performance on CS2 Round Prediction\n(2507 samples from 13 matches)', 
             fontsize=14, fontweight='bold')
ax.set_ylim([50, 75])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../../docs/img/boosting_performance_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved performance_comparison.png")

plt.close()

# 2. Comparison chart
fig, ax = plt.subplots(figsize=(12, 6))

comparison_data = {
    'Method': ['Baseline\n(Old Features\n288 samples)', 
               'AdaBoost\n(Enhanced\n2507 samples)', 
               'XGBoost\n(Enhanced\n2507 samples)',
               'Gradient Boosting\n(Enhanced\n2507 samples)'],
    'Accuracy': [58.70, 68.90, 67.16, 66.54],
    'Color': ['#cccccc', '#FFD700', '#2ecc71', '#3498db']
}

bars = ax.bar(comparison_data['Method'], comparison_data['Accuracy'], 
              color=comparison_data['Color'], alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Accuracy Comparison: Impact of Enhanced Features and Larger Dataset', 
             fontsize=14, fontweight='bold')
ax.set_ylim([50, 75])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../../docs/img/boosting_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved accuracy_comparison.png")

print("\n✓ All visualizations regenerated!")

