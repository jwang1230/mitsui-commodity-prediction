import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the validation metrics
val_metrics = pd.read_csv('results/baseline/validation_metrics.csv')
train_metrics = pd.read_csv('results/baseline_performance.csv')

# Merge the data
performance_analysis = train_metrics.merge(val_metrics, on='target', suffixes=('_train', '_val'))

print("Baseline Model Performance Analysis")
print("=" * 60)

# Overall statistics
print(f"\nTraining Performance (Cross-validation):")
print(f"  Average RMSE: {performance_analysis['rmse_train'].mean():.4f}")
print(f"  RMSE Range: {performance_analysis['rmse_train'].min():.4f} - {performance_analysis['rmse_train'].max():.4f}")
print(f"  RMSE Std: {performance_analysis['rmse_train'].std():.4f}")

print(f"\nValidation Performance:")
print(f"  Average RMSE: {performance_analysis['rmse_val'].mean():.4f}")
print(f"  RMSE Range: {performance_analysis['rmse_val'].min():.4f} - {performance_analysis['rmse_val'].max():.4f}")
print(f"  RMSE Std: {performance_analysis['rmse_val'].std():.4f}")

print(f"\nR² Performance:")
print(f"  Average R²: {performance_analysis['r2'].mean():.4f}")
print(f"  R² Range: {performance_analysis['r2'].min():.4f} - {performance_analysis['r2'].max():.4f}")
print(f"  Positive R²: {(performance_analysis['r2'] > 0).sum()}/{len(performance_analysis)} targets")

# Performance comparison
print(f"\nPerformance Comparison (Train vs Validation):")
performance_analysis['rmse_diff'] = performance_analysis['rmse_val'] - performance_analysis['rmse_train']
performance_analysis['rmse_ratio'] = performance_analysis['rmse_val'] / performance_analysis['rmse_train']

print(f"  Average RMSE increase: {performance_analysis['rmse_diff'].mean():.4f}")
print(f"  Average RMSE ratio: {performance_analysis['rmse_ratio'].mean():.2f}x")
print(f"  Overfitting (val > train): {(performance_analysis['rmse_val'] > performance_analysis['rmse_train']).sum()}/{len(performance_analysis)} targets")

# Best and worst performers
print(f"\nBest Performers (by validation RMSE):")
best_performers = performance_analysis.nsmallest(3, 'rmse_val')[['target', 'rmse_train', 'rmse_val', 'r2']]
print(best_performers.to_string(index=False))

print(f"\nWorst Performers (by validation RMSE):")
worst_performers = performance_analysis.nlargest(3, 'rmse_val')[['target', 'rmse_train', 'rmse_val', 'r2']]
print(worst_performers.to_string(index=False))

# R² analysis
print(f"\nR² Analysis:")
r2_positive = performance_analysis[performance_analysis['r2'] > 0]
r2_negative = performance_analysis[performance_analysis['r2'] <= 0]

print(f"  Targets with positive R²: {len(r2_positive)}")
if len(r2_positive) > 0:
    print(f"    Average R²: {r2_positive['r2'].mean():.4f}")
    print(f"    Best R²: {r2_positive['r2'].max():.4f}")

print(f"  Targets with negative R²: {len(r2_negative)}")
if len(r2_negative) > 0:
    print(f"    Average R²: {r2_negative['r2'].mean():.4f}")
    print(f"    Worst R²: {r2_negative['r2'].min():.4f}")

# Detailed breakdown
print(f"\nDetailed Performance Breakdown:")
print(performance_analysis[['target', 'rmse_train', 'rmse_val', 'r2', 'rmse_ratio']].to_string(index=False))

# Summary
print(f"\n" + "=" * 60)
print("SUMMARY:")
print(f"• The baseline models show significant overfitting (validation RMSE > training RMSE)")
print(f"• Most targets have negative R², indicating the model performs worse than predicting the mean")
print(f"• The log return features alone may not be sufficient for good predictions")
print(f"• Consider using all engineered features or different modeling approaches")
print(f"• The time gap between training and validation may be causing distribution shift")
