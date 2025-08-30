# Results Directory

This directory contains analysis results, performance metrics, predictions, and model comparisons.

## Structure

- **`baseline/`**: Results for baseline models
  - `predictions.csv`: Model predictions on test/validation set
  - `performance.csv`: Performance metrics
  - `feature_importance.csv`: Feature importance analysis
- **`factor_model/`**: Results for factor model approaches
- **`ensemble/`**: Results for ensemble models
- **`model_comparisons.csv`**: Comparison of different model approaches

## File Descriptions

### `baseline_performance.csv`
Columns:
- `target`: Target variable name
- `rmse`: Root Mean Square Error
- `mae`: Mean Absolute Error
- `r2`: R-squared score
- `training_time`: Model training time in seconds

### `feature_importance_summary.csv`
Columns:
- `feature`: Feature name
- `target`: Target variable
- `importance`: Feature importance score
- `rank`: Rank of feature importance for this target

### `model_comparisons.csv`
Columns:
- `model_type`: Type of model (baseline, factor_model, etc.)
- `avg_rmse`: Average RMSE across all targets
- `avg_r2`: Average R-squared across all targets
- `training_time`: Total training time
- `num_features`: Number of features used
