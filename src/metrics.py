"""
Competition metrics for Mitsui Commodity Prediction Challenge.
"""

import numpy as np
import pandas as pd

SOLUTION_NULL_FILLER = -999999


def rank_correlation_sharpe_ratio(merged_df: pd.DataFrame) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).

    :param merged_df: DataFrame containing prediction columns (starting with 'prediction_')
                      and target columns (starting with 'target_')
    :return: Sharpe ratio of the rank correlation
    """
    prediction_cols = [col for col in merged_df.columns if col.startswith('prediction_')]
    target_cols = [col for col in merged_df.columns if col.startswith('target_')]

    def _compute_rank_correlation(row):
        non_null_targets = [col for col in target_cols if not pd.isnull(row[col])]
        matching_predictions = [col for col in prediction_cols if col.replace('prediction', 'target') in non_null_targets]
        if not non_null_targets:
            return np.nan  # Return NaN instead of raising error
        if row[non_null_targets].std(ddof=0) == 0 or row[matching_predictions].std(ddof=0) == 0:
            return np.nan  # Return NaN instead of raising error
        return np.corrcoef(row[matching_predictions].rank(method='average'), row[non_null_targets].rank(method='average'))[0, 1]

    daily_rank_corrs = merged_df.apply(_compute_rank_correlation, axis=1)
    # Remove NaN values
    daily_rank_corrs = daily_rank_corrs.dropna()
    
    if len(daily_rank_corrs) == 0:
        return 0.0  # Return 0 if no valid correlations
    
    std_dev = daily_rank_corrs.std(ddof=0)
    if std_dev == 0:
        return 0.0  # Return 0 if no variance
    sharpe_ratio = daily_rank_corrs.mean() / std_dev
    return float(sharpe_ratio)


def calculate_competition_score(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """
    Calculate the competition score (rank correlation Sharpe ratio) for validation data.
    
    :param y_true: DataFrame with true target values (columns starting with 'target_')
    :param y_pred: DataFrame with predicted values (columns starting with 'target_')
    :return: Competition score (rank correlation Sharpe ratio)
    """
    # Ensure both DataFrames have the same columns
    common_cols = list(set(y_true.columns) & set(y_pred.columns))
    y_true_subset = y_true[common_cols].copy()
    y_pred_subset = y_pred[common_cols].copy()
    
    # Rename prediction columns to match competition format
    y_pred_subset.columns = [col.replace('target_', 'prediction_') for col in y_pred_subset.columns]
    
    # Combine true and predicted values
    merged_df = pd.concat([y_true_subset, y_pred_subset], axis=1)
    
    return rank_correlation_sharpe_ratio(merged_df)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Official competition scoring function.
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).
    
    :param solution: DataFrame with true target values
    :param submission: DataFrame with predicted values
    :param row_id_column_name: Name of the row ID column to remove
    :return: Competition score
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert all(solution.columns == submission.columns)

    submission = submission.rename(columns={col: col.replace('target_', 'prediction_') for col in submission.columns})

    # Not all securities trade on all dates, but solution files cannot contain nulls.
    # The filler value allows us to handle trading halts, holidays, & delistings.
    solution = solution.replace(SOLUTION_NULL_FILLER, None)
    return rank_correlation_sharpe_ratio(pd.concat([solution, submission], axis='columns'))


def interpret_competition_score(score: float) -> str:
    """
    Interpret the competition score and provide a qualitative assessment.
    
    :param score: Competition score (rank correlation Sharpe ratio)
    :return: Qualitative interpretation string
    """
    if score > 0.5:
        return "EXCELLENT (strong rank correlation with low variance)"
    elif score > 0.2:
        return "GOOD (decent rank correlation)"
    elif score > 0.0:
        return "POOR (weak rank correlation)"
    else:
        return "VERY POOR (negative or zero correlation)"
