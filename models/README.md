# Models Directory

This directory contains trained models and model parameters.

## Structure

- **`baseline/`**: Baseline models using simple approaches (e.g., log returns only)
- **`factor_model/`**: Factor model approaches (PCA, etc.)
- **`ensemble/`**: Ensemble models combining multiple approaches
- **`feature_importance/`**: Feature importance analysis and rankings

## File Naming Convention

- Models: `{target_name}_model.joblib`
- Feature importance: `{model_type}_feature_importance.csv`
- Predictions: `{model_type}_predictions.csv`

## Usage

```python
import joblib

# Load a trained model
model = joblib.load('models/baseline/target_0_model.joblib')

# Load predictions
predictions = pd.read_csv('models/predictions/baseline_predictions.csv')
```
