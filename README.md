# Mitsui Commodity Prediction Challenge

This repository contains the solution for the Kaggle competition: [Mitsui Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge).

## Project Overview

The goal is to predict commodity spread returns using various financial instruments including:
- JPX Futures
- LME Metals
- US Stocks
- FX Pairs

## Project Structure

```
mitsui-commodity-prediction/
├── data/
│   ├── raw/           # Raw competition data
│   └── processed/     # Processed/transformed data
├── models/            # Trained models and model artifacts
├── notebooks/         # Jupyter notebooks for analysis
├── src/              # Source code modules
├── submissions/       # Competition submissions
└── visualizations/    # Generated plots and charts
```

## Setup

### Environment Setup
```bash
# Create conda environment
conda create -n commodity-prediction python=3.11

# Activate environment
conda activate commodity-prediction

# Install required packages
conda install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly jupyter ipykernel
```

### Data Setup
1. Download competition data using Kaggle CLI
2. Extract data to `data/raw/` directory
3. Run initial EDA notebook: `notebooks/01_eda_overview.ipynb`

## Data Description

- **Features**: 558 financial instrument prices across multiple markets
- **Targets**: 425 spread return predictions with varying lags
- **Time Period**: 1917 time points (daily data)
- **Test Set**: 90 time points for predictions

## Current Progress

- [x] Environment setup
- [x] Data download and exploration
- [x] Initial EDA and data understanding
- [x] Target definition verification
- [x] Log returns feature creation
- [ ] Feature engineering
- [ ] Model development
- [ ] Submission generation

## Key Insights

- Targets are forward-looking log returns: `log(price_{t+lag+1} / price_{t+1})`
- For pairs: difference between two instrument returns
- High missing values in some US stock features (e.g., US_Stock_GOlD ~80% missing)
- Features grouped by market: JPX_Futures, LME_Metals, US_Stocks, FX_Pairs

## Next Steps

1. Complete feature engineering
2. Develop baseline models
3. Feature selection and optimization
4. Model ensemble and tuning
5. Generate final predictions
