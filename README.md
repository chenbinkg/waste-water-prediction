# Wastewater Flow Prediction using Machine Learning

A comprehensive machine learning framework for predicting wastewater flow using meteorological features and time series analysis.

## Overview

This project implements a multi-algorithm ensemble approach to predict wastewater flow based on:
- Rainfall patterns and antecedent precipitation
- Temperature and soil moisture
- Temporal features (hour, day, month)
- Lag features derived from ACF/PACF analysis

## Project Structure

```
waste-water-modelling/
├── scripts/
│   ├── feature_analysis.py      # Core analysis functions
│   ├── add_lag_features.py      # Lag feature engineering
│   └── ...
├── run_experiments.py           # Main experiment runner
├── consolidate_ablation.py      # Consolidate ablation results
├── data/
│   └── processed/              # Processed datasets with lag features
├── results/
│   ├── acf/                    # ACF/PACF analysis plots
│   ├── ablation_*.csv          # Feature ablation results
│   ├── algorithms_*.csv        # Algorithm benchmark results
│   ├── importance_*.csv        # Feature importance rankings
│   └── models/                 # Trained ensemble models
└── README.md
```

## Features

### 1. Time Series Analysis (ACF/PACF)
- Stationarity testing (ADF, KPSS)
- Autocorrelation and partial autocorrelation analysis
- Lag feature recommendations based on statistical significance

### 2. Feature Engineering
- **Temporal features**: hour, day of week, month, day of year
- **Cyclical encoding**: sin/cos transformations for hour
- **Rainfall features**: 
  - Lag features (1, 2, 3 hours)
  - Rolling sums (3, 6, 12, 24, 48 hours)
- **Temperature features**:
  - Lag features (1, 2, 3, 24 hours)
  - Rolling means (6, 12, 24 hours)
- **Derived indices**:
  - Antecedent Precipitation Index (API)
  - Soil Moisture Index
  - Lag features for API and soil moisture

### 3. Feature Ablation Study
- Tests 4 accurate algorithms: Random Forest, XGBoost, Extra Trees, CatBoost
- Cumulative feature addition approach
- Consensus-based feature selection (≥3 out of 4 algorithms agree)
- Tracks RMSE, MAE, and R² improvements

### 4. Algorithm Benchmarking
- Compares 6 algorithms: XGBoost, GradientBoosting, CatBoost, RandomForest, ExtraTrees, LinearRegression
- Uses consensus features from ablation study
- Generates performance heatmaps across sites

### 5. Feature Importance Analysis
- Uses Extra Trees Regressor (best performing algorithm)
- Generates horizontal bar plots
- Ranks features by importance score

### 6. Stacked Ensemble Model
- **Layer 1 (Base)**: 6 models × 5-fold CV = 30 models
  - XGBoost, GradientBoosting, CatBoost, RandomForest, ExtraTrees, LinearRegression
- **Layer 2 (Stack)**: 3 models × 5-fold CV = 15 models
  - RandomForest, XGBoost, ExtraTrees
- **Layer 3 (Meta)**: Greedy weighted ensemble
- Total: 46 models trained per site

## Installation

```bash
# Create conda environment
conda create -n wastewater python=3.8
conda activate wastewater

# Install dependencies
pip install pandas numpy matplotlib seaborn
pip install scikit-learn xgboost catboost
pip install statsmodels boto3 sagemaker
```

## Usage

### Run Full Experiment Pipeline

```python
python run_experiments.py
```

This runs all experiments with default settings:
- ACF/PACF analysis
- Feature ablation study
- Algorithm benchmarking
- Feature importance analysis
- Stacked ensemble training (optional)

### Customize Experiments

```python
from run_experiments import main

# Run only specific components
results = main(
    run_acf=True,           # ACF/PACF analysis
    run_ablation=True,      # Feature ablation
    run_benchmark=True,     # Algorithm benchmark
    run_importance=True,    # Feature importance
    run_ensemble=False      # Stacked ensemble (slow)
)
```

### Consolidate Ablation Results

```python
python consolidate_ablation.py
```

Generates:
- `ablation_all_combined.csv`: All ablation results
- `ablation_consensus_summary.csv`: Consensus percentage by feature group

## Configuration

Edit `run_experiments.py` to modify:

```python
target_sites = ['Lower Hutt', 'Stokes Valley', 'Upper Hutt']
scenarios = ['CUR', 'ABI', 'ABJ', 'ABK', 'ABL']
train_size = 0.8  # 80-20 split
split_method = 'stratified'  # or 'time'
```

## Output Files

### ACF/PACF Analysis
- `results/acf/{site}/stationarity_tests.csv`: Stationarity test results
- `results/acf/{site}/lag_recommendations.csv`: Recommended lag features
- `results/acf/{site}/*_acf_pacf.png`: ACF/PACF plots
- `results/acf/{site}/*_series_comparison.png`: Original vs differenced series

### Feature Ablation
- `results/ablation_{site}_{scenario}.csv`: Feature group effectiveness
  - Columns: `feature_group`, `n_features`, `n_effective`, `consensus`, `{ALGO}_rmse`, `{ALGO}_mae`, `{ALGO}_r2`, `{ALGO}_eff`

### Algorithm Benchmark
- `results/algorithms_{site}_{scenario}.csv`: Algorithm performance comparison
  - Columns: `algorithm`, `rmse`, `mae`, `r2`

### Feature Importance
- `results/importance_{site}_{scenario}.csv`: Feature importance rankings
- `results/importance_{site}_{scenario}.png`: Horizontal bar plot

### Ensemble Models
- `results/models/ensemble_{site}.pkl`: Trained ensemble model
- `results/models/predictions_{site}.csv`: Predictions with ground truth
- `results/models/timeseries_{site}.png`: Time series plot (stratified sampled)
- `results/models/scatter_{site}.png`: Scatter plot with fit line

## Key Algorithms

### Feature Selection
- **Consensus approach**: Feature group is effective if ≥3 out of 4 algorithms show improvement
- **Improvement criteria**: Both RMSE and MAE must decrease

### Ensemble Training
- **K-fold bagging**: Each model trained on 5 different folds
- **Out-of-fold predictions**: Used to train next layer without overfitting
- **Greedy weighting**: Layer 3 finds optimal weights for Layer 2 models

### Time Series Plotting
- **Stratified sampling**: Samples consistently from each year-month
- **Max 5000 points**: Prevents matplotlib overflow errors
- **Maintains temporal patterns**: Random sampling within each month

## Performance Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **Consensus**: Percentage of sites/scenarios where feature is effective

## Data Requirements

Input CSV files should contain:
- `Time`: Datetime column
- `Rainfall`: Hourly rainfall (mm)
- `Dry bulb degC`: Air temperature (°C)
- Target column: Wastewater flow measurement

## Notes

- **Data cleaning**: Invalid strings converted to NaN, then forward filled
- **Parallel processing**: Most algorithms use `n_jobs=-1` for speed
- **CatBoost**: Uses `thread_count=-1` instead of `n_jobs`
- **GradientBoosting**: No parallel support (inherently sequential)
- **Reduced trees**: Using 50 estimators for speed (vs 100-200 for production)

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```

## License

[Your license here]

## Contact

[Your contact information here]
