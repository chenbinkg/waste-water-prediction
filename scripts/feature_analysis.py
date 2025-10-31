"""
Feature effectiveness and ACF analysis for water demand prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def run_acf_analysis(df, target_col, variables, max_lag=48, save_dir='results/acf', 
                     diff_vars=['Tmax', 'Sun', 'PE']):
    """Run ACF/PACF analysis for target and key variables with stationarity tests
    
    Args:
        diff_vars: Variables to difference before ACF/PACF (non-stationary variables)
    """
    import os
    from statsmodels.tsa.stattools import adfuller, kpss
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    stationarity_summary = []
    
    for var in [target_col] + variables:
        original_data = df[var].dropna()
        
        # Stationarity tests on original series
        adf_orig = adfuller(original_data, autolag='AIC')
        kpss_orig = kpss(original_data, regression='c', nlags='auto')
        
        # ACF-based stationarity check (practical test)
        n = len(original_data)
        conf_band = 1.96 / np.sqrt(n)
        acf_orig = acf(original_data, nlags=min(40, len(original_data)//2))
        # Count how many of first 24 lags are outside confidence band
        significant_lags_orig = np.sum(np.abs(acf_orig[1:25]) > conf_band) if len(acf_orig) > 24 else 0
        # If >17 out of 24 lags significant, series has strong persistence
        acf_decays = significant_lags_orig < 17

        is_stationary_orig = (adf_orig[1] < 0.05) and (kpss_orig[1] > 0.05) and acf_decays
        
        # Stationarity tests on differenced series
        diff_data = original_data.diff().dropna()
        adf_diff = adfuller(diff_data, autolag='AIC')
        kpss_diff = kpss(diff_data, regression='c', nlags='auto')
        
        # ACF-based check for differenced series
        acf_diff = acf(diff_data, nlags=min(40, len(diff_data)//2))
        significant_lags_diff = np.sum(np.abs(acf_diff[1:25]) > conf_band) if len(acf_diff) > 24 else 0
        acf_diff_decays = significant_lags_diff < 17
        
        is_stationary_diff = (adf_diff[1] < 0.05) and (kpss_diff[1] > 0.05) and acf_diff_decays
        
        stationarity_summary.append({
            'variable': var,
            'orig_adf_pvalue': adf_orig[1],
            'orig_kpss_pvalue': kpss_orig[1],
            'orig_sig_lags': significant_lags_orig,
            'orig_stationary': is_stationary_orig,
            'diff_adf_pvalue': adf_diff[1],
            'diff_kpss_pvalue': kpss_diff[1],
            'diff_sig_lags': significant_lags_diff,
            'diff_stationary': is_stationary_diff
        })
        
        # Plot comparison: original vs first differenced
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(original_data.index, original_data.values)
        axes[0].set_title(f'{var} - Original Series')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(diff_data.index, diff_data.values)
        axes[1].set_title(f'{var} - First Differenced')
        axes[1].set_ylabel('Difference')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{var}_series_comparison.png', dpi=300)
        plt.close()
        
        # Use differenced data for ACF/PACF
        data = diff_data
        title_suffix = ' (differenced)'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(data, lags=max_lag, ax=ax1)
        plot_pacf(data, lags=max_lag, ax=ax2)
        ax1.set_title(f'ACF - {var}{title_suffix}')
        ax2.set_title(f'PACF - {var}{title_suffix}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{var}_acf_pacf.png', dpi=300)
        plt.close()
        
        acf_vals = acf(data, nlags=max_lag)
        pacf_vals = pacf(data, nlags=max_lag)
        results[var] = {
            'acf': acf_vals, 
            'pacf': pacf_vals, 
            'differenced': True,
            'orig_stationary': is_stationary_orig,
            'diff_stationary': is_stationary_diff
        }
    
    # Save stationarity test results
    stationarity_df = pd.DataFrame(stationarity_summary)
    stationarity_df.to_csv(f'{save_dir}/stationarity_tests.csv', index=False)
    
    # Analyze ACF/PACF for lag recommendations
    lag_recommendations = []
    for idx, var in enumerate([target_col] + variables):
        original_data = df[var].dropna()
        
        # Use original if stationary, differenced if non-stationary
        is_orig_stationary = stationarity_summary[idx]['orig_stationary']
        if is_orig_stationary:
            analysis_data = original_data
            data_type = 'original'
        else:
            analysis_data = original_data.diff().dropna()
            data_type = 'differenced'
        
        acf_vals = acf(analysis_data, nlags=min(20, len(analysis_data)//2))
        pacf_vals = pacf(analysis_data, nlags=min(20, len(analysis_data)//2))
        
        n = len(analysis_data)
        conf_band = 1.96 / np.sqrt(n)
        
        # Find significant PACF lags (direct effects)
        sig_pacf_lags = []
        for lag in range(1, min(11, len(pacf_vals))):
            if np.abs(pacf_vals[lag]) > conf_band:
                sig_pacf_lags.append(lag)
        
        # Recommend lags based on PACF (direct effects) and practical threshold
        practical_threshold = 0.05  # Practical significance
        recommended_lags = []
        for lag in sig_pacf_lags:
            if lag <= 10 and np.abs(pacf_vals[lag]) > practical_threshold:
                recommended_lags.append(lag)
        
        # Limit to first 3 lags for parsimony (domain knowledge)
        recommended_lags = [l for l in recommended_lags if l <= 3]
        
        lag_recommendations.append({
            'variable': var,
            'data_used': data_type,
            'sig_pacf_lags': sig_pacf_lags[:7] if sig_pacf_lags else [],
            'pacf_values': [f'{pacf_vals[l]:.3f}' for l in sig_pacf_lags[:7]] if sig_pacf_lags else [],
            'recommended_lags': recommended_lags,
            'reasoning': f'PACF sig + |PACF|>0.05 + lag<=3 ({data_type})'
        })
    
    # Save lag recommendations
    lag_df = pd.DataFrame(lag_recommendations)
    lag_df.to_csv(f'{save_dir}/lag_recommendations.csv', index=False)
    
    # Print summary
    print("\n  Stationarity Test Results (Original vs Differenced):")
    print("  " + "="*105)
    print(f"  {'Variable':<20} {'Original':<35} {'After 1st Diff':<35}")
    print(f"  {'':<20} {'ADF p':<8} {'KPSS p':<8} {'SigLag':<8} {'Stat':<5} {'ADF p':<8} {'KPSS p':<8} {'SigLag':<8} {'Stat':<5}")
    print("  " + "-"*105)
    for _, row in stationarity_df.iterrows():
        orig_status = 'Y' if row['orig_stationary'] else 'N'
        diff_status = 'Y' if row['diff_stationary'] else 'N'
        print(f"  {row['variable']:<20} {row['orig_adf_pvalue']:<8.4f} {row['orig_kpss_pvalue']:<8.4f} "
              f"{row['orig_sig_lags']:<8.0f} {orig_status:<5} "
              f"{row['diff_adf_pvalue']:<8.4f} {row['diff_kpss_pvalue']:<8.4f} "
              f"{row['diff_sig_lags']:<8.0f} {diff_status:<5}")
    print("  " + "="*105)
    print("  Stationary requires: ADF p<0.05 AND KPSS p>0.05 AND <7 significant lags (out of 10)")
    print(f"  Confidence band: +/-{conf_band:.4f} | SigLag = # of lags 1-10 outside band")
    print("  All ACF/PACF plots use differenced series")
    
    # Print lag recommendations
    print("\n  Lag Feature Recommendations (based on PACF):")
    print("  " + "="*105)
    print(f"  {'Variable':<20} {'Data':<10} {'Sig PACF Lags':<25} {'PACF Values':<25} {'Recommended':<15}")
    print("  " + "-"*105)
    for _, row in lag_df.iterrows():
        sig_lags_str = str(row['sig_pacf_lags'][:7]) if row['sig_pacf_lags'] else 'None'
        pacf_str = str(row['pacf_values'][:7]) if row['pacf_values'] else '-'
        rec_str = str(row['recommended_lags']) if row['recommended_lags'] else 'None'
        print(f"  {row['variable']:<20} {row['data_used']:<10} {sig_lags_str:<25} {pacf_str:<25} {rec_str:<15}")
    print("  " + "="*105)
    print("  Criteria: Statistically significant (outside conf band) + |PACF|>0.05 + lag<=3")
    print("  Data: 'original' for stationary series, 'differenced' for non-stationary series")
    print("  Note: Lag<=3 constraint based on domain knowledge (recent weather matters most)")
    
    return results

def define_feature_groups(site_name=None):
    """Define feature groups for ablation study with granular grouping"""
    groups = {
        'date_basic': ['doy', 'month', 'mday', 'wday', 'hour'],
        'temp': ['Dry bulb degC'],
        'temp_lag1': ['Dry bulb degC lag1'],
        'temp_lag2_3': ['Dry bulb degC lag2', 'Dry bulb degC lag3'],
        'temp_lag24': ['Dry bulb degC lag24'],
        'temp_rolling': ['Temp_6HR_mean', 'Temp_12HR_mean', 'Temp_24HR_mean'],
        'soilm': ['SoilMoisture'],
        'soilm_lag1': ['SoilMoisturelag1'],
        'rain': ['Rainfall'],
        'rain_lag1_2': ['Rainlag1', 'Rainlag2'],
        'rain_lag3': ['Rainlag3'],
        'rain_rolling': ['Rain_L3HR', 'Rain_L6HR', 'Rain_L12HR', 'Rain_L24HR', 
                         'Rain_L48HR'],
        'cyclical': ['sin_hour', 'cos_hour'],
        'api': ['API'],
        'api_lag1': ['lag1API']
    }
    
    return groups

def get_algorithms():
    """Define algorithms for benchmarking"""
    return {
        'XGBoost': XGBRegressor(random_state=42, n_estimators=50, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=50),
        'CatBoost': CatBoostRegressor(random_state=42, iterations=50, thread_count=-1, verbose=0),
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=50, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=50, n_jobs=-1),
        'LinearRegression': LinearRegression()
    }

def feature_ablation_study(df, target_col, feature_groups, train_size=0.7, split_method='stratified'):
    """Run feature ablation study with multi-algorithm consensus"""
    if split_method == 'time':
        split_idx = int(len(df) * train_size)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
    else:
        from sklearn.model_selection import train_test_split
        if 'Restriction level' in df.columns:
            train, test = train_test_split(df, train_size=train_size, 
                                          stratify=df['Restriction level'].fillna(0).astype(int),
                                          random_state=42)
        else:
            train, test = train_test_split(df, train_size=train_size, random_state=42)
    
    # Test algorithms for consensus (4 accurate algorithms)
    test_algorithms = {
        'RF': RandomForestRegressor(random_state=42, n_estimators=50, n_jobs=-1),
        'XGB': XGBRegressor(random_state=42, n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesRegressor(random_state=42, n_estimators=50, n_jobs=-1),
        'CB': CatBoostRegressor(random_state=42, iterations=50, thread_count=-1, verbose=0)
    }
    
    results = []
    cumulative_features = []
    prev_metrics = {algo: {'rmse': float('inf'), 'mae': float('inf')} for algo in test_algorithms}
    
    for group_name, features in feature_groups.items():
        cumulative_features.extend(features)
        available_features = [f for f in cumulative_features if f in train.columns]
        
        if len(available_features) == 0:
            continue
            
        X_train = train[available_features].ffill().bfill()
        y_train = train[target_col].ffill().bfill()
        X_test = test[available_features].ffill().bfill()
        y_test = test[target_col].ffill().bfill()
        
        # Test on all 3 algorithms
        algo_results = {}
        improvements = []
        
        for algo_name, model in test_algorithms.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            algo_results[algo_name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            
            # Check if both RMSE and MAE improved
            rmse_improved = rmse < prev_metrics[algo_name]['rmse']
            mae_improved = mae < prev_metrics[algo_name]['mae']
            improvements.append(rmse_improved and mae_improved)
            
            prev_metrics[algo_name] = {'rmse': rmse, 'mae': mae}
        
        # Consensus: >=3 out of 4 algorithms show improvement
        consensus = sum(improvements) >= 3
        
        result_dict = {
            'feature_group': group_name,
            'n_features': len(available_features),
            'n_effective': sum(improvements),
            'consensus': consensus
        }
        
        # Add metrics for each algorithm
        for i, (algo_name, metrics) in enumerate(algo_results.items()):
            result_dict[f'{algo_name}_rmse'] = metrics['rmse']
            result_dict[f'{algo_name}_mae'] = metrics['mae']
            result_dict[f'{algo_name}_r2'] = metrics['r2']
            result_dict[f'{algo_name}_eff'] = improvements[i]
        
        results.append(result_dict)
    
    return pd.DataFrame(results)

def algorithm_benchmark(df, target_col, features, train_size=0.7, split_method='stratified'):
    """Benchmark all algorithms with full feature set"""
    if split_method == 'time':
        split_idx = int(len(df) * train_size)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
    else:
        from sklearn.model_selection import train_test_split
        if 'Restriction level' in df.columns:
            train, test = train_test_split(df, train_size=train_size,
                                          stratify=df['Restriction level'].fillna(0).astype(int),
                                          random_state=42)
        else:
            train, test = train_test_split(df, train_size=train_size, random_state=42)
    
    available_features = [f for f in features if f in train.columns]
    X_train = train[available_features].ffill().bfill()
    y_train = train[target_col].ffill().bfill()
    X_test = test[available_features].ffill().bfill()
    y_test = test[target_col].ffill().bfill()
    
    algorithms = get_algorithms()
    results = []
    
    for name, model in algorithms.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'algorithm': name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
    
    return pd.DataFrame(results)

def feature_importance_analysis(df, target_col, features, train_size=0.7, split_method='stratified', save_path=None):
    """Get feature importance from tree-based models"""
    if split_method == 'time':
        split_idx = int(len(df) * train_size)
        train = df.iloc[:split_idx].copy()
    else:
        from sklearn.model_selection import train_test_split
        if 'Restriction level' in df.columns:
            train, _ = train_test_split(df, train_size=train_size,
                                       stratify=df['Restriction level'].fillna(0).astype(int),
                                       random_state=42)
        else:
            train, _ = train_test_split(df, train_size=train_size, random_state=42)
    
    available_features = [f for f in features if f in train.columns]
    X_train = train[available_features].ffill().bfill()
    y_train = train[target_col].ffill().bfill()
    
    model = ExtraTreesRegressor(random_state=42, n_estimators=50, n_jobs=-1)
    model.fit(X_train, y_train)
    
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot horizontal bar chart
    if save_path:
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Extra Trees)')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return importance_df

def plot_algorithm_benchmark_heatmap(results_dir='results', output_path='results/algorithm_benchmark_heatmap.png'):
    """Create heatmap of R2 scores across sites and algorithms
    
    Args:
        results_dir: Directory containing algorithm benchmark CSV files
        output_path: Path to save the heatmap
    """
    import os
    import glob
    import seaborn as sns
    
    # Find all algorithm benchmark CSV files
    csv_files = glob.glob(os.path.join(results_dir, 'algorithms*.csv'))
    
    if not csv_files:
        print(f"No algorithm benchmark CSV files found in {results_dir}")
        return
    
    # Collect R2 scores from all sites
    data = []
    for csv_file in csv_files:
        site_name = os.path.basename(csv_file).replace('algorithms_', '')
        site_name = site_name.replace('.csv', '')
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            data.append({
                'Site': site_name,
                'Algorithm': row['algorithm'],
                'R2': row['r2']
            })
    
    # Create pivot table
    df_all = pd.DataFrame(data)
    pivot_table = df_all.pivot(index='Algorithm', columns='Site', values='R2')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_table.columns) * 0.8), 6))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'R² Score'},
                linewidths=0.5, ax=ax)
    ax.set_title('Algorithm Performance Across Sites (R² Score)', fontsize=14, pad=20)
    ax.set_xlabel('Site', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_path}")
    print(f"\nSummary statistics:")
    print(pivot_table.describe())
    
    return pivot_table

def train_stacked_ensemble(df, target_col, features, n_folds=5, train_size=0.7, split_method='stratified', random_state=42, save_dir=None, 
                           site_name=None, scenario=None):
    """Train 3-layer stacked ensemble with bagging
    
    Layer 1 (Base): XGB, GB, CB, RF, ET, MLP, LR with K-fold bagging
    Layer 2 (Stack): RF, XGB, MLP with K-fold bagging using Layer 1 OOF predictions
    Layer 3 (Meta): Weighted ensemble combining Layer 2 predictions
    
    Args:
        df: Input dataframe
        target_col: Target column name
        features: List of feature column names
        n_folds: Number of folds for bagging (default: 5)
        train_size: Train/test split ratio (default: 0.7)
        split_method: 'time' or 'stratified' (default: 'stratified')
        random_state: Random seed (default: 42)
        save_dir: Directory to save models and predictions (default: None)
        site_name: Site name for file naming (default: None)
        scenario: Scenario name for file naming (default: None)

    Returns:
        Dictionary containing trained models and predictions
    """
    import pickle
    import os
    from sklearn.model_selection import KFold, train_test_split
    
    # Split data
    if split_method == 'time':
        split_idx = int(len(df) * train_size)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
    else:
        if 'Restriction level' in df.columns:
            train, test = train_test_split(df, train_size=train_size,
                                          stratify=df['Restriction level'].fillna(0).astype(int),
                                          random_state=random_state)
        else:
            train, test = train_test_split(df, train_size=train_size, random_state=random_state)
    
    available_features = [f for f in features if f in train.columns]
    X_train = train[available_features].ffill().bfill().values
    y_train = train[target_col].ffill().bfill().values
    X_test = test[available_features].ffill().bfill().values
    y_test = test[target_col].ffill().bfill().values
    
    # Layer 1: Base models with bagging
    layer1_models = {
        'XGB': XGBRegressor(random_state=random_state, n_estimators=50, n_jobs=-1),
        'GB': GradientBoostingRegressor(random_state=random_state, n_estimators=50),
        'CB': CatBoostRegressor(random_state=random_state, iterations=50, verbose=0, thread_count=-1),
        'RF': RandomForestRegressor(random_state=random_state, n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesRegressor(random_state=random_state, n_estimators=50, n_jobs=-1),
        'LR': LinearRegression()
    }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    layer1_oof = np.zeros((len(X_train), len(layer1_models)))
    layer1_test = np.zeros((len(X_test), len(layer1_models)))
    layer1_trained = {}
    
    print(f"Training Layer 1: {len(layer1_models)} base models x {n_folds} folds = {len(layer1_models) * n_folds} models")
    for model_idx, (name, model) in enumerate(layer1_models.items()):
        fold_models = []
        fold_preds = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            
            # OOF predictions for training layer 2
            layer1_oof[val_idx, model_idx] = fold_model.predict(X_fold_val)
            
            # Test predictions (will be averaged)
            fold_preds.append(fold_model.predict(X_test))
            fold_models.append(fold_model)
        
        # Average test predictions from all folds
        layer1_test[:, model_idx] = np.mean(fold_preds, axis=0)
        layer1_trained[name] = fold_models
        
        oof_score = r2_score(y_train, layer1_oof[:, model_idx])
        print(f"  {name}: OOF R2 = {oof_score:.4f}")
    
    # Layer 2: Stacked models with bagging using Layer 1 OOF + original features
    layer2_models = {
        'RF': RandomForestRegressor(random_state=random_state, n_estimators=50, n_jobs=-1),
        'XGB': XGBRegressor(random_state=random_state, n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesRegressor(random_state=random_state, n_estimators=50, n_jobs=-1)
    }
    
    X_train_layer2 = np.hstack([X_train, layer1_oof])
    X_test_layer2 = np.hstack([X_test, layer1_test])
    
    layer2_oof = np.zeros((len(X_train_layer2), len(layer2_models)))
    layer2_test = np.zeros((len(X_test_layer2), len(layer2_models)))
    layer2_trained = {}
    
    print(f"\nTraining Layer 2: {len(layer2_models)} stacked models x {n_folds} folds = {len(layer2_models) * n_folds} models")
    for model_idx, (name, model) in enumerate(layer2_models.items()):
        fold_models = []
        fold_preds = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_layer2)):
            X_fold_train, X_fold_val = X_train_layer2[train_idx], X_train_layer2[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            
            # OOF predictions for training layer 3
            layer2_oof[val_idx, model_idx] = fold_model.predict(X_fold_val)
            
            # Test predictions (will be averaged)
            fold_preds.append(fold_model.predict(X_test_layer2))
            fold_models.append(fold_model)
        
        # Average test predictions from all folds
        layer2_test[:, model_idx] = np.mean(fold_preds, axis=0)
        layer2_trained[name] = fold_models
        
        oof_score = r2_score(y_train, layer2_oof[:, model_idx])
        print(f"  {name}: OOF R2 = {oof_score:.4f}")
    
    # Layer 3: Greedy weighted ensemble (meta-model)
    print(f"\nTraining Layer 3: Weighted ensemble (1 model)")
    
    # Greedy ensemble: iteratively find best weights
    weights = np.zeros(len(layer2_models))
    best_score = -np.inf
    
    for i in range(len(layer2_models)):
        best_weight = 0
        best_idx = -1
        
        for idx in range(len(layer2_models)):
            if weights[idx] > 0:
                continue
            
            # Try adding this model
            test_weights = weights.copy()
            test_weights[idx] = 1.0
            test_weights = test_weights / test_weights.sum()
            
            pred = (layer2_oof * test_weights).sum(axis=1)
            score = r2_score(y_train, pred)
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_weight = 1.0
        
        if best_idx >= 0:
            weights[best_idx] = best_weight
            weights = weights / weights.sum()
    
    # Final predictions
    final_train_pred = (layer2_oof * weights).sum(axis=1)
    final_test_pred = (layer2_test * weights).sum(axis=1)
    
    train_r2 = r2_score(y_train, final_train_pred)
    test_r2 = r2_score(y_test, final_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, final_test_pred))
    test_mae = mean_absolute_error(y_test, final_test_pred)
    
    print(f"\nFinal Ensemble Weights: {dict(zip(layer2_models.keys(), weights))}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    print(f"\nTotal models trained: {len(layer1_models) * n_folds + len(layer2_models) * n_folds + 1}")
    
    result = {
        'layer1_models': layer1_trained,
        'layer2_models': layer2_trained,
        'ensemble_weights': weights,
        'feature_names': available_features,
        'predictions': {
            'train': final_train_pred,
            'test': final_test_pred,
            'y_train': y_train,
            'y_test': y_test,
            'train_df': train,
            'test_df': test
        },
        'metrics': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
    }
    
    # Save models and predictions if save_dir provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save ensemble model
        model_name = f"ensemble_{site_name}_{scenario}.pkl" if site_name else "ensemble_model.pkl"
        model_path = os.path.join(save_dir, model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"\n  Model saved to: {model_path}")
        
        # Save predictions with ground truth
        pred_name = f"predictions_{site_name}_{scenario}.csv" if site_name else "predictions.csv"
        pred_path = os.path.join(save_dir, pred_name)
        
        # Combine train and test predictions
        train_df = pd.DataFrame({
            'split': 'train',
            'ground_truth': y_train,
            'prediction': final_train_pred,
            'residual': y_train - final_train_pred
        })
        
        test_df = pd.DataFrame({
            'split': 'test',
            'ground_truth': y_test,
            'prediction': final_test_pred,
            'residual': y_test - final_test_pred
        })
        
        pred_df = pd.concat([train_df, test_df], ignore_index=True)
        pred_df.to_csv(pred_path, index=False)
        print(f"  Predictions saved to: {pred_path}")
        
        # Plot time series
        plot_name = f"timeseries_{site_name}_{scenario}.png" if site_name else "timeseries.png"
        plot_path = os.path.join(save_dir, plot_name)
        
        # Get datetime from original dataframes
        train_df_sorted = train.copy()
        test_df_sorted = test.copy()
        
        # Add predictions
        train_df_sorted['prediction'] = final_train_pred
        test_df_sorted['prediction'] = final_test_pred
        
        # Sort by Date
        if 'Date' in train_df_sorted.columns:
            train_df_sorted = train_df_sorted.sort_values('Date')
            test_df_sorted = test_df_sorted.sort_values('Date')
            train_dates = train_df_sorted['Date']
            test_dates = test_df_sorted['Date']
        else:
            train_dates = train_df_sorted.index
            test_dates = test_df_sorted.index
        
        train_actual = train_df_sorted[target_col].values
        train_pred = train_df_sorted['prediction'].values
        test_actual = test_df_sorted[target_col].values
        test_pred = test_df_sorted['prediction'].values
        
        # Stratified sampling by year-month
        def stratified_sample(df_sorted, max_points=5000):
            date_col = df_sorted['Date'] if 'Date' in df_sorted.columns else df_sorted.index
            if not isinstance(date_col, pd.DatetimeIndex):
                date_col = pd.to_datetime(date_col)
            df_sorted['year_month'] = date_col.to_period('M')
            n_groups = df_sorted['year_month'].nunique()
            sample_per_group = max(1, max_points // n_groups)
            sampled = df_sorted.groupby('year_month', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_per_group), random_state=42)
            ).sort_index()
            return sampled
        
        train_sampled = stratified_sample(train_df_sorted)
        test_sampled = stratified_sample(test_df_sorted)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        ax1.plot(train_sampled.get('Date', train_sampled.index), train_sampled[target_col], label='Actual', alpha=0.7, linewidth=1)
        ax1.plot(train_sampled.get('Date', train_sampled.index), train_sampled['prediction'], label='Predicted', alpha=0.7, linewidth=1)
        ax1.set_title(f'Training Set - R² = {train_r2:.4f}')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(test_sampled.get('Date', test_sampled.index), test_sampled[target_col], label='Actual', alpha=0.7, linewidth=1)
        ax2.plot(test_sampled.get('Date', test_sampled.index), test_sampled['prediction'], label='Predicted', alpha=0.7, linewidth=1)
        ax2.set_title(f'Testing Set - R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}')
        ax2.set_ylabel('Value')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Time series plot saved to: {plot_path}")
        
        # Plot scatter
        scatter_name = f"scatter_{site_name}_{scenario}.png" if site_name else "scatter.png"
        scatter_path = os.path.join(save_dir, scatter_name)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training scatter
        ax1.scatter(y_train, final_train_pred, alpha=0.5, s=20)
        z = np.polyfit(y_train, final_train_pred, 1)
        p = np.poly1d(z)
        ax1.plot(y_train, p(y_train), 'r--', linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', linewidth=1, label='Perfect')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'Training Set\nR² = {train_r2:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Testing scatter
        ax2.scatter(y_test, final_test_pred, alpha=0.5, s=20)
        z = np.polyfit(y_test, final_test_pred, 1)
        p = np.poly1d(z)
        ax2.plot(y_test, p(y_test), 'r--', linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=1, label='Perfect')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'Testing Set\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Scatter plot saved to: {scatter_path}")
    
    return result

def plot_feature_importance_heatmap(results_dir='results', output_path='results/feature_importance_heatmap.png', top_n=20):
    """Create heatmap of feature importance scores across sites
    
    Args:
        results_dir: Directory containing feature importance CSV files
        output_path: Path to save the heatmap
        top_n: Number of top features to display (default: 20)
    """
    import os
    import glob
    
    # Find all feature importance CSV files
    csv_files = glob.glob(os.path.join(results_dir, 'importance_*.csv'))
    
    if not csv_files:
        print(f"No feature importance CSV files found in {results_dir}")
        return
    
    # Collect importance scores from all sites
    data = []
    for csv_file in csv_files:
        site_name = os.path.basename(csv_file).replace('importance_', '').replace('.csv', '')
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            data.append({
                'Site': site_name,
                'Feature': row['feature'],
                'Importance': row['importance']
            })
    
    # Create pivot table
    df_all = pd.DataFrame(data)
    pivot_table = df_all.pivot(index='Site', columns='Feature', values='Importance')
    
    # Get top N features by average importance
    top_features = pivot_table.mean().nlargest(top_n).index
    pivot_table = pivot_table[top_features]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(top_features) * 0.5), max(6, len(pivot_table) * 0.4)))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Importance Score'},
                linewidths=0.5, ax=ax)
    ax.set_title(f'Top {top_n} Feature Importance Across Sites', fontsize=14, pad=20)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Site', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_path}")
    print(f"\nTop {top_n} features by average importance:")
    print(pivot_table.mean().sort_values(ascending=False))
    
    return pivot_table

if __name__ == "__main__":
    # Example usage
    print("Feature Analysis Framework Ready")
    print("\nRecommended workflow:")
    print("1. Run ACF analysis: run_acf_analysis()")
    print("2. Feature ablation: feature_ablation_study()")
    print("3. Algorithm benchmark: algorithm_benchmark()")
    print("4. Feature importance: feature_importance_analysis()")
    print("5. Train stacked ensemble: train_stacked_ensemble()")
    print("6. Plot benchmark heatmap: plot_algorithm_benchmark_heatmap()")
    print("7. Plot feature importance heatmap: plot_feature_importance_heatmap()")
