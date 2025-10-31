"""
Main script to run all experiments for technical paper
"""
import pandas as pd
import numpy as np
from scripts.feature_analysis import (
    run_acf_analysis, 
    define_feature_groups,
    feature_ablation_study,
    algorithm_benchmark,
    feature_importance_analysis,
    train_stacked_ensemble
)
import os
import boto3


def prepare_data(target_site, scenario, rf_name, bucket='niwa-water-demand-modelling', save_local=True):
    """Load and prepare data for a specific target site from S3"""
    from scripts.add_lag_features import add_lag_features

    s3 = boto3.client('s3')
    # PCD Lower Hutt wet weather flow ABI

#     ['WWL/PCD_Lower_Hutt_wet_weather_flow_ABI_birch_lane.csv',
#  'WWL/PCD_Lower_Hutt_wet_weather_flow_ABJ_birch_lane.csv',
#  'WWL/PCD_Lower_Hutt_wet_weather_flow_ABK_birch_lane.csv',
#  'WWL/PCD_Lower_Hutt_wet_weather_flow_ABL_birch_lane.csv',
#  'WWL/PCD_Lower_Hutt_wet_weather_flow_birch_lane.csv',
#  'WWL/PCD_Stokes_Valley_wet_weather_flow_ABI_birch_lane.csv',
#  'WWL/PCD_Stokes_Valley_wet_weather_flow_ABJ_birch_lane.csv',
#  'WWL/PCD_Stokes_Valley_wet_weather_flow_ABK_birch_lane.csv',
#  'WWL/PCD_Stokes_Valley_wet_weather_flow_ABL_birch_lane.csv',
#  'WWL/PCD_Stokes_Valley_wet_weather_flow_birch_lane.csv',
#  'WWL/PCD_Upper_Hutt_wet_weather_flow_ABI_pine_haven.csv',
#  'WWL/PCD_Upper_Hutt_wet_weather_flow_ABJ_pine_haven.csv',
#  'WWL/PCD_Upper_Hutt_wet_weather_flow_ABK_pine_haven.csv',
#  'WWL/PCD_Upper_Hutt_wet_weather_flow_ABL_pine_haven.csv',
#  'WWL/PCD_Upper_Hutt_wet_weather_flow_pine_haven.csv']
    if scenario != 'CUR':
        target_col = f"PCD_{target_site.replace(' ', '_')}_wet_weather_flow_{scenario}"
    else:
        target_col = f"PCD_{target_site.replace(' ', '_')}_wet_weather_flow"
    file_name = f"{target_col}_{rf_name.replace(' ', '_')}.csv"
    key = f"WWL/{file_name}"
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Rename columns to match expected format
    target_col = target_col.replace("_", " ") # e.g. 'PCD Lower Hutt wet weather flow ABI'
    if target_site in df.columns:
        df = df.rename(columns={target_site: target_col})

    # Clean data: convert invalid strings to NaN, then forward fill
    for col in df.columns:
        if col != 'Time':
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill()
    
    # Add new lag features based on PACF analysis
    df = add_lag_features(df, site_name=target_site)

    # Save locally if requested
    if save_local:
        local_dir = 'data/processed'
        os.makedirs(local_dir, exist_ok=True)
        site_name = target_site.replace(")", "").replace("(", "").replace(" ", "_")
        local_path = f"{local_dir}/{site_name}_{scenario}_with_lags.csv"
        df.to_csv(local_path, index=False)
        print(f"  Saved processed data to: {local_path}")
    
    return df, target_col

def main(run_acf=True, run_ablation=True, run_benchmark=True, run_importance=True, run_ensemble=False):
    """Run experiments with optional components
    
    Args:
        run_acf: Run ACF/PACF analysis
        run_ablation: Run feature ablation study
        run_benchmark: Run algorithm benchmark
        run_importance: Run feature importance analysis
        run_ensemble: Run stacked ensemble training
    """
    # Configuration
    target_sites = [
        'Lower Hutt',
        'Stokes Valley',
        'Upper Hutt'
    ]
    scenarios = [
        'CUR',
        'ABI',
        'ABJ',
        'ABK',
        'ABL'
    ]
    rf_dict = {
    'Upper Hutt': 'pine haven',
    'Stokes Valley': 'birch lane',
    'Lower Hutt': 'birch lane',
    }
    
    train_size = 0.8  # 80-20 split
    split_method = 'stratified'  # 'stratified' or 'time'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Variables for ACF analysis
    acf_variables = ['API', 'SoilMoisture', 'Dry bulb degC', 'Rainfall']
    
    all_results = {}
    
    for site in target_sites:
        site_results = {}
        rf_name = rf_dict[site]
        print(f"\n{'='*60}")
        print(f"Processing: {site}, RF name: {rf_name}")
        print(f"{'='*60}")
        # use fixed scenario to run ACF
        df, target_col = prepare_data(site, scenarios[0], rf_name)
        # 1. ACF Analysis on site level
        if run_acf:
            print("\n1. Running ACF/PACF analysis...")
            acf_results = run_acf_analysis(
                df, target_col, acf_variables, 
                max_lag=48, 
                save_dir=f'{results_dir}/acf/{site}'
            )
            site_results['acf'] = acf_results
        # loop by scenario
        for scenario in scenarios:
            scenario_results = {}
            print(f"\n=== Processing scenario: {scenario} ===")
            df, target_col = prepare_data(site, scenario, rf_name)
        
            # 2. Feature Ablation Study
            if run_ablation:
                print(f"\n2. Running feature ablation study with multi-algorithm consensus...")
                print(f"   ({split_method} split, {train_size:.0%} train, testing 4 algorithms)")
                feature_groups = define_feature_groups(site_name=site)
                ablation_results = feature_ablation_study(df, target_col, feature_groups, train_size, split_method)
                ablation_results.to_csv(f'{results_dir}/ablation_{site}_{scenario}.csv', index=False)
                scenario_results['ablation'] = ablation_results
                print("   ✓ Ablation results saved")
                
                # Display summary with consensus
                summary_cols = ['feature_group', 'n_features', 'n_effective', 'RF_rmse', 'RF_mae', 'RF_r2', 
                            'RF_eff', 'XGB_eff', 'ET_eff', 'CB_eff', 'consensus']
                print(ablation_results[summary_cols].to_string(index=False))
                
                # Count effective groups
                n_consensus = ablation_results['consensus'].sum()
                n_total = len(ablation_results)
                print(f"\n   Consensus: {n_consensus}/{n_total} groups effective (>=3 of 4 algorithms agree)")
        
            # 3. Algorithm Benchmark
            if run_benchmark:
                # Try to load ablation results from file if not in memory
                ablation_df = None
                if run_ablation and 'ablation' in scenario_results:
                    ablation_df = scenario_results['ablation']
                else:
                    # Try loading from saved file
                    ablation_file = f'{results_dir}/ablation_{site}_{scenario}.csv'
                    if os.path.exists(ablation_file):
                        ablation_df = pd.read_csv(ablation_file)
                        print(f"   Loaded ablation results from {ablation_file}")
                
                if ablation_df is not None:
                    # Use only consensus features
                    print("\n3. Running algorithm benchmark (consensus features only, excluding target lags)...")
                    consensus_groups = ablation_df[ablation_df['consensus'] == True]['feature_group'].tolist()
                    
                    feature_groups_full = define_feature_groups(site_name=site)
                    all_features = []
                    for group_name, features in feature_groups_full.items():
                        if group_name != 'target_lag1_2_3' and group_name in consensus_groups:
                            all_features.extend(features)
                    
                    print(f"   Using {len(all_features)} features from {len(consensus_groups)} consensus groups")
                else:
                    # Fallback: use all features except target lags
                    print("\n3. Running algorithm benchmark (all features, excluding target lags)...")
                    feature_groups_full = define_feature_groups(site_name=site)
                    all_features = []
                    for group_name, features in feature_groups_full.items():
                        if group_name != 'target_lag1_2_3':
                            all_features.extend(features)
                
                algo_results = algorithm_benchmark(df, target_col, all_features, train_size, split_method)
                algo_results.to_csv(f'{results_dir}/algorithms_{site}_{scenario}.csv', index=False)
                scenario_results['algorithms'] = algo_results
                print("   ✓ Algorithm results saved")
                print(algo_results.to_string(index=False))
        
            # 4. Feature Importance
            if run_importance:
                # Try to load ablation results from file if not in memory
                ablation_df = None
                if run_ablation and 'ablation' in scenario_results:
                    ablation_df = scenario_results['ablation']
                else:
                    # Try loading from saved file
                    ablation_file = f'{results_dir}/ablation_{site}_{scenario}.csv'
                    if os.path.exists(ablation_file):
                        ablation_df = pd.read_csv(ablation_file)
                        print(f"   Loaded ablation results from {ablation_file}")
                
                if ablation_df is not None:
                    # Use only consensus features
                    print("\n4. Analyzing feature importance (consensus features only, excluding target lags)...")
                    consensus_groups = ablation_df[ablation_df['consensus'] == True]['feature_group'].tolist()
                    
                    feature_groups_full = define_feature_groups(site_name=site)
                    all_features = []
                    for group_name, features in feature_groups_full.items():
                        if group_name != 'target_lag1_2_3' and group_name in consensus_groups:
                            all_features.extend(features)
                    
                    print(f"   Using {len(all_features)} features from {len(consensus_groups)} consensus groups")
                else:
                    # Fallback: use all features except target lags
                    print("\n4. Analyzing feature importance (all features, excluding target lags)...")
                    feature_groups_full = define_feature_groups(site_name=site)
                    all_features = []
                    for group_name, features in feature_groups_full.items():
                        if group_name != 'target_lag1_2_3':
                            all_features.extend(features)
                
                importance = feature_importance_analysis(
                    df, target_col, all_features, train_size, split_method,
                    save_path=f'{results_dir}/importance_{site}_{scenario}.png'
                )
                importance.to_csv(f'{results_dir}/importance_{site}_{scenario}.csv', index=False)
                scenario_results['importance'] = importance
                print("   ✓ Feature importance plot saved")
                print("   ✓ Top 10 features:")
                print(importance.head(10).to_string(index=False))
        
            # 5. Stacked Ensemble Training
            if run_ensemble:
                # Try to load ablation results from file if not in memory
                ablation_df = None
                if run_ablation and 'ablation' in scenario_results:
                    ablation_df = scenario_results['ablation']
                else:
                    # Try loading from saved file
                    ablation_file = f'{results_dir}/ablation_{site}_{scenario}.csv'
                    if os.path.exists(ablation_file):
                        ablation_df = pd.read_csv(ablation_file)
                        print(f"   Loaded ablation results from {ablation_file}")
                
                if ablation_df is not None:
                    # Use only consensus features
                    print("\n5. Training stacked ensemble (consensus features only, excluding target lags)...")
                    consensus_groups = ablation_df[ablation_df['consensus'] == True]['feature_group'].tolist()
                    
                    feature_groups_full = define_feature_groups(site_name=site)
                    all_features = []
                    for group_name, features in feature_groups_full.items():
                        if group_name != 'target_lag1_2_3' and group_name in consensus_groups:
                            all_features.extend(features)
                    
                    print(f"   Using {len(all_features)} features from {len(consensus_groups)} consensus groups")
                else:
                    # Fallback: use all features except target lags
                    print("\n5. Training stacked ensemble (all features, excluding target lags)...")
                    feature_groups_full = define_feature_groups(site_name=site)
                    all_features = []
                    for group_name, features in feature_groups_full.items():
                        if group_name != 'target_lag1_2_3':
                            all_features.extend(features)
                
                ensemble_result = train_stacked_ensemble(
                    df, target_col, all_features, 
                    n_folds=5, train_size=train_size, 
                    split_method=split_method, random_state=42,
                    save_dir=f'{results_dir}/models',
                    site_name=site,
                    scenario=scenario
                )
                
                # Save ensemble metrics summary
                ensemble_metrics = pd.DataFrame([ensemble_result['metrics']])
                ensemble_metrics.to_csv(f'{results_dir}/ensemble_metrics_{site}_{scenario}.csv', index=False)
                scenario_results['ensemble'] = ensemble_result
                print("   ✓ Ensemble model, predictions, and metrics saved")
            site_results[scenario] = scenario_results
        all_results[site] = site_results
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Results saved in: {results_dir}/")
    print("\nGenerated files:")
    if run_acf:
        print("- ACF/PACF plots for each site and variable")
    if run_ablation:
        print("- Feature ablation results with multi-algorithm consensus (CSV)")
        print("  Algorithms: RF/GB/CB/XGB/ET/MLP/LR (7 total)")
        print("  *_eff: True if both RMSE and MAE improved for that algorithm")
        print("  n_effective: Count of algorithms showing improvement")
        print("  consensus: True if >50% (>3.5, i.e., 4+) algorithms agree on effectiveness")
    if run_benchmark:
        print("- Algorithm benchmark results (CSV)")
        print("  Uses only consensus features from ablation study (if run)")
    if run_importance:
        print("- Feature importance rankings (CSV)")
        print("  Uses only consensus features from ablation study (if run)")
    if run_ensemble:
        print("- Stacked ensemble training results:")
        print("  * ensemble_metrics_{site}.csv - Performance metrics")
        print("  * models/ensemble_{site}.pkl - Trained ensemble model")
        print("  * models/predictions_{site}.csv - Predictions with ground truth")
        print("  * models/timeseries_{site}.png - Time series plot (train/test)")
        print("  * models/scatter_{site}.png - Scatter plot with fitted line (train/test)")
        print("  3-layer architecture: Base (7 models) → Stack (3 models) → Weighted Ensemble")
        print("  Uses only consensus features from ablation study (if run)")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run water demand prediction experiments')
    parser.add_argument('--acf', action='store_true', help='Run ACF/PACF analysis only')
    parser.add_argument('--ablation', action='store_true', help='Run feature ablation study only')
    parser.add_argument('--benchmark', action='store_true', help='Run algorithm benchmark only')
    parser.add_argument('--importance', action='store_true', help='Run feature importance analysis only')
    parser.add_argument('--ensemble', action='store_true', help='Run stacked ensemble training (saves models and predictions)')
    parser.add_argument('--all', action='store_true', default=True, help='Run all experiments (default)')
    
    args = parser.parse_args()
    
    # If any specific flag is set, disable default 'all'
    if args.acf or args.ablation or args.benchmark or args.importance or args.ensemble:
        args.all = False
    
    # Determine what to run
    run_acf = args.all or args.acf
    run_ablation = args.all or args.ablation
    run_benchmark = args.all or args.benchmark
    run_importance = args.all or args.importance
    run_ensemble = args.ensemble  # Only run if explicitly requested
    
    results = main(run_acf=run_acf, run_ablation=run_ablation, 
                   run_benchmark=run_benchmark, run_importance=run_importance,
                   run_ensemble=run_ensemble)
