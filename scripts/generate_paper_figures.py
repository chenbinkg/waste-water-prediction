"""
Generate publication-quality figures for technical paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

def plot_ablation_comparison(results_dir, sites, save_path='results/figures'):
    """Compare feature ablation across sites"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ['rmse', 'mae', 'r2']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for site in sites:
            df = pd.read_csv(f'{results_dir}/ablation_{site}.csv')
            ax.plot(df['feature_group'], df[metric], marker='o', label=site)
        
        ax.set_xlabel('Feature Group')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Feature Group')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/ablation_comparison.png', bbox_inches='tight')
    plt.close()

def plot_algorithm_comparison(results_dir, sites, save_path='results/figures'):
    """Compare algorithms across sites"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(sites))
    width = 0.12
    
    all_algos = pd.read_csv(f'{results_dir}/algorithms_{sites[0]}.csv')['algorithm'].tolist()
    
    for i, algo in enumerate(all_algos):
        rmse_values = []
        for site in sites:
            df = pd.read_csv(f'{results_dir}/algorithms_{site}.csv')
            rmse = df[df['algorithm'] == algo]['rmse'].values[0]
            rmse_values.append(rmse)
        
        ax.bar(x_pos + i*width, rmse_values, width, label=algo)
    
    ax.set_xlabel('Site')
    ax.set_ylabel('RMSE')
    ax.set_title('Algorithm Performance Comparison')
    ax.set_xticks(x_pos + width * 3)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/algorithm_comparison.png', bbox_inches='tight')
    plt.close()

def plot_feature_importance(results_dir, site, top_n=15, save_path='results/figures'):
    """Plot top N important features"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    df = pd.read_csv(f'{results_dir}/importance_{site}.csv')
    df_top = df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_top['feature'], df_top['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features - {site}')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/importance_{site}.png', bbox_inches='tight')
    plt.close()

def plot_acf_summary(results_dir, site, variable, save_path='results/figures'):
    """Create ACF summary plot"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Copy existing ACF plots
    import shutil
    src = f'{results_dir}/acf/{site}/{variable}_acf_pacf.png'
    dst = f'{save_path}/acf_{site}_{variable}.png'
    if os.path.exists(src):
        shutil.copy(src, dst)

def generate_statistics_table(results_dir, sites):
    """Generate summary statistics table"""
    summary = []
    
    for site in sites:
        # Best algorithm
        algo_df = pd.read_csv(f'{results_dir}/algorithms_{site}.csv')
        best_algo = algo_df.loc[algo_df['rmse'].idxmin()]
        
        # Feature ablation improvement
        ablation_df = pd.read_csv(f'{results_dir}/ablation_{site}.csv')
        baseline_rmse = ablation_df.iloc[0]['rmse']
        final_rmse = ablation_df.iloc[-1]['rmse']
        improvement = (baseline_rmse - final_rmse) / baseline_rmse * 100
        
        summary.append({
            'Site': site,
            'Best Algorithm': best_algo['algorithm'],
            'Best RMSE': best_algo['rmse'],
            'Best R²': best_algo['r2'],
            'Feature Improvement (%)': improvement
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{results_dir}/summary_statistics.csv', index=False)
    return summary_df

if __name__ == "__main__":
    results_dir = 'results'
    sites = ['Lower Hutt', 'Wellington High (Moa)', 'North Wellington (Moa)']
    
    print("Generating figures for technical paper...")
    
    plot_ablation_comparison(results_dir, sites)
    print("✓ Ablation comparison plot")
    
    plot_algorithm_comparison(results_dir, sites)
    print("✓ Algorithm comparison plot")
    
    for site in sites:
        plot_feature_importance(results_dir, site)
    print("✓ Feature importance plots")
    
    summary = generate_statistics_table(results_dir, sites)
    print("✓ Summary statistics table")
    print("\n", summary.to_string(index=False))
