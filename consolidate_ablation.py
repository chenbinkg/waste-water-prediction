"""
Consolidate feature ablation results across all sites and scenarios
"""
import pandas as pd
import glob
import os

def consolidate_ablation_results(results_dir='results'):
    """Consolidate ablation results and calculate consensus percentage"""
    
    # Find all ablation CSV files
    ablation_files = glob.glob(f'{results_dir}/ablation_*.csv')
    
    if not ablation_files:
        print(f"No ablation files found in {results_dir}")
        return None
    
    print(f"Found {len(ablation_files)} ablation files")
    
    # Collect all results
    all_data = []
    for file in ablation_files:
        # Extract site and scenario from filename
        basename = os.path.basename(file).replace('ablation_', '').replace('.csv', '')
        parts = basename.rsplit('_', 1)
        site = parts[0] if len(parts) > 1 else basename
        scenario = parts[1] if len(parts) > 1 else 'unknown'
        
        df = pd.read_csv(file)
        df['site'] = site
        df['scenario'] = scenario
        all_data.append(df)
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Calculate consensus percentage for each feature group
    consensus_summary = combined.groupby('feature_group').agg({
        'consensus': lambda x: (x.sum() / len(x) * 100),  # Percentage
        'n_features': 'first',  # Same for all
        'n_effective': 'mean'  # Average across sites/scenarios
    }).reset_index()
    
    consensus_summary.columns = ['feature_group', 'consensus_pct', 'n_features', 'avg_n_effective']
    consensus_summary = consensus_summary.sort_values('consensus_pct', ascending=False)
    
    # Save consolidated results
    combined.to_csv(f'{results_dir}/ablation_all_combined.csv', index=False)
    consensus_summary.to_csv(f'{results_dir}/ablation_consensus_summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print("FEATURE ABLATION CONSENSUS SUMMARY")
    print(f"{'='*60}")
    print(f"Total sites x scenarios: {len(ablation_files)}")
    print(f"\nConsensus Percentage by Feature Group:")
    print(consensus_summary.to_string(index=False))
    print(f"\n{'='*60}")
    print(f"Saved to:")
    print(f"  - {results_dir}/ablation_all_combined.csv")
    print(f"  - {results_dir}/ablation_consensus_summary.csv")
    
    return consensus_summary

if __name__ == '__main__':
    consolidate_ablation_results()
