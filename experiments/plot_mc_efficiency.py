"""Plots inference time vs metrics for the MC sample efficiency experiment.

Reads mc_efficiency_metrics.csv files from a given experiment directory,
aggregates the results across seeds, and generates line plots showing
the trade-off between number of MC samples, inference time, and metrics.
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Adjust plot styles
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

import paths as pt

def load_and_aggregate_results(exp_dir, dataset=None):
    """Loads all mc_efficiency_metrics.csv files in the exp_dir,
    optionally filtered by dataset, and aggregates across seeds."""
    all_files = list(Path(exp_dir).rglob("mc_efficiency_metrics.csv"))
    if not all_files:
        print(f"No mc_efficiency_metrics.csv files found in {exp_dir}")
        return None
        
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        if dataset and df["DatasetName"].iloc[0] != dataset:
            continue
        dfs.append(df)
        
    if not dfs:
        if dataset:
            print(f"No results found for dataset {dataset}")
        return None
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Identify unique groups
    group_cols = ["DatasetName", "ModelName", "LossConfig", "MC_Samples"]
    metric_cols = ["InferenceTime_sec", "TimePerPatient_ms", "CI", "IBS", "DCalib", "ICI", "INBLL", "MAEHinge"]
    
    # Calculate Mean and Std across seeds for each group
    agg_funcs = {col: ["mean", "std"] for col in metric_cols}
    agg_funcs["Seed"] = "count" # track n_seeds
    
    agg_df = full_df.groupby(group_cols).agg(agg_funcs).reset_index()
    
    # Flatten multi-level columns
    agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns.values]
    agg_df.rename(columns={'Seed_count': 'n_seeds'}, inplace=True)
    
    return agg_df

def plot_efficiency_curves(agg_df, out_dir, dataset):
    """Plots Metric vs MC Samples and Metric vs Inference Time."""
    
    metrics_to_plot = [
        ("CI", "CI_mean", "CI_std", "Concordance Index (CI)", True),  # True = higher is better
        ("IBS", "IBS_mean", "IBS_std", "Integrated Brier Score (IBS)", False),
        ("INBLL", "INBLL_mean", "INBLL_std", "INBLL", False),
        ("Time", "TimePerPatient_ms_mean", "TimePerPatient_ms_std", "Time per patient (ms)", False)
    ]
    
    loss_configs = agg_df["LossConfig"].unique()
    
    # Set up colors/markers for diff loss configs
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_configs)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # 1. Plot Metrics vs MC Samples (Log scale x-axis)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    axes1 = axes1.flatten()
    
    for loss_idx, loss_cfg in enumerate(loss_configs):
        loss_df = agg_df[agg_df["LossConfig"] == loss_cfg].sort_values("MC_Samples")
        
        # Plot each metric
        for i, (metric_name, mean_col, std_col, ylabel, _) in enumerate(metrics_to_plot):
            ax = axes1[i]
            x = loss_df["MC_Samples"]
            y = loss_df[mean_col]
            yerr = loss_df[std_col]
            
            ax.plot(x, y, marker=markers[loss_idx % len(markers)], color=colors[loss_idx], 
                    label=loss_cfg, linewidth=2, markersize=8)
            ax.fill_between(x, y - yerr, y + yerr, color=colors[loss_idx], alpha=0.15)
            
            if loss_idx == 0: # Set labels once
                ax.set_xscale('log')
                ax.set_xticks([1, 5, 10, 25, 50, 100, 200, 500])
                ax.set_xticklabels([1, 5, 10, 25, 50, 100, 200, 500])
                ax.set_xlabel('Number of Monte Carlo Samples (S)')
                ax.set_ylabel(ylabel)
                ax.grid(True, linestyle='--', alpha=0.7)
                if metric_name == "Time":
                    ax.set_yscale('log')
    
    axes1[0].legend(title="Loss Configuration", loc='best')
    fig1.suptitle(f"{dataset} - Performance vs. MC Samples", fontsize=16)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(out_dir / f"mc_efficiency_vs_samples_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2. Plot Performance Tradeoff (Metric vs Time)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    metrics_tradeoff = metrics_to_plot[:3] # Exclude Time from y-axis
    
    for loss_idx, loss_cfg in enumerate(loss_configs):
        loss_df = agg_df[agg_df["LossConfig"] == loss_cfg].sort_values("MC_Samples")
        x_time = loss_df["TimePerPatient_ms_mean"]
        
        for i, (metric_name, mean_col, std_col, ylabel, _) in enumerate(metrics_tradeoff):
            ax = axes2[i]
            y = loss_df[mean_col]
            
            ax.plot(x_time, y, marker=markers[loss_idx % len(markers)], color=colors[loss_idx], 
                    label=loss_cfg, linewidth=2, markersize=8)
            
            # Annotate points with S=...
            for j, (x_val, y_val, s_val) in enumerate(zip(x_time, y, loss_df["MC_Samples"])):
                if s_val in [1, 10, 50, 200, 500]: # Annotate a subset to avoid clutter
                    ax.annotate(f"S={s_val}", (x_val, y_val), textcoords="offset points", 
                                xytext=(5,5), ha='left', fontsize=8, color=colors[loss_idx])
            
            if loss_idx == 0:
                ax.set_xscale('log')
                ax.set_xlabel('Inference Time per Patient (ms)')
                ax.set_ylabel(ylabel)
                ax.grid(True, linestyle='--', alpha=0.7)
    
    axes2[0].legend(title="Loss Configuration", loc='best')
    fig2.suptitle(f"{dataset} - Performance Trade-off (Time vs. Metric)", fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(out_dir / f"mc_efficiency_tradeoff_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"  Plots saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MC Sample Efficiency plots")
    parser.add_argument("--experiment-name", required=True, help="Name of the posthoc experiment folder")
    parser.add_argument("--dataset", nargs="+", default=["METABRIC"], help="Datasets to analyze")
    args = parser.parse_args()
    
    exp_dir = Path(pt.RESULTS_DIR) / args.experiment_name
    out_dir = exp_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n============================================================")
    print(f"Generating MC Sample Efficiency Report")
    print(f"  Experiment: {args.experiment_name}")
    print(f"============================================================")
    
    datasets = args.dataset
    if len(datasets) == 1 and datasets[0] == "ALL":
        # Discover datasets based on folders
        datasets = [d.name for d in exp_dir.iterdir() if d.is_dir() and d.name != "comparisons"]
        
    for ds in datasets:
        print(f"\nAnalyzing dataset: {ds}")
        agg_df = load_and_aggregate_results(exp_dir, dataset=ds)
        
        if agg_df is not None:
            # Save aggregated data
            agg_csv_path = out_dir / f"mc_efficiency_summary_{ds}.csv"
            agg_df.to_csv(agg_csv_path, index=False)
            print(f"  Saved aggregated summary to {agg_csv_path}")
            
            # Generate plots
            plot_efficiency_curves(agg_df, out_dir, ds)
            
    print(f"\n============================================================")
    print(f"  All outputs in: {out_dir}")
    print(f"============================================================")
