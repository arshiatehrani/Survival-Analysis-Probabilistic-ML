"""Compare Post-Hoc Calibration Results.

Aggregates posthoc_metrics.csv files across seeds, computes mean ± std,
performs paired t-tests (before vs after calibration), and generates
comparison tables and plots.

Usage:
    python experiments/compare_posthoc.py \
        --experiment-name 20260317_posthoc_calibration \
        --dataset METABRIC
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon

import paths as pt


METRICS = ["CI", "IBS", "DCalib", "ICI", "INBLL", "MAEHinge", "MAEPseudo", "KM"]
KEY_METRICS = ["CI", "IBS", "DCalib", "ICI", "INBLL"]
HIGHER_BETTER = {"CI": True, "DCalib": True}
LOWER_BETTER = {"IBS": True, "ICI": True, "INBLL": True, "MAEHinge": True, "MAEPseudo": True, "KM": True}


def load_all_results(experiment_dir, dataset=None):
    """Load all posthoc_metrics.csv files."""
    rows = []
    for csv_path in experiment_dir.rglob("posthoc_metrics.csv"):
        df = pd.read_csv(csv_path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    if dataset:
        all_df = all_df[all_df["DatasetName"] == dataset]
    return all_df


def aggregate(df):
    """Aggregate results across seeds: mean ± std per (LossConfig, Calibration)."""
    group_cols = ["DatasetName", "LossConfig", "Calibration"]
    agg = df.groupby(group_cols).agg(
        n_seeds=("Seed", "count"),
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std": (m, "std") for m in METRICS},
        Temperature_T_mean=("Temperature_T", "mean"),
    ).reset_index()
    return agg


def significance_tests(df, baseline_calib="none"):
    """Paired t-tests: 'none' vs 'temp_scaling' and 'none' vs 'isotonic'."""
    results = []
    datasets = df["DatasetName"].unique()
    loss_configs = df["LossConfig"].unique()
    calibrations = [c for c in df["Calibration"].unique() if c != baseline_calib]

    for ds in datasets:
        for lc in loss_configs:
            for calib in calibrations:
                for metric in KEY_METRICS:
                    baseline = df[(df["DatasetName"] == ds) &
                                  (df["LossConfig"] == lc) &
                                  (df["Calibration"] == baseline_calib)]
                    compared = df[(df["DatasetName"] == ds) &
                                  (df["LossConfig"] == lc) &
                                  (df["Calibration"] == calib)]

                    if len(baseline) < 3 or len(compared) < 3:
                        continue

                    # Match by seed
                    merged = baseline.merge(compared, on="Seed", suffixes=("_base", "_comp"))
                    if len(merged) < 3:
                        continue

                    base_vals = merged[f"{metric}_base"].values
                    comp_vals = merged[f"{metric}_comp"].values

                    t_stat, t_pval = ttest_rel(base_vals, comp_vals)

                    try:
                        w_stat, w_pval = wilcoxon(base_vals, comp_vals)
                    except ValueError:
                        w_stat, w_pval = np.nan, np.nan

                    diff = comp_vals - base_vals
                    pooled_std = np.sqrt((np.std(base_vals)**2 + np.std(comp_vals)**2) / 2)
                    cohen_d = diff.mean() / pooled_std if pooled_std > 0 else 0

                    # Determine winner
                    if metric in HIGHER_BETTER:
                        winner = calib if diff.mean() > 0 else "none"
                    else:
                        winner = calib if diff.mean() < 0 else "none"

                    results.append({
                        "Dataset": ds, "LossConfig": lc, "Metric": metric,
                        "Baseline": baseline_calib, "Compared": calib,
                        "Baseline_mean": base_vals.mean(), "Compared_mean": comp_vals.mean(),
                        "Diff_mean": diff.mean(), "Cohen_d": cohen_d,
                        "t_stat": t_stat, "t_pval": t_pval,
                        "w_stat": w_stat, "w_pval": w_pval,
                        "n_pairs": len(merged), "Winner": winner,
                    })

    sig_df = pd.DataFrame(results)
    if len(sig_df) > 0:
        n_tests = len(sig_df)
        sig_df["t_pval_adj"] = (sig_df["t_pval"] * n_tests).clip(upper=1.0)
        sig_df["w_pval_adj"] = (sig_df["w_pval"] * n_tests).clip(upper=1.0)
        sig_df["Significant_005"] = sig_df["t_pval_adj"] < 0.05
        sig_df["Significant_001"] = sig_df["t_pval_adj"] < 0.01
    return sig_df


def plot_before_after(df, dataset, out_dir):
    """Bar chart comparing metrics before and after calibration."""
    for loss_config in df["LossConfig"].unique():
        subset = df[(df["DatasetName"] == dataset) & (df["LossConfig"] == loss_config)]
        if len(subset) == 0:
            continue

        fig, axes = plt.subplots(1, len(KEY_METRICS), figsize=(4*len(KEY_METRICS), 5))
        if len(KEY_METRICS) == 1:
            axes = [axes]

        calibrations = ["none", "temp_scaling", "isotonic"]
        colors = ["#5470C6", "#EE6666", "#91CC75"]
        labels = ["None", "Temp Scaling", "Isotonic"]

        for ax_idx, metric in enumerate(KEY_METRICS):
            ax = axes[ax_idx]
            means = []
            stds = []
            for calib in calibrations:
                calib_data = subset[subset["Calibration"] == calib][metric]
                means.append(calib_data.mean() if len(calib_data) > 0 else 0)
                stds.append(calib_data.std() if len(calib_data) > 1 else 0)

            x = np.arange(len(calibrations))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                         edgecolor='white', linewidth=0.5)
            ax.set_title(metric, fontweight='bold', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f"{dataset} — {loss_config}: Post-Hoc Calibration",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(out_dir / f"posthoc_bars_{dataset}_{loss_config}.pdf",
                    bbox_inches='tight', dpi=150)
        plt.close(fig)


def plot_dcal_improvement(df, dataset, out_dir):
    """Scatter plot: D-Cal change after calibration for each loss config."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"temp_scaling": "#EE6666", "isotonic": "#91CC75"}
    markers = {"temp_scaling": "o", "isotonic": "s"}
    labels_map = {"temp_scaling": "Temperature Scaling", "isotonic": "Isotonic Regression"}

    for calib in ["temp_scaling", "isotonic"]:
        data_by_lc = []
        for lc in df["LossConfig"].unique():
            base = df[(df["DatasetName"] == dataset) & (df["LossConfig"] == lc) &
                       (df["Calibration"] == "none")]
            comp = df[(df["DatasetName"] == dataset) & (df["LossConfig"] == lc) &
                       (df["Calibration"] == calib)]
            if len(base) > 0 and len(comp) > 0:
                data_by_lc.append({
                    "LossConfig": lc,
                    "DCalib_before": base["DCalib"].mean(),
                    "DCalib_after": comp["DCalib"].mean(),
                    "CI_before": base["CI"].mean(),
                    "CI_after": comp["CI"].mean(),
                })

        if data_by_lc:
            plot_df = pd.DataFrame(data_by_lc)
            ax.scatter(plot_df["DCalib_before"], plot_df["DCalib_after"],
                      c=colors[calib], marker=markers[calib], s=80,
                      label=labels_map[calib], alpha=0.8, edgecolors='white', zorder=3)
            for _, row in plot_df.iterrows():
                ax.annotate(row["LossConfig"], (row["DCalib_before"], row["DCalib_after"]),
                           fontsize=7, ha='left', va='bottom')

    # Diagonal line (no improvement)
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, label="No change")
    ax.axhline(y=0.05, color='red', linestyle=':', alpha=0.4, label="p=0.05 threshold")

    ax.set_xlabel("D-Cal Before Calibration", fontsize=11)
    ax.set_ylabel("D-Cal After Calibration", fontsize=11)
    ax.set_title(f"{dataset}: D-Calibration Improvement", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / f"posthoc_dcal_{dataset}.pdf", bbox_inches='tight', dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare post-hoc calibration results")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    experiment_dir = Path(pt.RESULTS_DIR) / args.experiment_name
    out_dir = experiment_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Post-Hoc Calibration Comparison")
    print(f"  Experiment: {args.experiment_name}")
    print(f"{'='*60}")

    # Load all results
    df = load_all_results(experiment_dir, args.dataset)
    if df.empty:
        print("  No results found!")
        sys.exit(1)

    datasets = df["DatasetName"].unique()
    print(f"  Datasets: {list(datasets)}")
    print(f"  Total rows: {len(df)}")

    # Aggregated results
    agg = aggregate(df)
    agg.to_csv(out_dir / "aggregated_posthoc.csv", index=False)
    print(f"\n  Aggregated results: {out_dir / 'aggregated_posthoc.csv'}")

    # Per-dataset summaries and plots
    for ds in datasets:
        ds_df = df[df["DatasetName"] == ds]
        ds_agg = agg[agg["DatasetName"] == ds]

        # Print summary table
        print(f"\n  === {ds} Post-Hoc Summary (mean ± std across seeds) ===")
        for _, row in ds_agg.iterrows():
            parts = [f"  {row['LossConfig']:<20} {row['Calibration']:<15} (n={int(row['n_seeds'])})"]
            for m in KEY_METRICS:
                parts.append(f"{m}={row[f'{m}_mean']:.4f}±{row[f'{m}_std']:.4f}")
            print(" | ".join(parts))

        # Summary CSV
        summary_path = out_dir / f"summary_posthoc_{ds}.csv"
        ds_agg.to_csv(summary_path, index=False)
        print(f"  Summary: {summary_path}")

        # Plots
        plot_before_after(ds_df, ds, out_dir)
        plot_dcal_improvement(ds_df, ds, out_dir)

    # Significance tests
    sig_df = significance_tests(df)
    if len(sig_df) > 0:
        sig_path = out_dir / "significance_posthoc.csv"
        sig_df.to_csv(sig_path, index=False)
        print(f"\n  Significance tests: {sig_path}")

        # Print highlights
        sig_results = sig_df[sig_df["t_pval"] < 0.05]
        if len(sig_results) > 0:
            print(f"\n  === Significant improvements (unadjusted p < 0.05) ===")
            for _, row in sig_results.iterrows():
                direction = "↑" if row["Diff_mean"] > 0 else "↓"
                print(f"    {row['Dataset']} | {row['LossConfig']} | {row['Metric']} | "
                      f"{row['Baseline']} → {row['Compared']}: "
                      f"{direction}{abs(row['Diff_mean']):.4f} (p={row['t_pval']:.4f}, d={row['Cohen_d']:.2f})")

    print(f"\n{'='*60}")
    print(f"  All outputs in: {out_dir}")
    print(f"{'='*60}")
