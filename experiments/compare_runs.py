"""Compare results across calibration-loss experiments.

Reads the hierarchical structure and aggregates across seeds.
Generates: summary tables (mean ± std), Pareto plots, bar charts, significance tests.

Usage:
    python experiments/compare_runs.py --experiment-name 20260316_calibration_loss
    python experiments/compare_runs.py --experiment-name 20260316_calibration_loss --dataset METABRIC
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import paths as pt

import warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib; set non-interactive backend for cluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

METRIC_COLS = ["CI", "IBS", "DCalib", "CCalib", "ICI", "MAEHinge", "MAEPseudo", "KM", "INBLL"]
HIGHER_BETTER = {"CI": True, "DCalib": True, "CCalib": True}
LOWER_BETTER = {"IBS": True, "ICI": True, "MAEHinge": True, "MAEPseudo": True, "KM": True, "INBLL": True}


def load_experiment_data(experiment_dir, dataset_filter=None):
    """Load all metrics.csv from the hierarchical structure."""
    experiment_dir = Path(experiment_dir)
    rows = []

    for dataset_dir in sorted(experiment_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in ("comparisons",):
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue

        for loss_dir in sorted(dataset_dir.iterdir()):
            if not loss_dir.is_dir():
                continue
            for seed_dir in sorted(loss_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                metrics_file = seed_dir / "metrics.csv"
                if metrics_file.exists():
                    df = pd.read_csv(metrics_file)
                    rows.append(df)

    if not rows:
        print("No results found!")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def aggregate_across_seeds(df):
    """Compute mean ± std per (Dataset, LossConfig) across seeds."""
    group_cols = ["DatasetName", "LossConfig", "LossType", "Lambda", "Mu", "ModelName"]
    agg_rows = []
    for key, grp in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_seeds"] = len(grp)
        row["Seeds"] = sorted(grp["Seed"].tolist())
        for m in METRIC_COLS:
            if m in grp.columns:
                vals = grp[m].dropna()
                row[f"{m}_mean"] = vals.mean()
                row[f"{m}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                row[f"{m}_values"] = vals.tolist()
        row["BestEpoch_mean"] = grp["BestEpoch"].mean()
        row["TrainTime_mean"] = grp["TrainTime"].mean()
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def run_significance_tests(df, baseline_config="cox"):
    """Run paired t-test and Wilcoxon signed-rank test between baseline and each other config."""
    results = []
    for dataset in df["DatasetName"].unique():
        ds_df = df[df["DatasetName"] == dataset]
        baseline = ds_df[ds_df["LossConfig"] == baseline_config]
        if baseline.empty:
            print(f"  Warning: no baseline '{baseline_config}' found for {dataset}")
            continue

        baseline_row = baseline.iloc[0]
        for _, row in ds_df.iterrows():
            if row["LossConfig"] == baseline_config:
                continue
            for m in ["CI", "IBS", "DCalib", "INBLL"]:
                vals_key = f"{m}_values"
                if vals_key not in row or vals_key not in baseline_row:
                    continue
                baseline_vals = baseline_row[vals_key]
                other_vals = row[vals_key]

                n = min(len(baseline_vals), len(other_vals))
                if n < 3:
                    continue
                bv = np.array(baseline_vals[:n])
                ov = np.array(other_vals[:n])

                # Paired t-test
                t_stat, t_pval = stats.ttest_rel(bv, ov)

                # Wilcoxon signed-rank (needs n >= 6 ideally, but will try)
                try:
                    w_stat, w_pval = stats.wilcoxon(bv, ov, alternative='two-sided')
                except ValueError:
                    w_stat, w_pval = np.nan, np.nan

                # Effect size (Cohen's d for paired samples)
                diff = bv - ov
                cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

                better = "baseline" if (m in HIGHER_BETTER and bv.mean() > ov.mean()) or \
                         (m in LOWER_BETTER and bv.mean() < ov.mean()) else row["LossConfig"]

                results.append({
                    "Dataset": dataset,
                    "Metric": m,
                    "Baseline": baseline_config,
                    "Compared": row["LossConfig"],
                    "Baseline_mean": bv.mean(),
                    "Compared_mean": ov.mean(),
                    "Diff_mean": diff.mean(),
                    "Cohen_d": cohens_d,
                    "t_stat": t_stat,
                    "t_pval": t_pval,
                    "w_stat": w_stat,
                    "w_pval": w_pval,
                    "n_pairs": n,
                    "Significant_005": t_pval < 0.05 if not np.isnan(t_pval) else False,
                    "Winner": better,
                })

    return pd.DataFrame(results)


def make_summary_table(agg_df, output_dir, dataset):
    """Create a readable summary CSV with mean ± std."""
    rows = []
    for _, r in agg_df[agg_df["DatasetName"] == dataset].iterrows():
        row = {
            "Loss Config": r["LossConfig"],
            "Seeds": r["n_seeds"],
        }
        for m in METRIC_COLS:
            mean_val = r.get(f"{m}_mean", np.nan)
            std_val = r.get(f"{m}_std", np.nan)
            if not np.isnan(mean_val):
                row[m] = f"{mean_val:.4f} ± {std_val:.4f}"
        row["Avg Epoch"] = f"{r['BestEpoch_mean']:.1f}"
        row["Avg Train(s)"] = f"{r['TrainTime_mean']:.1f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = output_dir / f"summary_{dataset}.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  Summary: {out_path}")

    # Also print to stdout
    print(f"\n  === {dataset} Summary (mean ± std across {agg_df['n_seeds'].iloc[0]} seeds) ===")
    print(summary_df.to_string(index=False))
    return summary_df


def plot_pareto_frontier(agg_df, output_dir, dataset):
    """Plot CI vs D-Cal with error bars across seeds."""
    ds = agg_df[agg_df["DatasetName"] == dataset].copy()
    if ds.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ds)))

    for i, (_, row) in enumerate(ds.iterrows()):
        ax.errorbar(
            row["CI_mean"], row["DCalib_mean"],
            xerr=row["CI_std"], yerr=row["DCalib_std"],
            fmt='o', markersize=10, capsize=5, capthick=1.5,
            color=colors[i], label=row["LossConfig"],
            zorder=3
        )

    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='D-Cal p=0.05')
    ax.set_xlabel("Concordance Index (CI) ↑", fontsize=13)
    ax.set_ylabel("D-Calibration p-value ↑", fontsize=13)
    ax.set_title(f"Discrimination vs Calibration — {dataset}\n(mean ± std across seeds)", fontsize=14)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / f"pareto_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Pareto plot: {out_path}")


def plot_metric_bars(agg_df, output_dir, dataset):
    """Bar chart comparing key metrics across loss configs with error bars."""
    ds = agg_df[agg_df["DatasetName"] == dataset].copy()
    if ds.empty:
        return

    metrics_to_plot = ["CI", "IBS", "DCalib", "INBLL"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 6))

    x = np.arange(len(ds))
    labels = ds["LossConfig"].tolist()

    for ax, m in zip(axes, metrics_to_plot):
        means = ds[f"{m}_mean"].values
        stds = ds[f"{m}_std"].values
        bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Colour best bar green
        if m in HIGHER_BETTER:
            best_idx = np.argmax(means)
        else:
            best_idx = np.argmin(means)
        bars[best_idx].set_facecolor('#2ecc71')

        ax.set_title(m, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f"Metric Comparison — {dataset} (mean ± std)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / f"metrics_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Metric bars: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare calibration loss experiment results")
    parser.add_argument("--experiment-name", required=True, help="Experiment session name")
    parser.add_argument("--dataset", default=None, help="Filter to one dataset")
    args = parser.parse_args()

    experiment_dir = Path(pt.RESULTS_DIR) / args.experiment_name
    if not experiment_dir.exists():
        print(f"ERROR: {experiment_dir} does not exist")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Comparing results: {experiment_dir}")
    print(f"{'='*60}")

    # Load all data
    df = load_experiment_data(experiment_dir, args.dataset)
    if df.empty:
        return
    print(f"  Loaded {len(df)} individual results")

    # Aggregate across seeds
    agg_df = aggregate_across_seeds(df)
    print(f"  Found {len(agg_df)} unique configurations")

    # Create output dir
    comp_dir = experiment_dir / "comparisons"
    comp_dir.mkdir(exist_ok=True)

    # Per-dataset analysis
    for dataset in sorted(df["DatasetName"].unique()):
        print(f"\n{'#'*40}")
        print(f"# {dataset}")
        print(f"{'#'*40}")
        make_summary_table(agg_df, comp_dir, dataset)
        plot_pareto_frontier(agg_df, comp_dir, dataset)
        plot_metric_bars(agg_df, comp_dir, dataset)

    # Significance tests
    if agg_df["n_seeds"].max() >= 3:
        print(f"\n{'='*60}")
        print("SIGNIFICANCE TESTS (paired t-test, Wilcoxon)")
        print(f"{'='*60}")
        sig_df = run_significance_tests(agg_df, baseline_config="cox")
        if not sig_df.empty:
            sig_df.to_csv(comp_dir / "significance_tests.csv", index=False)
            print(f"  Saved: {comp_dir / 'significance_tests.csv'}")

            # Print significant results
            sig_only = sig_df[sig_df["Significant_005"]]
            if not sig_only.empty:
                print("\n  Statistically significant differences (p < 0.05):")
                for _, r in sig_only.iterrows():
                    direction = "↑" if r["Winner"] != "cox" else "↓"
                    print(f"    {r['Dataset']} | {r['Metric']}: "
                          f"{r['Baseline']}: {r['Baseline_mean']:.4f} vs "
                          f"{r['Compared']}: {r['Compared_mean']:.4f} "
                          f"(p={r['t_pval']:.4f}, d={r['Cohen_d']:.2f}) {direction}")
            else:
                print("  No statistically significant differences found at p < 0.05")
    else:
        print("\n  Skipping significance tests (need at least 3 seeds)")

    # Save aggregated results
    agg_out = agg_df.drop(columns=[c for c in agg_df.columns if c.endswith("_values")], errors='ignore')
    agg_out.to_csv(comp_dir / "aggregated_results.csv", index=False)
    print(f"\n  Aggregated results: {comp_dir / 'aggregated_results.csv'}")
    print(f"\nDone! All outputs in: {comp_dir}")


if __name__ == "__main__":
    main()
