"""Compare results across uncertainty-aware training experiments.

Reads the hierarchical structure and aggregates across seeds.
Generates: summary tables, Pareto plots, bar charts, box plots, uncertainty evolution.

Usage:
    python experiments/compare_uncertainty_runs.py --experiment-name 20260329_uncertainty_training
    python experiments/compare_uncertainty_runs.py --experiment-name 20260329_uncertainty_training --dataset METABRIC
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRIC_COLS = ["CI", "IBS", "DCalib", "CCalib", "ICI", "MAEHinge", "MAEPseudo", "KM", "INBLL"]
HIGHER_BETTER = {"CI": True, "DCalib": True, "CCalib": True}
LOWER_BETTER = {"IBS": True, "ICI": True, "MAEHinge": True, "MAEPseudo": True, "KM": True, "INBLL": True}


def load_experiment_data(experiment_dir, dataset_filter=None):
    experiment_dir = Path(experiment_dir)
    rows = []

    for dataset_dir in sorted(experiment_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in ("comparisons",):
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue

        for config_dir in sorted(dataset_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            for seed_dir in sorted(config_dir.iterdir()):
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


def load_uncertainty_histories(experiment_dir, dataset_filter=None):
    """Load uncertainty_history.csv from all runs for evolution plots."""
    experiment_dir = Path(experiment_dir)
    histories = []

    for dataset_dir in sorted(experiment_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in ("comparisons",):
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue

        for config_dir in sorted(dataset_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            for seed_dir in sorted(config_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                hist_file = seed_dir / "uncertainty_history.csv"
                if hist_file.exists():
                    h = pd.read_csv(hist_file)
                    config_file = seed_dir / "config.json"
                    if config_file.exists():
                        import json
                        with open(config_file) as f:
                            cfg = json.load(f)
                        h["dataset"] = cfg.get("dataset", "unknown")
                        h["unc_mode"] = cfg.get("unc_mode", "unknown")
                        h["loss_config"] = cfg.get("loss_config", "unknown")
                        h["seed"] = cfg.get("seed", -1)
                        h["temperature"] = cfg.get("temperature", 0)
                        h["warmup_epochs"] = cfg.get("warmup_epochs", 0)
                    histories.append(h)

    if not histories:
        return pd.DataFrame()
    return pd.concat(histories, ignore_index=True)


def aggregate_across_seeds(df):
    group_cols = [c for c in ["DatasetName", "LossConfig", "LossType", "UncMode",
                              "Lambda", "Mu", "Temperature", "WarmupEpochs",
                              "McPasses", "CurriculumStart", "CurriculumEnd",
                              "ModelName"] if c in df.columns]
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
        if "BestEpoch" in grp.columns:
            row["BestEpoch_mean"] = grp["BestEpoch"].mean()
        if "TrainTime" in grp.columns:
            row["TrainTime_mean"] = grp["TrainTime"].mean()
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def run_significance_tests(df, baseline_mode="none"):
    """Paired t-test between baseline (unc_mode=none) and each other config."""
    results = []
    for dataset in df["DatasetName"].unique():
        ds_df = df[df["DatasetName"] == dataset]
        if "UncMode" in ds_df.columns:
            baseline = ds_df[ds_df["UncMode"] == baseline_mode]
        else:
            baseline = ds_df[ds_df["LossConfig"].str.startswith("unc_none")]

        if baseline.empty:
            print(f"  Warning: no baseline mode='{baseline_mode}' found for {dataset}")
            continue

        baseline_row = baseline.iloc[0]
        for _, row in ds_df.iterrows():
            if row.get("UncMode", "") == baseline_mode:
                continue
            for m in ["CI", "IBS", "DCalib", "CCalib", "INBLL"]:
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

                t_stat, t_pval = stats.ttest_rel(bv, ov)
                try:
                    w_stat, w_pval = stats.wilcoxon(bv, ov, alternative='two-sided')
                except ValueError:
                    w_stat, w_pval = np.nan, np.nan

                diff = bv - ov
                cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

                better = "baseline" if (m in HIGHER_BETTER and bv.mean() > ov.mean()) or \
                         (m in LOWER_BETTER and bv.mean() < ov.mean()) else row.get("LossConfig", "other")

                results.append({
                    "Dataset": dataset,
                    "Metric": m,
                    "Baseline": f"unc_{baseline_mode}",
                    "Compared": row.get("LossConfig", ""),
                    "UncMode": row.get("UncMode", ""),
                    "Baseline_mean": bv.mean(),
                    "Compared_mean": ov.mean(),
                    "Diff_mean": diff.mean(),
                    "Cohen_d": cohens_d,
                    "t_stat": t_stat,
                    "t_pval": t_pval,
                    "w_stat": w_stat,
                    "w_pval": w_pval,
                    "n_pairs": n,
                    "Winner": better,
                })

    sig_df = pd.DataFrame(results)
    if sig_df.empty:
        return sig_df

    for col_raw, col_adj in [("t_pval", "t_pval_adj"), ("w_pval", "w_pval_adj")]:
        adjusted = []
        for _, grp in sig_df.groupby(["Dataset", "Metric"]):
            n_comparisons = len(grp)
            adj = grp[col_raw].clip(upper=1.0) * n_comparisons
            adjusted.append(adj.clip(upper=1.0))
        sig_df[col_adj] = pd.concat(adjusted).sort_index()

    sig_df["Significant_005"] = sig_df["t_pval_adj"] < 0.05
    sig_df["Significant_001"] = sig_df["t_pval_adj"] < 0.01

    return sig_df


def make_summary_table(agg_df, output_dir, dataset):
    rows = []
    for _, r in agg_df[agg_df["DatasetName"] == dataset].iterrows():
        row = {
            "Unc Mode": r.get("UncMode", ""),
            "Loss Config": r.get("LossConfig", ""),
            "Seeds": r["n_seeds"],
        }
        for m in METRIC_COLS:
            mean_val = r.get(f"{m}_mean", np.nan)
            std_val = r.get(f"{m}_std", np.nan)
            if not np.isnan(mean_val):
                row[m] = f"{mean_val:.4f} ± {std_val:.4f}"
        row["Avg Epoch"] = f"{r.get('BestEpoch_mean', 0):.1f}"
        row["Avg Train(s)"] = f"{r.get('TrainTime_mean', 0):.1f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = output_dir / f"summary_{dataset}.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  Summary: {out_path}")

    print(f"\n  === {dataset} Summary (mean ± std) ===")
    print(summary_df.to_string(index=False))
    return summary_df


def plot_pareto_frontier(agg_df, output_dir, dataset):
    ds = agg_df[agg_df["DatasetName"] == dataset].copy()
    if ds.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    mode_colors = {"none": "#e74c3c", "soft": "#3498db", "curriculum": "#2ecc71", "both": "#9b59b6"}

    for _, row in ds.iterrows():
        mode = row.get("UncMode", "none")
        color = mode_colors.get(mode, "#95a5a6")
        label = row.get("LossConfig", "")
        if len(label) > 40:
            label = label[:37] + "..."
        ax.errorbar(
            row["CI_mean"], row["DCalib_mean"],
            xerr=row["CI_std"], yerr=row["DCalib_std"],
            fmt='o', markersize=8, capsize=4, capthick=1.2,
            color=color, label=label, zorder=3
        )

    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='D-Cal p=0.05')
    ax.set_xlabel("Concordance Index (CI) ↑", fontsize=13)
    ax.set_ylabel("D-Calibration p-value ↑", fontsize=13)
    ax.set_title(f"CI vs D-Calibration — {dataset}\n(Uncertainty Training)", fontsize=14)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / f"pareto_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Pareto plot: {out_path}")


def plot_metric_bars(agg_df, output_dir, dataset):
    ds = agg_df[agg_df["DatasetName"] == dataset].copy()
    if ds.empty:
        return

    metrics_to_plot = ["CI", "IBS", "DCalib", "CCalib", "INBLL"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 7))

    x = np.arange(len(ds))
    labels = []
    for _, r in ds.iterrows():
        mode = r.get("UncMode", "")
        lc = r.get("LossConfig", "")
        short = f"{mode}" if mode else lc[:20]
        labels.append(short)

    mode_colors = {"none": "#e74c3c", "soft": "#3498db", "curriculum": "#2ecc71", "both": "#9b59b6"}

    for ax, m in zip(axes, metrics_to_plot):
        means = ds[f"{m}_mean"].values
        stds = ds[f"{m}_std"].values
        colors = [mode_colors.get(r.get("UncMode", "none"), "#95a5a6") for _, r in ds.iterrows()]
        bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.85, edgecolor='black',
                      linewidth=0.5, color=colors)

        if m in HIGHER_BETTER:
            best_idx = np.argmax(means)
        else:
            best_idx = np.argmin(means)
        bars[best_idx].set_edgecolor('#f39c12')
        bars[best_idx].set_linewidth(3)

        ax.set_title(m, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=65, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f"Metric Comparison — {dataset} (Uncertainty Training)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / f"metrics_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Metric bars: {out_path}")


def plot_box_plots(raw_df, output_dir, dataset):
    ds = raw_df[raw_df["DatasetName"] == dataset].copy()
    if ds.empty:
        return

    metrics_to_plot = ["CI", "IBS", "DCalib", "CCalib", "INBLL"]
    loss_configs = sorted(ds["LossConfig"].unique())
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))

    for ax, m in zip(axes, metrics_to_plot):
        data_per_config = []
        labels = []
        for lc in loss_configs:
            vals = ds.loc[ds["LossConfig"] == lc, m].dropna().values
            if len(vals) > 0:
                data_per_config.append(vals)
                short = lc[:25] + "..." if len(lc) > 25 else lc
                labels.append(short)

        if not data_per_config:
            continue

        bp = ax.boxplot(data_per_config, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=5),
                        medianprops=dict(color='black', linewidth=1.5))

        colors = plt.cm.Set3(np.linspace(0, 1, len(data_per_config)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_title(m, fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=65, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f"Metric Distributions — {dataset} (Uncertainty Training)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / f"boxplots_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Box plots: {out_path}")


def plot_uncertainty_evolution(hist_df, output_dir, dataset):
    """Plot how uncertainty stats evolve over epochs for each config."""
    ds = hist_df[hist_df["dataset"] == dataset].copy()
    if ds.empty:
        return

    active = ds[ds["phase"] == "active"]
    if active.empty:
        return

    configs = active["loss_config"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for config in configs:
        cfg_data = active[active["loss_config"] == config]
        avg = cfg_data.groupby("epoch").agg({
            "unc_mean": "mean",
            "kept_frac": "mean",
            "kept_unc_mean": "mean",
        }).reset_index()

        short = config[:30] + "..." if len(config) > 30 else config
        axes[0].plot(avg["epoch"], avg["unc_mean"], marker='.', markersize=3, label=short)
        axes[1].plot(avg["epoch"], avg["kept_frac"], marker='.', markersize=3, label=short)
        axes[2].plot(avg["epoch"], avg["kept_unc_mean"], marker='.', markersize=3, label=short)

    axes[0].set_title("Mean Uncertainty (all samples)", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalised MC Variance")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Fraction of Data Kept (curriculum)", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Kept Fraction")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Mean Uncertainty (kept subset)", fontsize=12)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Normalised MC Variance")
    axes[2].grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc='best')
    plt.suptitle(f"Uncertainty Evolution — {dataset}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / f"uncertainty_evolution_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Uncertainty evolution: {out_path}")


def plot_mode_comparison(agg_df, output_dir, dataset):
    """Grouped bar chart comparing unc_modes directly."""
    ds = agg_df[agg_df["DatasetName"] == dataset].copy()
    if ds.empty or "UncMode" not in ds.columns:
        return

    modes = sorted(ds["UncMode"].unique())
    mode_colors = {"none": "#e74c3c", "soft": "#3498db", "curriculum": "#2ecc71", "both": "#9b59b6"}

    metrics_to_plot = ["CI", "IBS", "CCalib", "INBLL"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 6))

    for ax, m in zip(axes, metrics_to_plot):
        x = np.arange(len(modes))
        for i, mode in enumerate(modes):
            mode_data = ds[ds["UncMode"] == mode]
            if mode_data.empty:
                continue
            mean_val = mode_data[f"{m}_mean"].mean()
            std_val = mode_data[f"{m}_std"].mean()
            ax.bar(i, mean_val, yerr=std_val, capsize=5, alpha=0.85,
                   color=mode_colors.get(mode, "#95a5a6"), edgecolor='black',
                   linewidth=0.5, label=mode if ax == axes[0] else "")

        ax.set_title(m, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modes, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    axes[0].legend(fontsize=10)
    plt.suptitle(f"Uncertainty Mode Comparison — {dataset}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / f"mode_comparison_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Mode comparison: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare uncertainty training experiment results")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    experiment_dir = Path(pt.RESULTS_DIR) / args.experiment_name
    if not experiment_dir.exists():
        print(f"ERROR: {experiment_dir} does not exist")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Comparing uncertainty training results: {experiment_dir}")
    print(f"{'='*60}")

    df = load_experiment_data(experiment_dir, args.dataset)
    if df.empty:
        return
    print(f"  Loaded {len(df)} individual results")

    agg_df = aggregate_across_seeds(df)
    print(f"  Found {len(agg_df)} unique configurations")

    comp_dir = experiment_dir / "comparisons"
    comp_dir.mkdir(exist_ok=True)

    hist_df = load_uncertainty_histories(experiment_dir, args.dataset)
    if not hist_df.empty:
        print(f"  Loaded {len(hist_df)} uncertainty history rows")

    for dataset in sorted(df["DatasetName"].unique()):
        print(f"\n{'#'*40}")
        print(f"# {dataset}")
        print(f"{'#'*40}")
        make_summary_table(agg_df, comp_dir, dataset)
        plot_pareto_frontier(agg_df, comp_dir, dataset)
        plot_metric_bars(agg_df, comp_dir, dataset)
        plot_box_plots(df, comp_dir, dataset)
        plot_mode_comparison(agg_df, comp_dir, dataset)

        if not hist_df.empty:
            plot_uncertainty_evolution(hist_df, comp_dir, dataset)

    if agg_df["n_seeds"].max() >= 3:
        print(f"\n{'='*60}")
        print("SIGNIFICANCE TESTS (vs baseline unc_mode=none, Bonferroni)")
        print(f"{'='*60}")
        sig_df = run_significance_tests(agg_df, baseline_mode="none")
        if not sig_df.empty:
            sig_df.to_csv(comp_dir / "significance_tests.csv", index=False)
            print(f"  Saved: {comp_dir / 'significance_tests.csv'}")

            sig_only = sig_df[sig_df["Significant_005"]]
            if not sig_only.empty:
                print("\n  Statistically significant (Bonferroni p < 0.05):")
                for _, r in sig_only.iterrows():
                    star = "**" if r.get("Significant_001", False) else "*"
                    print(f"    {r['Dataset']} | {r['Metric']}: "
                          f"baseline={r['Baseline_mean']:.4f} vs "
                          f"{r['Compared']}={r['Compared_mean']:.4f} "
                          f"(p_adj={r['t_pval_adj']:.4f}, d={r['Cohen_d']:.2f}) {star}")
            else:
                print("  No statistically significant differences found")
    else:
        print("\n  Skipping significance tests (need >= 3 seeds)")

    agg_out = agg_df.drop(columns=[c for c in agg_df.columns if c.endswith("_values")], errors='ignore')
    agg_out.to_csv(comp_dir / "aggregated_results.csv", index=False)
    print(f"\n  Aggregated results: {comp_dir / 'aggregated_results.csv'}")
    print(f"\nDone! All outputs in: {comp_dir}")


if __name__ == "__main__":
    main()
