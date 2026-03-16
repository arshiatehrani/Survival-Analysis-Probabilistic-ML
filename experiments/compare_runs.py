"""Compare calibration-aware loss experiment runs.

Reads runs_index.csv (or individual run_metadata.json files), filters for
calibration loss experiments, and produces:
  1. Summary table (CSV + printed)
  2. Pareto frontier plot (CI vs D-Cal)
  3. Bar charts for key metrics

Usage:
    python experiments/compare_runs.py
    python experiments/compare_runs.py --dataset METABRIC
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths as pt

SCRIPT_FILTER = "exp_calibration_loss.py"


def load_experiment_results(results_dir, dataset_filter=None):
    """Scan run directories for experiment results CSVs and metadata."""
    rows = []
    results_path = Path(results_dir)
    for run_dir in sorted(results_path.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "experiment_results.csv"
        meta_path = run_dir / "run_metadata.json"
        if not csv_path.exists() or not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("script") != SCRIPT_FILTER:
            continue

        df = pd.read_csv(csv_path)
        df["run_id"] = meta.get("run_id", run_dir.name)
        df["run_dir"] = str(run_dir)

        if dataset_filter and df["DatasetName"].iloc[0] != dataset_filter:
            continue

        rows.append(df)

    if not rows:
        print("No experiment results found.")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def print_summary_table(df):
    """Print formatted summary table."""
    cols = ["DatasetName", "LossType", "Lambda", "Mu",
            "CI", "IBS", "DCalib", "CCalib", "ICI", "KM", "INBLL",
            "BestEpoch", "TrainTime"]
    available = [c for c in cols if c in df.columns]
    summary = df[available].copy()

    # Format floats
    for col in ["CI", "IBS", "DCalib", "CCalib", "ICI", "KM", "INBLL"]:
        if col in summary.columns:
            summary[col] = summary[col].map(lambda x: f"{x:.4f}")
    for col in ["TrainTime"]:
        if col in summary.columns:
            summary[col] = summary[col].map(lambda x: f"{x:.1f}s")

    print("\n" + "=" * 100)
    print("CALIBRATION LOSS EXPERIMENT COMPARISON")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100)

    # Highlight best results
    numeric_df = df[["LossType", "Lambda", "CI", "IBS", "DCalib", "ICI"]].copy()
    best_ci = numeric_df.loc[numeric_df["CI"].idxmax()]
    best_dcal = numeric_df.loc[numeric_df["DCalib"].idxmax()]
    best_ibs = numeric_df.loc[numeric_df["IBS"].idxmin()]
    best_ici = numeric_df.loc[numeric_df["ICI"].idxmin()]

    print(f"\n  Best CI:    {best_ci['LossType']} (λ={best_ci['Lambda']}) → {best_ci['CI']:.4f}")
    print(f"  Best D-Cal: {best_dcal['LossType']} (λ={best_dcal['Lambda']}) → {best_dcal['DCalib']:.4f}")
    print(f"  Best IBS:   {best_ibs['LossType']} (λ={best_ibs['Lambda']}) → {best_ibs['IBS']:.4f}")
    print(f"  Best ICI:   {best_ici['LossType']} (λ={best_ici['Lambda']}) → {best_ici['ICI']:.4f}")

    # Validation-based selection: best CI among D-Cal > 0.05
    calibrated = numeric_df[numeric_df["DCalib"] > 0.05]
    if len(calibrated) > 0:
        selected = calibrated.loc[calibrated["CI"].idxmax()]
        print(f"\n  ★ Recommended (best CI with D-Cal>0.05): {selected['LossType']} (λ={selected['Lambda']})")
        print(f"    CI={selected['CI']:.4f}, D-Cal={selected['DCalib']:.4f}, IBS={selected['IBS']:.4f}")
    else:
        print(f"\n  ⚠ No experiment achieved D-Cal p-value > 0.05")


def plot_pareto_frontier(df, save_dir):
    """Plot CI vs D-Cal Pareto frontier."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color by loss type family
    colors = {
        "cox": "#e74c3c", "ibs": "#3498db", "crps": "#2ecc71",
        "joint_ibs": "#9b59b6", "joint_crps": "#f39c12", "joint_crps_kl": "#1abc9c"
    }

    for _, row in df.iterrows():
        lt = row["LossType"]
        color = colors.get(lt, "#95a5a6")
        label = lt
        if "joint" in lt:
            label = f"{lt} (λ={row['Lambda']}"
            if row["Mu"] > 0:
                label += f", μ={row['Mu']}"
            label += ")"

        ax.scatter(row["CI"], row["DCalib"], c=color, s=120, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(label, (row["CI"], row["DCalib"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8,
                    color=color, fontweight="bold")

    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="D-Cal p=0.05 threshold")
    ax.set_xlabel("Concordance Index (CI) →", fontsize=12)
    ax.set_ylabel("D-Calibration p-value →", fontsize=12)
    ax.set_title(f"Pareto Frontier: Discrimination vs Calibration\n{df['DatasetName'].iloc[0]}", fontsize=14)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    save_path = save_dir / "pareto_frontier.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Pareto plot saved: {save_path}")
    return save_path


def plot_metric_bars(df, save_dir):
    """Bar chart comparison of key metrics across experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [("CI", "Concordance Index (↑)", True),
               ("IBS", "Integrated Brier Score (↓)", False),
               ("DCalib", "D-Calibration p-value (↑)", True),
               ("ICI", "ICI (↓)", False)]

    labels = []
    for _, row in df.iterrows():
        lt = row["LossType"]
        if "joint" in lt:
            lbl = f"{lt}\nλ={row['Lambda']}"
            if row.get("Mu", 0) > 0:
                lbl += f"\nμ={row['Mu']}"
        else:
            lbl = lt
        labels.append(lbl)

    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for idx, (metric, title, higher_better) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = df[metric].values
        bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Highlight best
        best_idx = values.argmax() if higher_better else values.argmin()
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    fig.suptitle(f"Metric Comparison: {df['DatasetName'].iloc[0]}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = save_dir / "metric_comparison.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Metrics plot saved: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare calibration loss experiments")
    parser.add_argument("--dataset", default=None, help="Filter by dataset name")
    parser.add_argument("--results-dir", default=None, help="Results directory (default: auto)")
    args = parser.parse_args()

    results_dir = args.results_dir or str(pt.RESULTS_DIR)
    df = load_experiment_results(results_dir, dataset_filter=args.dataset)

    if df.empty:
        print("No experiment results found. Run experiments first.")
        sys.exit(0)

    # Per-dataset analysis
    for dataset_name, group in df.groupby("DatasetName"):
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*60}")

        print_summary_table(group)

        save_dir = Path(results_dir) / "comparisons"
        save_dir.mkdir(parents=True, exist_ok=True)

        if len(group) >= 2:
            plot_pareto_frontier(group, save_dir)
            plot_metric_bars(group, save_dir)

        # Save summary CSV
        csv_path = save_dir / f"summary_{dataset_name.lower()}.csv"
        group.to_csv(csv_path, index=False)
        print(f"  Summary CSV saved: {csv_path}")
