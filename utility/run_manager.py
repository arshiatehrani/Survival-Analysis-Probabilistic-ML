"""
run_manager.py
====================================
Experiment tracking: creates timestamped run directories with metadata,
metrics, and model artifacts for every training invocation.

Usage:
    from utility.run_manager import RunManager

    run = RunManager(
        base_results_dir=pt.RESULTS_DIR,
        script_name="train_bnn_models.py",
        datasets=["METABRIC"],
        models=["mcd1"],
        cli_args=vars(args),
    )
    # run.run_dir   → Path to this run's output directory
    # run.models_dir → Path to this run's models subdirectory

    # After training each model:
    run.log_model_result("METABRIC", "mcd1",
        config={...}, metrics={...},
        extra={"best_epoch": 42, "early_stopped": True, ...})

    # At the very end:
    run.finalize()
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _git_info(repo_dir):
    """Return (short_hash, is_dirty) or (None, None) if not a git repo."""
    try:
        short_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL
        ) != 0
        return short_hash, dirty
    except Exception:
        return None, None


class RunManager:
    """Lightweight experiment tracker that creates per-run directories.

    Directory layout created::

        results/
        ├── runs_index.csv
        └── 20260316_182507_METABRIC_mcd1/
            ├── run_metadata.json
            ├── models/          ← weights + TF checkpoints
            └── ...              ← CSVs, plots, logs
    """

    def __init__(self, base_results_dir, script_name, datasets, models,
                 cli_args=None):
        self._base = Path(base_results_dir)
        self._base.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        self._timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
        ts_short = now.strftime("%Y%m%d_%H%M%S")

        # Build a readable but bounded run_id
        ds_tag = "_".join(d[:4] for d in datasets)  # META, SEER, SUPP, MIMI
        md_tag = "_".join(models[:4])  # cap at 4 model names
        self.run_id = f"{ts_short}_{ds_tag}_{md_tag}"
        # Truncate to keep filesystem paths reasonable
        if len(self.run_id) > 80:
            self.run_id = self.run_id[:80]

        self.run_dir = self._base / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.run_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Capture git info
        repo_dir = Path(__file__).resolve().parent.parent
        git_hash, git_dirty = _git_info(repo_dir)

        self._metadata = {
            "run_id": self.run_id,
            "timestamp": self._timestamp,
            "script": script_name,
            "datasets": list(datasets),
            "models": list(models),
            "cli_args": dict(cli_args) if cli_args else {},
            "python_version": sys.version,
            "git_hash": git_hash,
            "git_dirty": git_dirty,
            "per_model": {},
        }
        # Write initial metadata so the file exists even if the run crashes
        self._write_metadata()

        print(f"[RunManager] Run directory: {self.run_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_model_result(self, dataset, model, config=None, metrics=None,
                         extra=None):
        """Record results for one (dataset, model) combination.

        Parameters
        ----------
        dataset, model : str
        config : dict, optional
            Hyperparameters / config used.
        metrics : dict, optional
            Evaluation metrics (CI, IBS, DCalib, ...).
        extra : dict, optional
            Anything else: best_epoch, early_stopped, n_params, train_time, ...
        """
        key = f"{dataset}_{model}"
        entry = {}
        if config is not None:
            entry["config"] = config
        if metrics is not None:
            entry["metrics"] = metrics
        if extra is not None:
            entry.update(extra)
        self._metadata["per_model"][key] = entry
        self._write_metadata()

    def finalize(self):
        """Write final metadata and append a summary row to runs_index.csv."""
        self._write_metadata()
        self._append_index()
        print(f"[RunManager] Run finalized: {self.run_dir}")
        self._print_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_metadata(self):
        path = self.run_dir / "run_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, cls=_NumpyEncoder)

    def _append_index(self):
        """Append one summary row to the global runs_index.csv."""
        index_path = self._base / "runs_index.csv"

        row = {
            "run_id": self.run_id,
            "timestamp": self._timestamp,
            "script": self._metadata["script"],
            "datasets": ",".join(self._metadata["datasets"]),
            "models": ",".join(self._metadata["models"]),
            "git_hash": self._metadata.get("git_hash", ""),
        }

        # Flatten per-model metrics into columns like METABRIC_mcd1_CI
        for key, entry in self._metadata["per_model"].items():
            metrics = entry.get("metrics", {})
            for metric_name, val in metrics.items():
                row[f"{key}_{metric_name}"] = val
            for field in ("best_epoch", "early_stopped", "n_params",
                          "train_time_s", "test_time_s"):
                if field in entry:
                    row[f"{key}_{field}"] = entry[field]

        new_row = pd.DataFrame([row])
        if index_path.exists():
            existing = pd.read_csv(index_path)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row
        combined.to_csv(index_path, index=False)

    def _print_summary(self):
        """Print a summary of the run for Slurm logs."""
        print(f"\n{'='*60}")
        print(f"[RunManager] RUN SUMMARY")
        print(f"{'='*60}")
        print(f"  Run ID:    {self.run_id}")
        print(f"  Directory: {self.run_dir}")
        print(f"  Git:       {self._metadata.get('git_hash', 'N/A')}"
              f"{' (dirty)' if self._metadata.get('git_dirty') else ''}")
        print(f"  Script:    {self._metadata['script']}")
        print(f"  Datasets:  {self._metadata['datasets']}")
        print(f"  Models:    {self._metadata['models']}")

        for key, entry in self._metadata["per_model"].items():
            metrics = entry.get("metrics", {})
            ci = metrics.get("CI", "N/A")
            ibs = metrics.get("IBS", "N/A")
            dcal = metrics.get("DCalib", "N/A")
            best_ep = entry.get("best_epoch", "N/A")
            es = entry.get("early_stopped", "N/A")
            ci_s = f"{ci:.4f}" if isinstance(ci, float) else ci
            ibs_s = f"{ibs:.4f}" if isinstance(ibs, float) else ibs
            dcal_s = f"{dcal:.4f}" if isinstance(dcal, float) else dcal
            print(f"  {key}: CI={ci_s} IBS={ibs_s} D-Cal={dcal_s}"
                  f" best_ep={best_ep} early_stop={es}")

        # List saved files
        files = sorted(self.run_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        print(f"\n  Saved files ({len(files)}):")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"    {f.relative_to(self.run_dir)}  ({size_kb:.1f} KB)")
        print(f"{'='*60}\n")
