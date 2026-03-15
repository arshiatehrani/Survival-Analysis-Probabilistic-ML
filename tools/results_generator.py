"""
Generic results generator for survival analysis models.

Produces comprehensive metrics, plots, and log files for ANY model
that generates survival curves -- including novel/custom models.

Usage (from training script or notebook):

    from tools.results_generator import ResultsGenerator, TeeLogger

    # Redirect stdout to a log file (optional)
    logger = TeeLogger.start("results/training_log.txt")

    # After training and computing surv_preds DataFrame:
    rg = ResultsGenerator("results")
    ext_metrics, calib_data = rg.generate_all(
        evaluator=lifelines_eval,   # LifelinesEvaluator instance
        event_times=event_times,
        t_test=t_test,
        e_test=e_test,
        dataset_name="SUPPORT",
        model_name="my_model",
        surv_preds_df=sanitized_surv_preds,   # for calibration collection
        event_times_pct=event_times_pct,       # for calibration collection
    )

    # After all models for a dataset, plot joint calibration curves:
    rg.plot_calibration_curves_all(all_calib_data, event_times_pct, model_names, dataset_name)

    # For BNN training loss curves:
    rg.plot_training_loss_curves(training_results_df, dataset_name, model_names)

    logger.close()
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from tools.evaluator import LifelinesEvaluator
from utility.survival import survival_probability_calibration, calculate_percentiles
from utility.plot import plot_calibration_curves, plot_training_curves


class TeeLogger:
    """Duplicate stdout to a log file so all print output is persisted.

    Works as a transparent wrapper: every ``print()`` call writes
    to both the real terminal and the log file.
    """

    def __init__(self, log_path):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        """Restore original stdout and close the log file."""
        sys.stdout = self.stdout
        self.log_file.close()

    @classmethod
    def start(cls, log_path):
        """Create a TeeLogger and install it as sys.stdout."""
        logger = cls(log_path)
        sys.stdout = logger
        return logger


class ResultsGenerator:
    """Model-agnostic evaluation & visualization for survival analysis.

    Every method accepts standard data structures (numpy arrays,
    DataFrames, evaluator instances) so it works identically for
    CoxPH, RSF, DSM, BayCox, MLP, SNGP, VI, MCD, and any future
    custom model.
    """

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_fig(self, fig, dataset_name, model_name, plot_type):
        """Save a figure to PDF and return the path."""
        path = self.results_dir / f"{dataset_name.lower()}_{model_name}_{plot_type}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Item 5 / 6 / 7 / 8: Extended metrics
    # ------------------------------------------------------------------

    def compute_extended_metrics(self, evaluator, event_times, event_times_for_pct=None):
        """Compute metrics beyond the paper's standard set.

        Returns a dict with:
          MSE_Hinge, RMSE_Hinge, X_Cal,
          AUC_25, AUC_50, AUC_75,
          One_Cal_25, One_Cal_50, One_Cal_75

        event_times_for_pct: optional array of times for percentile calculation.
          If None, uses event_times. Use np.concatenate([t_train, t_test]) for
          better AUC/One-Cal balance (avoids single-class at extreme percentiles).
        """
        metrics = {}

        for name, fn in [
            ("MSE_Hinge", lambda: float(evaluator.mse(method="Hinge"))),
            ("RMSE_Hinge", lambda: float(evaluator.rmse(method="Hinge"))),
            ("X_Cal", lambda: float(evaluator.x_calibration())),
        ]:
            try:
                metrics[name] = fn()
            except Exception:
                metrics[name] = np.nan

        times_for_pct = event_times_for_pct if event_times_for_pct is not None else event_times
        times_for_pct = np.asarray(times_for_pct).ravel()
        median_time = float(np.median(times_for_pct)) if len(times_for_pct) > 0 else np.nan

        event_times_pct = calculate_percentiles(times_for_pct)
        for q, t0 in event_times_pct.items():
            # [DEBUG] Check for NaN in predict_probs before AUC/1-Cal
            try:
                predict_probs = evaluator.predict_probability_from_curve(t0)
                nan_count = int(np.isnan(predict_probs).sum())
                if nan_count > 0:
                    print(f"     [DEBUG] @{q} t0={t0}: {nan_count}/{len(predict_probs)} NaN in predict_probs")
            except Exception as ex:
                print(f"     [DEBUG] @{q} t0={t0}: predict_probability_from_curve failed: {ex}")

            for prefix, fn, fallback_fn in [
                (f"AUC_{q}", lambda _t=t0: float(evaluator.auc(_t)),
                 lambda: float(evaluator.auc(median_time))),
                (f"One_Cal_{q}", lambda _t=t0: float(evaluator.one_calibration(_t)[0]),
                 lambda: float(evaluator.one_calibration(median_time)[0])),
            ]:
                try:
                    val = fn()
                    if np.isnan(val) and not np.isnan(median_time):
                        try:
                            val = fallback_fn()
                        except Exception:
                            pass
                    metrics[prefix] = val
                except Exception:
                    metrics[prefix] = np.nan

        return metrics

    # ------------------------------------------------------------------
    # Item 3: Individual survival curves
    # ------------------------------------------------------------------

    def plot_survival_curves(self, evaluator, event_times,
                              dataset_name, model_name,
                              sample_indices=None, n_samples=5):
        """Plot predicted S(t) for selected test patients."""
        try:
            n_total = evaluator.predicted_curves.shape[0]
            if sample_indices is None:
                step = max(1, n_total // n_samples)
                sample_indices = list(range(0, n_total, step))[:n_samples]

            fig, ax = plt.subplots(figsize=(10, 6))
            time_coords = evaluator.time_coordinates
            for idx in sample_indices:
                if idx >= n_total:
                    continue
                curve = evaluator.predicted_curves[idx, :]
                t_actual = evaluator.event_times[idx]
                ev = evaluator.event_indicators[idx]
                status = "event" if ev else "censor"
                ax.plot(time_coords, curve, linewidth=1.5, alpha=0.8,
                        label=f"#{idx} (t={t_actual:.0f}, {status})")

            ax.set_xlabel("Time")
            ax.set_ylabel("Survival Probability S(t)")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{model_name.upper()} - Survival Curves ({dataset_name})")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

            path = self._save_fig(fig, dataset_name, model_name, "survival_curves")
            print(f"     Survival curves saved: {path}")
            return path
        except Exception as e:
            print(f"     Survival curves skipped: {e}")
            return None

    # ------------------------------------------------------------------
    # Item 4: Time-dependent Brier score curve
    # ------------------------------------------------------------------

    def plot_brier_score_curve(self, evaluator, event_times,
                                dataset_name, model_name, num_points=50):
        """Plot BS(t) over time with shaded IBS area."""
        try:
            t_min = event_times[0]
            t_max = np.amax(np.concatenate(
                (evaluator.event_times, evaluator.train_event_times))) * 0.9
            time_points = np.linspace(t_min, t_max, num_points)

            bs_scores = evaluator.brier_score_multiple_points(time_points)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_points, bs_scores, "b-", linewidth=2)
            ax.fill_between(time_points, 0, bs_scores, alpha=0.15, color="blue")
            ibs = evaluator.integrated_brier_score()
            ax.set_xlabel("Time")
            ax.set_ylabel("Brier Score BS(t)")
            ax.set_title(f"{model_name.upper()} - Brier Score Curve ({dataset_name})"
                         f"\nIBS = {ibs:.4f}")
            ax.grid(True, alpha=0.3)

            path = self._save_fig(fig, dataset_name, model_name, "brier_curve")
            print(f"     Brier score curve saved: {path}")
            return path
        except Exception as e:
            print(f"     Brier score curve skipped: {e}")
            return None

    # ------------------------------------------------------------------
    # Item 9: Predicted vs actual scatter
    # ------------------------------------------------------------------

    def plot_predicted_vs_actual(self, evaluator, t_test, e_test,
                                  dataset_name, model_name):
        """Scatter plot of predicted median survival time vs actual time."""
        try:
            pred_times = evaluator.predicted_event_times

            fig, ax = plt.subplots(figsize=(8, 8))
            event_mask = e_test.astype(bool)
            censor_mask = ~event_mask

            if np.any(event_mask):
                ax.scatter(t_test[event_mask], pred_times[event_mask],
                           c="#F15854", alpha=0.5, s=20,
                           label=f"Events (n={event_mask.sum()})")
            if np.any(censor_mask):
                ax.scatter(t_test[censor_mask], pred_times[censor_mask],
                           c="#5DA5DA", alpha=0.5, s=20,
                           label=f"Censored (n={censor_mask.sum()})")

            max_val = max(np.max(t_test), np.max(pred_times)) * 1.05
            ax.plot([0, max_val], [0, max_val], "k--", linewidth=1,
                    alpha=0.5, label="Perfect prediction")
            ax.set_xlabel("Actual Time")
            ax.set_ylabel("Predicted Median Survival Time")
            ax.set_title(f"{model_name.upper()} - Predicted vs Actual ({dataset_name})")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            path = self._save_fig(fig, dataset_name, model_name, "pred_vs_actual")
            print(f"     Predicted vs actual saved: {path}")
            return path
        except Exception as e:
            print(f"     Predicted vs actual skipped: {e}")
            return None

    # ------------------------------------------------------------------
    # Item 10: Predicted survival time histogram
    # ------------------------------------------------------------------

    def plot_survival_time_histogram(self, evaluator, t_test, e_test,
                                      dataset_name, model_name):
        """Histogram of predicted and actual survival times."""
        try:
            pred_times = evaluator.predicted_event_times

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(pred_times, bins=30, color="steelblue",
                         edgecolor="white", alpha=0.8)
            axes[0].axvline(x=np.mean(pred_times), color="red", linestyle="--",
                            linewidth=2, label=f"Mean: {np.mean(pred_times):.0f}")
            axes[0].axvline(x=np.median(pred_times), color="orange", linestyle="--",
                            linewidth=2, label=f"Median: {np.median(pred_times):.0f}")
            axes[0].set_xlabel("Predicted Median Survival Time")
            axes[0].set_ylabel("Count")
            axes[0].set_title(f"{model_name.upper()} - Predicted Time Distribution")
            axes[0].legend(fontsize=9)

            ev_mask = e_test.astype(bool)
            axes[1].hist(t_test[ev_mask], bins=30, color="#F15854",
                         edgecolor="white", alpha=0.7,
                         label=f"Events (n={ev_mask.sum()})")
            axes[1].hist(t_test[~ev_mask], bins=30, color="#5DA5DA",
                         edgecolor="white", alpha=0.7,
                         label=f"Censored (n={(~ev_mask).sum()})")
            axes[1].set_xlabel("Actual Time")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"Actual Time Distribution ({dataset_name})")
            axes[1].legend(fontsize=9)

            fig.suptitle(dataset_name, fontsize=14)
            plt.tight_layout()

            path = self._save_fig(fig, dataset_name, model_name, "time_histogram")
            print(f"     Time histogram saved: {path}")
            return path
        except Exception as e:
            print(f"     Time histogram skipped: {e}")
            return None

    # ------------------------------------------------------------------
    # Item 2: Calibration data collection + joint plot
    # ------------------------------------------------------------------

    def collect_calibration_data(self, surv_preds_df, t_test, e_test,
                                  event_times_pct):
        """Collect calibration data for one model.

        Returns ``{t0: (pred, obs, predictions_at_t0)}`` which can be
        accumulated across models and passed to
        ``plot_calibration_curves_all``.
        """
        calib_data = {}
        for t0 in event_times_pct.values():
            try:
                pred_t0, obs_t0, predictions_at_t0, _ = \
                    survival_probability_calibration(surv_preds_df, t_test, e_test, t0)
                calib_data[t0] = (pred_t0, obs_t0, predictions_at_t0)
            except Exception:
                pass
        return calib_data

    def plot_calibration_curves_all(self, all_calib_data, percentiles,
                                     model_names, dataset_name):
        """Plot calibration curves with all models on the same chart.

        Parameters
        ----------
        all_calib_data : dict
            ``{model_name: {t0: (pred, obs, predictions_at_t0)}}``
        percentiles : dict
            ``{quantile: time_value}``  (from ``calculate_percentiles``)
        model_names : list[str]
        dataset_name : str
        """
        try:
            pred_obs = defaultdict(dict)
            predictions = defaultdict(dict)
            active_models = []
            for mn in model_names:
                if mn in all_calib_data and all_calib_data[mn]:
                    active_models.append(mn)
                    for t0, (pred, obs, preds) in all_calib_data[mn].items():
                        pred_obs[t0][mn] = (pred, obs)
                        predictions[t0][mn] = preds

            if pred_obs and active_models:
                plot_calibration_curves(percentiles, pred_obs, predictions,
                                        active_models, dataset_name)
                print(f"  Calibration curves saved for {dataset_name}")
        except Exception as e:
            print(f"  Calibration curves skipped: {e}")

    # ------------------------------------------------------------------
    # Item 1: Training loss curves (BNN)
    # ------------------------------------------------------------------

    def plot_training_loss_curves(self, training_results_df, dataset_name,
                                   model_names):
        """Plot per-epoch train/valid loss and variance for BNN models."""
        try:
            metric_names = ["TrainLoss", "TrainVariance",
                            "ValidLoss", "ValidVariance"]
            plot_training_curves(training_results_df, dataset_name,
                                 model_names, metric_names)
            print(f"  Training loss curves saved for {dataset_name}")
        except Exception as e:
            print(f"  Training loss curves skipped: {e}")

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def generate_all(self, evaluator, event_times, t_test, e_test,
                     dataset_name, model_name,
                     surv_preds_df=None, event_times_pct=None,
                     event_times_for_pct=None):
        """Compute extended metrics and generate all plots for one model.

        Parameters
        ----------
        evaluator : LifelinesEvaluator
            Already constructed with sanitised predictions and train data.
        event_times : np.ndarray
            Time grid used for survival curve evaluation.
        t_test, e_test : np.ndarray
            Sanitised test times / event indicators (matching evaluator).
        dataset_name, model_name : str
        surv_preds_df : pd.DataFrame, optional
            Sanitised survival predictions (for calibration data collection).
        event_times_pct : dict, optional
            ``{quantile: time}`` (for calibration data collection).

        Returns
        -------
        ext_metrics : dict
            Extended metric values.
        calib_data : dict
            Calibration data for this model (pass to
            ``plot_calibration_curves_all`` after all models finish).
        """
        ext_metrics = self.compute_extended_metrics(
            evaluator, event_times, event_times_for_pct=event_times_for_pct)

        # Print extended metrics
        mse_h = ext_metrics.get("MSE_Hinge", np.nan)
        rmse_h = ext_metrics.get("RMSE_Hinge", np.nan)
        x_cal = ext_metrics.get("X_Cal", np.nan)
        mse_str = f"{mse_h:.2f}" if not np.isnan(mse_h) else "N/A"
        rmse_str = f"{rmse_h:.2f}" if not np.isnan(rmse_h) else "N/A"
        xcal_str = f"{x_cal:.4f}" if not np.isnan(x_cal) else "N/A"
        print(f"     MSE_H: {mse_str} | RMSE_H: {rmse_str} | X-Cal: {xcal_str}")

        auc_parts, one_cal_parts = [], []
        for key, val in ext_metrics.items():
            if key.startswith("AUC_"):
                q = key.split("_")[1]
                s = f"AUC@{q}: {val:.4f}" if not np.isnan(val) else f"AUC@{q}: N/A"
                auc_parts.append(s)
            elif key.startswith("One_Cal_"):
                q = key.split("_")[2]
                s = f"1-Cal@{q}: {val:.3f}" if not np.isnan(val) else f"1-Cal@{q}: N/A"
                one_cal_parts.append(s)
        if auc_parts:
            print(f"     {' | '.join(auc_parts)}")
        if one_cal_parts:
            print(f"     {' | '.join(one_cal_parts)}")

        # Generate plots
        self.plot_survival_curves(evaluator, event_times,
                                   dataset_name, model_name)
        self.plot_brier_score_curve(evaluator, event_times,
                                     dataset_name, model_name)
        self.plot_predicted_vs_actual(evaluator, t_test, e_test,
                                       dataset_name, model_name)
        self.plot_survival_time_histogram(evaluator, t_test, e_test,
                                           dataset_name, model_name)

        # Collect calibration data for joint plot later
        calib_data = {}
        if surv_preds_df is not None and event_times_pct is not None:
            calib_data = self.collect_calibration_data(
                surv_preds_df,
                t_test, e_test,
                event_times_pct)

        return ext_metrics, calib_data
