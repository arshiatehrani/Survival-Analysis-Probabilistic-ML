"""Post-Hoc Calibration Experiment Script.

Loads trained models from a calibration-loss experiment, applies
Temperature Scaling and Isotonic Regression on validation set,
and evaluates before/after calibration on test set.

Usage:
    python experiments/exp_posthoc_calibration.py \
        --source-experiment 20260316_calibration_loss \
        --dataset METABRIC --loss-config cox --seed 0
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import atexit
import copy
import gc
import json
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import chisquare
from pathlib import Path
from time import time as timer
from datetime import datetime
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*add_variable.*deprecated.*')
warnings.filterwarnings('ignore', message='.*RandomNormal is unseeded.*')

# Monkey-patch: scipy >= 1.14 removed simps, but pycox still uses it
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import paths as pt
from utility.config import load_config
from utility.training import get_data_loader, make_stratified_split, scale_data, split_time_event
from utility.risk import InputFunction
from utility.survival import (calculate_event_times, calculate_percentiles, convert_to_structured,
                              compute_deterministic_survival_curve, compute_nondeterministic_survival_curve,
                              coverage, survival_probability_calibration)
from utility.posthoc_calibration import TemperatureScaling, IsotonicCalibration
from tools.baysurv_trainer import Trainer
from tools.baysurv_builder import make_mcd_model, make_mlp_model, make_sngp_model, make_vi_model
from tools.evaluator import LifelinesEvaluator
from pycox.evaluation import EvalSurv

ALL_DATASETS = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
ALL_MODELS = ["mlp", "sngp", "mcd1", "mcd2", "mcd3", "vi"]


def count_parameters(model):
    try:
        return model.count_params()
    except Exception:
        return None


def evaluate_survival_preds(surv_preds, y_test, t_test, e_test,
                            t_train, e_train, event_times, event_times_pct,
                            label=""):
    """Evaluate survival predictions and return a dict of metrics."""
    surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0)
    bad_idx = surv_preds[surv_preds.iloc[:, 0] < 0.5].index
    sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
    sanitized_y_test = np.delete(y_test, bad_idx, axis=0)
    sanitized_t_test = np.delete(t_test, bad_idx, axis=0)
    sanitized_e_test = np.delete(e_test, bad_idx, axis=0)

    lifelines_eval = LifelinesEvaluator(
        sanitized_surv_preds.T, sanitized_y_test["time"],
        sanitized_y_test["event"], t_train, e_train
    )
    ibs = lifelines_eval.integrated_brier_score()
    mae_hinge = lifelines_eval.mae(method="Hinge")
    mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
    d_calib = lifelines_eval.d_calibration()[0]
    km_mse = lifelines_eval.km_calibration()

    ev = EvalSurv(sanitized_surv_preds.T, sanitized_y_test["time"],
                  sanitized_y_test["event"], censor_surv="km")
    inbll = ev.integrated_nbll(event_times)
    ci = ev.concordance_td()

    # ICI
    try:
        deltas = {}
        for t0 in event_times_pct.values():
            _, _, _, deltas_t0 = survival_probability_calibration(
                sanitized_surv_preds, sanitized_y_test["time"],
                sanitized_y_test["event"], t0
            )
            deltas[t0] = deltas_t0
        ici = list(deltas.values())[-1].mean()
    except Exception as e:
        ici = np.nan

    metrics = {
        "CI": ci, "IBS": ibs, "MAEHinge": mae_hinge, "MAEPseudo": mae_pseudo,
        "DCalib": d_calib, "ICI": ici, "KM": km_mse, "INBLL": inbll,
    }

    if label:
        dcal_str = f"{d_calib:.4f}" if d_calib > 0.05 else f"{d_calib:.4f}*"
        print(f"  [{label}] CI={ci:.4f} IBS={ibs:.4f} D-Cal={dcal_str} "
              f"ICI={ici:.4f} INBLL={inbll:.4f} MAE_H={mae_hinge:.2f}")

    return metrics


# ----------- GPU setup -----------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {gpus}")
else:
    print("No GPU found, using CPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-hoc calibration experiment (no retraining)"
    )
    parser.add_argument("--source-experiment", required=True,
                        help="Name of the calibration-loss experiment to read checkpoints from")
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--model", default="mcd1", choices=ALL_MODELS)
    parser.add_argument("--loss-config", required=True,
                        help="Loss config folder name, e.g., 'cox', 'crps', 'joint_crps_lam0.5'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples-test", type=int, default=100)
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Output experiment name (default: auto)")
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    loss_config = args.loss_config
    seed = args.seed

    # ---- Set seeds ----
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # ---- Locate source checkpoint ----
    source_dir = Path(pt.RESULTS_DIR) / args.source_experiment / dataset_name / loss_config / f"seed_{seed}"
    models_dir = source_dir / "models"
    config_path = source_dir / "config.json"

    if not config_path.exists():
        print(f"ERROR: Source config not found: {config_path}")
        print(f"  Make sure the calibration-loss experiment '{args.source_experiment}' has been run for")
        print(f"  dataset={dataset_name}, loss={loss_config}, seed={seed}")
        sys.exit(1)

    with open(config_path) as f:
        run_config = json.load(f)

    best_ep = run_config.get("best_epoch", 1)
    print(f"Loading checkpoint from {source_dir} (epoch {best_ep})")

    # ---- Output directory ----
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = datetime.now().strftime("%Y%m%d") + "_posthoc_calibration"

    out_dir = Path(pt.RESULTS_DIR) / experiment_name / dataset_name / loss_config / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Post-Hoc Calibration Experiment")
    print(f"  Source: {args.source_experiment}/{dataset_name}/{loss_config}/seed_{seed}")
    print(f"  Dataset: {dataset_name} | Model: {model_name} | Seed: {seed}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # ---- Load Config ----
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    optimizer_config = copy.deepcopy(config['optimizer'])
    if isinstance(optimizer_config, dict) and isinstance(optimizer_config.get("config"), dict):
        optimizer_config["config"].pop("decay", None)
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    l2_reg = config['l2_reg']
    batch_size = config['batch_size']
    n_samples_train = config['n_samples_train']
    n_samples_valid = config['n_samples_valid']
    n_samples_test = args.n_samples_test

    # ---- Load & Split Data (same seed = same split) ----
    dl = get_data_loader(dataset_name).load_data()
    num_features, cat_features = dl.get_features()
    df = dl.get_data()
    print(f"  Samples: {len(df)} | Features: {len(num_features)} num + {len(cat_features)} cat")

    df_train, df_valid, df_test = make_stratified_split(
        df, stratify_colname='both', frac_train=0.7,
        frac_valid=0.1, frac_test=0.2, random_state=seed
    )
    X_train = df_train[cat_features + num_features]
    X_valid = df_valid[cat_features + num_features]
    X_test = df_test[cat_features + num_features]

    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])

    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)

    t_train, e_train = split_time_event(y_train)
    t_valid, e_valid = split_time_event(y_valid)
    t_test, e_test = split_time_event(y_test)

    event_times = calculate_event_times(t_train, e_train)
    event_times_pct = calculate_percentiles(event_times)

    # ---- Rebuild model and restore checkpoint ----
    tf.keras.backend.clear_session()
    gc.collect()

    if model_name == "mlp":
        dropout_rate = config['dropout_rate']
        model = make_mlp_model(input_shape=X_train.shape[1:], output_dim=1,
                               layers=layers, activation_fn=activation_fn,
                               dropout_rate=dropout_rate, regularization_pen=l2_reg)
    elif model_name == "sngp":
        dropout_rate = config['dropout_rate']
        model = make_sngp_model(input_shape=X_train.shape[1:], output_dim=1,
                                layers=layers, activation_fn=activation_fn,
                                dropout_rate=dropout_rate, regularization_pen=l2_reg)
    elif model_name == "vi":
        dropout_rate = config['dropout_rate']
        model = make_vi_model(n_train_samples=len(X_train),
                              input_shape=X_train.shape[1:], output_dim=2,
                              layers=layers, activation_fn=activation_fn,
                              dropout_rate=dropout_rate, regularization_pen=l2_reg)
    elif model_name in ["mcd1", "mcd2", "mcd3"]:
        dropout_rates = {"mcd1": 0.1, "mcd2": 0.2, "mcd3": 0.5}
        dropout_rate = dropout_rates[model_name]
        model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                               layers=layers, activation_fn=activation_fn,
                               dropout_rate=dropout_rate, regularization_pen=l2_reg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Build model by running a dummy forward pass
    _ = model(tf.zeros([1] + list(X_train.shape[1:])))

    # Restore weights
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(str(models_dir / f"ckpt-{best_ep}"))
    print(f"  Checkpoint restored: epoch {best_ep}")

    # ---- Generate survival predictions ----
    print("\n  Generating survival predictions...")

    if model_name in ["mlp", "sngp"]:
        surv_preds_test = compute_deterministic_survival_curve(
            model, X_train, X_test, e_train, t_train, event_times, model_name)
        surv_preds_valid = compute_deterministic_survival_curve(
            model, X_train, X_valid, e_train, t_train, event_times, model_name)
    else:
        surv_preds_test = np.mean(compute_nondeterministic_survival_curve(
            model, np.array(X_train), np.array(X_test),
            e_train, t_train, event_times,
            n_samples_train, n_samples_test), axis=0)
        surv_preds_valid = np.mean(compute_nondeterministic_survival_curve(
            model, np.array(X_train), np.array(X_valid),
            e_train, t_train, event_times,
            n_samples_train, n_samples_test), axis=0)

    surv_preds_test = pd.DataFrame(surv_preds_test, columns=event_times)
    surv_preds_valid = pd.DataFrame(surv_preds_valid, columns=event_times)

    # ---- Evaluate BEFORE calibration ----
    print("\n  Evaluating (no calibration)...")
    metrics_none = evaluate_survival_preds(
        surv_preds_test, y_test, t_test, e_test,
        t_train, e_train, event_times, event_times_pct,
        label="None"
    )

    # ---- Fit Temperature Scaling on validation set ----
    print("\n  Fitting Temperature Scaling on validation set...")
    temp_scaler = TemperatureScaling(T_range=(0.1, 5.0), n_grid=200)
    temp_scaler.fit(surv_preds_valid, t_valid, e_valid, t_train, e_train)
    print(f"  Optimal T = {temp_scaler.T_:.3f} (val IBS={temp_scaler.best_ibs_:.4f})")

    surv_preds_temp = temp_scaler.transform(surv_preds_test)
    metrics_temp = evaluate_survival_preds(
        surv_preds_temp, y_test, t_test, e_test,
        t_train, e_train, event_times, event_times_pct,
        label="TempScal"
    )

    # ---- Fit Isotonic Regression on validation set ----
    print("\n  Fitting Isotonic Regression on validation set...")
    iso_calibrator = IsotonicCalibration(n_time_points=50)
    iso_calibrator.fit(surv_preds_valid, t_valid, e_valid)
    print(f"  Fitted {len(iso_calibrator.models_)} isotonic models at different time points")

    surv_preds_iso = iso_calibrator.transform(surv_preds_test)
    metrics_iso = evaluate_survival_preds(
        surv_preds_iso, y_test, t_test, e_test,
        t_train, e_train, event_times, event_times_pct,
        label="Isotonic"
    )

    # ---- Print Summary ----
    print(f"\n{'='*60}")
    print(f"POST-HOC CALIBRATION RESULTS: {dataset_name} | {loss_config} | seed={seed}")
    print(f"{'='*60}")
    header = f"{'Calibration':<15} {'CI':>8} {'IBS':>8} {'D-Cal':>8} {'ICI':>8} {'INBLL':>8}"
    print(header)
    print("-" * len(header))
    for name, m in [("None", metrics_none), ("TempScal", metrics_temp), ("Isotonic", metrics_iso)]:
        dcal_str = f"{m['DCalib']:.4f}" if m['DCalib'] > 0.05 else f"{m['DCalib']:.4f}*"
        print(f"{name:<15} {m['CI']:8.4f} {m['IBS']:8.4f} {dcal_str:>8} "
              f"{m['ICI']:8.4f} {m['INBLL']:8.4f}")
    print(f"{'='*60}")
    print(f"  Temperature T = {temp_scaler.T_:.3f}")

    # ---- Save Results ----
    rows = []
    for calib_name, metrics in [("none", metrics_none), ("temp_scaling", metrics_temp),
                                 ("isotonic", metrics_iso)]:
        row = {
            "DatasetName": dataset_name,
            "ModelName": model_name,
            "LossConfig": loss_config,
            "SourceExperiment": args.source_experiment,
            "Seed": seed,
            "Calibration": calib_name,
            "Temperature_T": temp_scaler.T_ if calib_name == "temp_scaling" else np.nan,
            "N_Isotonic_Models": len(iso_calibrator.models_) if calib_name == "isotonic" else np.nan,
            **metrics,
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "posthoc_metrics.csv", index=False)
    print(f"\n  Results: {out_dir / 'posthoc_metrics.csv'}")

    # ---- Cleanup ----
    del ckpt  # delete checkpoint object to avoid __del__ error
    del model
    tf.keras.backend.clear_session()
    gc.collect()
