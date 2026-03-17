"""Calibration-Aware Loss Experiment Script.

Trains one BNN model on one dataset with a configurable loss function and seed.
Each invocation creates one leaf directory in the hierarchical results structure:
  results/{experiment_name}/{dataset}/{loss_config}/seed_{seed}/

Usage examples:
    # Baseline (original Cox PH), seed 0
    python experiments/exp_calibration_loss.py --dataset METABRIC --loss-type cox --seed 0

    # Joint Cox + CRPS with lambda=0.3, seed 3
    python experiments/exp_calibration_loss.py --dataset METABRIC --loss-type joint_crps --lambda-val 0.3 --seed 3

    # Custom experiment name
    python experiments/exp_calibration_loss.py --dataset METABRIC --loss-type crps --experiment-name calibration_loss_v2
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import csv
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
                              coverage, survival_probability_calibration, KaplanMeier)
from utility.loss import (CoxPHLoss, CRPSLoss, BrierScoreLoss,
                          JointCoxCalibrationLoss, MarginalCalibrationLoss)
from tools.baysurv_trainer import Trainer
from tools.baysurv_builder import make_mcd_model, make_mlp_model, make_sngp_model, make_vi_model
from tools.evaluator import LifelinesEvaluator
from tools.results_generator import TeeLogger
from pycox.evaluation import EvalSurv

ALL_DATASETS = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
ALL_MODELS = ["mlp", "sngp", "mcd1", "mcd2", "mcd3", "vi"]
N_GRID_POINTS = 100


def count_parameters(model):
    try:
        return model.count_params()
    except Exception:
        return None


def make_loss_config_name(loss_type, lam, mu):
    """Create a folder-safe name for the loss configuration."""
    name = loss_type
    if "joint" in loss_type:
        name += f"_lam{lam}"
    if mu > 0:
        name += f"_mu{mu}"
    return name


def compute_censoring_km(t_train, e_train):
    censoring_indicator = 1 - e_train.astype(int)
    km = KaplanMeier(t_train, censoring_indicator)
    return km.survival_times, km.survival_probabilities


def build_time_grid(t_train, n_points=N_GRID_POINTS):
    return np.linspace(0, t_train.max(), n_points).astype(np.float32)


def build_loss_function(loss_type, lam, mu, t_train, e_train):
    time_grid = build_time_grid(t_train)
    if loss_type == "cox":
        return CoxPHLoss(), "CoxPH (baseline)"
    elif loss_type == "ibs":
        km_times, km_probs = compute_censoring_km(t_train, e_train)
        return BrierScoreLoss(time_grid, km_times, km_probs), "Integrated Brier Score"
    elif loss_type == "crps":
        return CRPSLoss(time_grid), "Right-censored CRPS"
    elif loss_type == "joint_ibs":
        km_times, km_probs = compute_censoring_km(t_train, e_train)
        cal = BrierScoreLoss(time_grid, km_times, km_probs)
        return JointCoxCalibrationLoss(cal, lam=lam), f"Joint Cox+IBS (λ={lam})"
    elif loss_type == "joint_crps":
        cal = CRPSLoss(time_grid)
        return JointCoxCalibrationLoss(cal, lam=lam), f"Joint Cox+CRPS (λ={lam})"
    elif loss_type == "joint_crps_kl":
        cal = CRPSLoss(time_grid)
        km_event = KaplanMeier(t_train, e_train.astype(int))
        km_surv_at_grid = km_event.predict(time_grid)
        marg = MarginalCalibrationLoss(time_grid, km_surv_at_grid)
        return (JointCoxCalibrationLoss(cal, lam=lam, marginal_loss=marg, mu=mu),
                f"Joint Cox+CRPS+KL (λ={lam}, μ={mu})")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def append_to_runs_index(index_path, row_dict):
    """Append one row to the master runs_index.csv (thread-safe via append mode)."""
    file_exists = index_path.exists()
    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {gpus}")
else:
    print("No GPU found, using CPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibration-aware loss experiment (single model × single dataset × single seed)"
    )
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--model", default="mcd1", choices=ALL_MODELS)
    parser.add_argument("--loss-type", required=True,
                        choices=["cox", "ibs", "crps", "joint_ibs", "joint_crps", "joint_crps_kl"])
    parser.add_argument("--lambda-val", type=float, default=0.3)
    parser.add_argument("--mu-val", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for data split and model init (default: 0)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--n-samples-test", type=int, default=100)
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment session name (default: auto-generated with date)")
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    loss_type = args.loss_type
    lam = args.lambda_val
    mu = args.mu_val
    seed = args.seed
    N_EPOCHS = args.epochs
    DISABLE_EARLY_STOP = args.no_early_stop

    # ---- Set all random seeds ----
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # ---- Build directory structure ----
    loss_config_name = make_loss_config_name(loss_type, lam, mu)
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = datetime.now().strftime("%Y%m%d") + "_calibration_loss"

    experiment_dir = Path(pt.RESULTS_DIR) / experiment_name
    seed_dir = experiment_dir / dataset_name / loss_config_name / f"seed_{seed}"
    models_dir = seed_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger = TeeLogger.start(seed_dir / "experiment_log.txt")

    print(f"{'='*60}")
    print(f"Calibration-Aware Loss Experiment")
    print(f"  Loss: {loss_type} | λ={lam} | μ={mu}")
    print(f"  Dataset: {dataset_name} | Model: {model_name} | Seed: {seed}")
    print(f"  Epochs: {N_EPOCHS} | Early stop: {not DISABLE_EARLY_STOP}")
    print(f"  Output: {seed_dir}")
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
    early_stop = config['early_stop'] if not DISABLE_EARLY_STOP else False
    patience = config['patience']
    n_samples_train = config['n_samples_train']
    n_samples_valid = config['n_samples_valid']
    n_samples_test = args.n_samples_test

    # ---- Load & Split Data (seed controls the split) ----
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

    # ---- Build Loss Function ----
    loss_function, loss_desc = build_loss_function(loss_type, lam, mu, t_train, e_train)
    print(f"  Loss function: {loss_desc}")

    # ---- Build Data Loaders ----
    train_ds = InputFunction(X_train, t_train, e_train, batch_size=batch_size, drop_last=True, shuffle=True)()
    valid_ds = InputFunction(X_valid, t_valid, e_valid, batch_size=batch_size)()

    # ---- Build Model ----
    tf.keras.backend.clear_session()
    gc.collect()

    optimizer = tf.keras.optimizers.deserialize(optimizer_config)

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

    # ---- Train ----
    trainer = Trainer(
        model=model, model_name=model_name,
        train_dataset=train_ds, valid_dataset=valid_ds,
        test_dataset=None, optimizer=optimizer,
        loss_function=loss_function, num_epochs=N_EPOCHS,
        early_stop=early_stop, patience=patience,
        n_samples_train=n_samples_train,
        n_samples_valid=n_samples_valid,
        n_samples_test=n_samples_test,
        use_wandb=False,
        checkpoint_dir=models_dir,
    )
    train_start = timer()
    trainer.train_and_evaluate()
    train_time = timer() - train_start
    n_params = count_parameters(model)
    best_ep = trainer.best_ep
    print(f"\n  Trained in {train_time:.2f}s (best epoch: {best_ep})")

    # ---- Save Training Curves ----
    n_epochs_ran = len(trainer.train_total)
    curves_data = {"epoch": list(range(1, n_epochs_ran + 1))}
    for name, vals in [("train_total", trainer.train_total),
                       ("train_nll", trainer.train_nll),
                       ("train_kl", trainer.train_kl),
                       ("train_var", trainer.train_variance),
                       ("valid_total", trainer.valid_total),
                       ("valid_nll", trainer.valid_nll),
                       ("valid_kl", trainer.valid_kl),
                       ("valid_var", trainer.valid_variance)]:
        if vals and len(vals) == n_epochs_ran:
            curves_data[name] = vals
    pd.DataFrame(curves_data).to_csv(seed_dir / "training_curves.csv", index=False)
    print(f"  Training curves: {seed_dir / 'training_curves.csv'}")

    # ---- Restore Best Checkpoint ----
    trainer.checkpoint.restore(models_dir / f"ckpt-{best_ep}")
    model = trainer.model

    # ---- Evaluate ----
    test_start = timer()
    if model_name in ["mlp", "sngp"]:
        surv_preds = compute_deterministic_survival_curve(
            model, X_train, X_test, e_train, t_train, event_times, model_name)
    else:
        surv_preds = np.mean(compute_nondeterministic_survival_curve(
            model, np.array(X_train), np.array(X_test),
            e_train, t_train, event_times,
            n_samples_train, n_samples_test), axis=0)
    test_time = timer() - test_start

    surv_preds = pd.DataFrame(surv_preds, columns=event_times)
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

    # C-calibration for stochastic models
    c_calib = 0.0
    if model_name in ["vi", "mcd1", "mcd2", "mcd3"]:
        surv_probs = compute_nondeterministic_survival_curve(
            model, X_train, np.delete(X_test, bad_idx, axis=0),
            e_train, t_train, event_times, n_samples_train, n_samples_test
        )
        credible_region_sizes = np.arange(0.1, 1, 0.1)
        surv_times = torch.from_numpy(surv_probs)
        coverage_stats = {}
        for pct in credible_region_sizes:
            drop_num = math.floor(0.5 * n_samples_test * (1 - pct))
            lower = torch.kthvalue(surv_times, k=1 + drop_num, dim=0)[0]
            upper = torch.kthvalue(surv_times, k=n_samples_test - drop_num, dim=0)[0]
            coverage_stats[pct] = coverage(event_times, upper, lower,
                                           sanitized_t_test, sanitized_e_test)
        expected = [x / sum(coverage_stats.keys()) * 100 for x in coverage_stats.keys()]
        observed = [x / sum(coverage_stats.values()) * 100 for x in coverage_stats.values()]
        _, p_value = chisquare(f_obs=observed, f_exp=expected)
        c_calib = p_value

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
        print(f"  WARNING: ICI computation failed ({type(e).__name__}: {e}). Setting ICI=NaN.")
        ici = np.nan

    # ---- Print Results ----
    dcal_str = f"{d_calib:.4f}" if d_calib > 0.05 else f"{d_calib:.4f}*"
    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name} | {model_name} | {loss_desc} | seed={seed}")
    print(f"  CI:     {ci:.4f}")
    print(f"  IBS:    {ibs:.4f}")
    print(f"  D-Cal:  {dcal_str}")
    print(f"  C-Cal:  {c_calib:.4f}")
    print(f"  ICI:    {ici:.4f}")
    print(f"  MAE_H:  {mae_hinge:.2f}")
    print(f"  MAE_PO: {mae_pseudo:.2f}")
    print(f"  KM:     {km_mse:.6f}")
    print(f"  INBLL:  {inbll:.4f}")
    print(f"  Train:  {train_time:.2f}s | Test: {test_time:.2f}s")
    print(f"{'='*60}")

    # ---- Save Results ----
    metrics_dict = {
        "CI": ci, "IBS": ibs, "MAEHinge": mae_hinge, "MAEPseudo": mae_pseudo,
        "DCalib": d_calib, "CCalib": c_calib, "ICI": ici,
        "KM": km_mse, "INBLL": inbll,
    }

    # Save per-seed metrics
    res_df = pd.DataFrame([metrics_dict])
    res_df["ModelName"] = model_name
    res_df["DatasetName"] = dataset_name
    res_df["LossType"] = loss_type
    res_df["LossConfig"] = loss_config_name
    res_df["Lambda"] = lam
    res_df["Mu"] = mu
    res_df["Seed"] = seed
    res_df["TrainTime"] = train_time
    res_df["TestTime"] = test_time
    res_df["BestEpoch"] = best_ep
    res_df.to_csv(seed_dir / "metrics.csv", index=False)

    # Save config
    run_config = {
        "experiment_name": experiment_name,
        "dataset": dataset_name, "model": model_name,
        "loss_type": loss_type, "loss_desc": loss_desc,
        "loss_config": loss_config_name,
        "lambda": lam, "mu": mu, "seed": seed,
        "epochs": N_EPOCHS, "early_stop": early_stop,
        "patience": patience, "batch_size": batch_size,
        "network_layers": layers, "activation_fn": activation_fn,
        "dropout_rate": dropout_rate, "l2_reg": l2_reg,
        "learning_rate": float(optimizer.learning_rate.numpy()),
        "n_params": n_params if n_params else 0,
        "best_epoch": best_ep,
        "early_stopped": early_stop and (best_ep < N_EPOCHS),
        "train_time_s": round(train_time, 2),
        "test_time_s": round(test_time, 2),
        "timestamp": datetime.now().isoformat(),
    }
    with open(seed_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Append to master runs_index.csv
    index_row = {**metrics_dict, **{
        "DatasetName": dataset_name, "ModelName": model_name,
        "LossType": loss_type, "LossConfig": loss_config_name,
        "Lambda": lam, "Mu": mu, "Seed": seed,
        "BestEpoch": best_ep, "TrainTime": round(train_time, 2),
        "TestTime": round(test_time, 2),
        "Path": str(seed_dir),
    }}
    append_to_runs_index(experiment_dir / "runs_index.csv", index_row)

    print(f"  Results: {seed_dir / 'metrics.csv'}")

    # ---- Cleanup ----
    del trainer.checkpoint, trainer.manager
    del trainer, model
    tf.keras.backend.clear_session()
    gc.collect()

    logger.close()
