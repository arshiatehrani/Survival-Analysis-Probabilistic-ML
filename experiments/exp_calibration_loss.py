"""Calibration-Aware Loss Experiment Script.

Trains one BNN model on one dataset with a configurable loss function.
Each invocation creates one run in the experiment tracking system.

Usage examples:
    # Baseline (original Cox PH)
    python experiments/exp_calibration_loss.py --dataset METABRIC --loss-type cox --model mcd1

    # CRPS only
    python experiments/exp_calibration_loss.py --dataset METABRIC --loss-type crps --model mcd1

    # Joint Cox + CRPS with lambda=0.3
    python experiments/exp_calibration_loss.py --dataset METABRIC --loss-type joint_crps --lambda-val 0.3

    # Joint Cox + CRPS + marginal KL
    python experiments/exp_calibration_loss.py --dataset METABRIC --loss-type joint_crps_kl --lambda-val 0.3 --mu-val 0.1
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import gc
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import chisquare
from pathlib import Path
from time import time as timer
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*add_variable.*deprecated.*')
warnings.filterwarnings('ignore', message='.*RandomNormal is unseeded.*')

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
from tools.results_generator import ResultsGenerator, TeeLogger
from pycox.evaluation import EvalSurv
from utility.run_manager import RunManager
from tools.Evaluations.util import make_monotonic, check_monotonicity

np.seterr(divide='ignore', invalid='ignore')
np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

ALL_DATASETS = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
ALL_MODELS = ["mlp", "sngp", "mcd1", "mcd2", "mcd3", "vi"]
N_GRID_POINTS = 100  # Time grid resolution for CRPS/IBS losses

def count_parameters(model):
    try:
        return model.count_params()
    except Exception:
        return None

def compute_censoring_km(t_train, e_train):
    """Compute KM estimate of the *censoring* distribution G(t) = P(C > t).
    Flip events: censoring is the 'event' for G(t)."""
    censoring_indicator = 1 - e_train.astype(int)
    km = KaplanMeier(t_train, censoring_indicator)
    return km.survival_times, km.survival_probabilities

def build_time_grid(t_train, n_points=N_GRID_POINTS):
    """Create a time grid from 0 to max(t_train) with n_points."""
    return np.linspace(0, t_train.max(), n_points).astype(np.float32)

def build_loss_function(loss_type, lam, mu, t_train, e_train):
    """Construct the loss function based on CLI args.

    Returns
    -------
    loss_fn : tf.keras.losses.Loss
    loss_desc : str
        Human-readable description for metadata.
    """
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
        description="Calibration-aware loss experiment (single model × single dataset)"
    )
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                        help="Dataset to train on")
    parser.add_argument("--model", default="mcd1", choices=ALL_MODELS,
                        help="Model architecture (default: mcd1)")
    parser.add_argument("--loss-type", required=True,
                        choices=["cox", "ibs", "crps", "joint_ibs", "joint_crps", "joint_crps_kl"],
                        help="Loss function type")
    parser.add_argument("--lambda-val", type=float, default=0.3,
                        help="Lambda for joint losses (default: 0.3)")
    parser.add_argument("--mu-val", type=float, default=0.0,
                        help="Mu for marginal KL regularizer (default: 0.0)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs (default: 100)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--n-samples-test", type=int, default=100,
                        help="MC samples for evaluation (default: 100)")
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    loss_type = args.loss_type
    lam = args.lambda_val
    mu = args.mu_val
    N_EPOCHS = args.epochs
    DISABLE_EARLY_STOP = args.no_early_stop

    # ---- Run Manager ----
    run_tag = f"{loss_type}"
    if "joint" in loss_type:
        run_tag += f"_lam{lam}"
    if mu > 0:
        run_tag += f"_mu{mu}"

    run = RunManager(
        base_results_dir=pt.RESULTS_DIR,
        script_name="exp_calibration_loss.py",
        datasets=[dataset_name],
        models=[model_name],
        cli_args=vars(args),
    )
    logger = TeeLogger.start(run.run_dir / "experiment_log.txt")

    print(f"{'='*60}")
    print(f"Calibration-Aware Loss Experiment")
    print(f"  Loss: {loss_type} | λ={lam} | μ={mu}")
    print(f"  Dataset: {dataset_name} | Model: {model_name}")
    print(f"  Epochs: {N_EPOCHS} | Early stop: {not DISABLE_EARLY_STOP}")
    print(f"  Run dir: {run.run_dir}")
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

    # ---- Load & Split Data ----
    dl = get_data_loader(dataset_name).load_data()
    num_features, cat_features = dl.get_features()
    df = dl.get_data()
    print(f"  Samples: {len(df)} | Features: {len(num_features)} num + {len(cat_features)} cat")

    df_train, df_valid, df_test = make_stratified_split(
        df, stratify_colname='both', frac_train=0.7,
        frac_valid=0.1, frac_test=0.2, random_state=0
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
        checkpoint_dir=run.models_dir,
    )
    train_start = timer()
    trainer.train_and_evaluate()
    train_time = timer() - train_start
    n_params = count_parameters(model)
    best_ep = trainer.best_ep
    print(f"\n  Trained in {train_time:.2f}s (best epoch: {best_ep})")

    # ---- Restore Best Checkpoint ----
    trainer.checkpoint.restore(run.models_dir / f"ckpt-{best_ep}")
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
    deltas = {}
    for t0 in event_times_pct.values():
        _, _, _, deltas_t0 = survival_probability_calibration(
            sanitized_surv_preds, sanitized_y_test["time"],
            sanitized_y_test["event"], t0
        )
        deltas[t0] = deltas_t0
    ici = list(deltas.values())[-1].mean()

    # ---- Print Results ----
    dcal_str = f"{d_calib:.4f}" if d_calib > 0.05 else f"{d_calib:.4f}*"
    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name} | {model_name} | {loss_desc}")
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
    res_df = pd.DataFrame([metrics_dict])
    res_df["ModelName"] = model_name
    res_df["DatasetName"] = dataset_name
    res_df["LossType"] = loss_type
    res_df["Lambda"] = lam
    res_df["Mu"] = mu
    res_df["TrainTime"] = train_time
    res_df["TestTime"] = test_time
    res_df["BestEpoch"] = best_ep
    res_df.to_csv(run.run_dir / "experiment_results.csv", index=False)

    # Save model weights
    weights_dir = run.models_dir / f"{dataset_name.lower()}_{model_name.lower()}"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_dir / "weights.weights.h5")
    print(f"  Model saved: {weights_dir}")

    # ---- Log Run Metadata ----
    model_config = {
        "network_layers": layers, "activation_fn": activation_fn,
        "dropout_rate": dropout_rate, "l2_reg": l2_reg,
        "batch_size": batch_size,
        "learning_rate": float(optimizer.learning_rate.numpy()),
        "loss_type": loss_type, "loss_desc": loss_desc,
        "lambda": lam, "mu": mu,
    }
    run.log_model_result(
        dataset_name, model_name,
        config=model_config,
        metrics=metrics_dict,
        extra={
            "n_params": n_params if n_params else 0,
            "best_epoch": best_ep,
            "early_stopped": early_stop and (best_ep < N_EPOCHS),
            "train_time_s": round(train_time, 2),
            "test_time_s": round(test_time, 2),
        }
    )

    # ---- Cleanup ----
    del trainer.checkpoint, trainer.manager
    del trainer, model
    tf.keras.backend.clear_session()
    gc.collect()

    logger.close()
    run.finalize()
