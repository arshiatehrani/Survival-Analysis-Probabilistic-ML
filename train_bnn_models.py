import atexit, gc
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import tensorflow as tf

def _cleanup_tf():
    """Clear TF session before exit to avoid _CheckpointRestoreCoordinatorDeleter TypeError."""
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

atexit.register(_cleanup_tf)
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from tools.baysurv_trainer import Trainer
from utility.config import load_config
from utility.training import get_data_loader, scale_data, split_time_event
from tools.baysurv_builder import make_mlp_model, make_vi_model, make_mcd_model, make_sngp_model, make_transformer_mcd_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss, CoxPHLossGaussian
from pathlib import Path
import paths as pt
from utility.survival import (calculate_event_times, calculate_percentiles, convert_to_structured,
                              compute_deterministic_survival_curve, compute_nondeterministic_survival_curve)
from utility.training import make_stratified_split
from time import time
from tools.evaluator import LifelinesEvaluator
from pycox.evaluation import EvalSurv
import math
from utility.survival import coverage
from scipy.stats import chisquare
import torch
from utility.survival import survival_probability_calibration
from tools.Evaluations.util import make_monotonic, check_monotonicity

import argparse
import os
from tools.results_generator import ResultsGenerator, TeeLogger
from utility.plot import plot_credible_interval
from utility.run_manager import RunManager

import warnings
import copy
warnings.simplefilter(action='ignore', category=FutureWarning)
# TFP DenseFlipout uses deprecated add_variable; suppress until TFP updates
warnings.filterwarnings('ignore', message='.*add_variable.*deprecated.*')
warnings.filterwarnings('ignore', message='.*RandomNormal is unseeded.*')

np.seterr(divide='ignore')
np.seterr(invalid='ignore')

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

ALL_DATASETS = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
ALL_MODELS = ["mlp", "sngp", "mcd1", "mcd2", "mcd3", "vi"]
N_EPOCHS = 100

def count_parameters(model):
    """Count trainable parameters for a Keras model."""
    try:
        return model.count_params()
    except Exception:
        return None

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {gpus}")
else:
    print("No GPU found, using CPU")

test_results = pd.DataFrame()
training_results = pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BNN survival analysis models")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS, help="Datasets to train on (default: all)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, help="Models to train (default: all)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--cri-samples", type=int, default=1000,
                        help="Number of MC samples for CrI visualization (paper uses 1000, default: 1000)")
    parser.add_argument("--max-full-mc-samples", type=int, default=32,
                        help="Max MC samples for full tensor operations (C-cal/CrI) to avoid OOM (default: 32)")
    parser.add_argument("--max-cri-samples", type=int, default=128,
                        help="Max MC samples for CrI plotting to avoid OOM (default: 128)")
    parser.add_argument("--cri-plot-samples", type=str, default=None,
                        help="Comma-separated sample indices for CrI plots (e.g. 0,42,100). Default: 42")
    parser.add_argument("--cri-plot-all", action="store_true",
                        help="Plot CrI for all test samples (many PDFs)")
    parser.add_argument("--cri-plot-random", action="store_true",
                        help="Use random sample when not specifying --cri-plot-samples or --cri-plot-all")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (overrides config files)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb experiment tracking")
    parser.add_argument("--wandb-project", default="baysurv-bnn", help="wandb project name")
    parser.add_argument("--tune", action="store_true",
                        help="Run Bayesian optimization per dataset before training (requires WANDB_API_KEY)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Use pre-tuned configs from configs/mlp/ (default)")
    parser.add_argument("--tune-iterations", type=int, default=10,
                        help="Bayesian optimization iterations when --tune (paper: 10)")
    args = parser.parse_args()

    DATASETS = args.datasets
    MODELS = args.models
    N_EPOCHS = args.epochs
    CRI_SAMPLES = args.cri_samples
    MAX_FULL_MC_SAMPLES = args.max_full_mc_samples
    MAX_CRI_SAMPLES = args.max_cri_samples
    CRI_PLOT_SAMPLES = args.cri_plot_samples
    CRI_PLOT_ALL = args.cri_plot_all
    CRI_PLOT_RANDOM = args.cri_plot_random
    DISABLE_EARLY_STOP = args.no_early_stop
    USE_WANDB = args.wandb
    TUNE_FIRST = args.tune and not args.no_tune

    if USE_WANDB or TUNE_FIRST:
        import wandb
        os.environ["WANDB_SILENT"] = "true"

    run = RunManager(
        base_results_dir=pt.RESULTS_DIR,
        script_name="train_bnn_models.py",
        datasets=DATASETS,
        models=MODELS,
        cli_args=vars(args),
    )
    logger = TeeLogger.start(run.run_dir / "bnn_training_log.txt")
    rg = ResultsGenerator(run.run_dir)

    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Early stopping: {'disabled (--no-early-stop)' if DISABLE_EARLY_STOP else 'per config'}")
    print(f"CrI samples: {CRI_SAMPLES}")
    print(f"Max full MC samples (OOM guard): {MAX_FULL_MC_SAMPLES}")
    print(f"Max CrI samples (OOM guard): {MAX_CRI_SAMPLES}")
    cri_plot_desc = "all" if CRI_PLOT_ALL else (CRI_PLOT_SAMPLES or ("random" if CRI_PLOT_RANDOM else "42"))
    print(f"CrI plot samples: {cri_plot_desc}")
    print(f"Wandb: {'enabled' if USE_WANDB else 'disabled'}")
    print(f"Hyperparameters: {'Bayesian optimization (--tune)' if TUNE_FIRST else 'pre-tuned configs (--no-tune)'}")
    print("  Reg = regularization term (KL for VI, L2 for MLP/MCD/SNGP, 0 if none)")

    # Run Bayesian optimization per dataset if --tune
    if TUNE_FIRST:
        import subprocess
        import sys
        for ds in DATASETS:
            print(f"\n[Bayesian optimization] Tuning {ds} ({args.tune_iterations} iterations)...")
            ret = subprocess.run([
                sys.executable, "tuning/tune_mlp_model.py",
                "--dataset", ds,
                "--iterations", str(args.tune_iterations),
                "--save-config",
            ], cwd=str(pt.ROOT_DIR))
            if ret.returncode != 0:
                print(f"[Bayesian optimization] Tuning failed for {ds}; exiting.")
                sys.exit(ret.returncode)
        print("\n[Bayesian optimization] Tuning complete. Starting training...\n")

    # For each dataset, train models and plot scores
    for dataset_name in DATASETS:
        
        # Load training parameters
        config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        optimizer_config = copy.deepcopy(config['optimizer'])
        if isinstance(optimizer_config, dict) and isinstance(optimizer_config.get("config"), dict):
            optimizer_config["config"].pop("decay", None)
        optimizer = tf.keras.optimizers.deserialize(optimizer_config)
        activation_fn = config['activiation_fn']
        layers = config['network_layers']
        l2_reg = config['l2_reg']
        batch_size = config['batch_size']
        early_stop = config['early_stop'] if not DISABLE_EARLY_STOP else False
        patience = config['patience']
        n_samples_train = config['n_samples_train']
        n_samples_valid = config['n_samples_valid']
        n_samples_test = config['n_samples_test']
        loss_function = CoxPHLoss()

        # Load data
        dl = get_data_loader(dataset_name).load_data()
        num_features, cat_features = dl.get_features()
        df = dl.get_data()

        event_rate = df["event"].mean()
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} | Samples: {len(df)} | "
                   f"Features: {len(num_features)} num + {len(cat_features)} cat | "
                   f"Event rate: {event_rate:.1%}")
        print(f"Config: layers={layers}, batch={batch_size}, lr={optimizer.learning_rate.numpy():.1e}, "
                   f"epochs={N_EPOCHS}, early_stop={early_stop}, patience={patience}")
        print(f"{'='*60}")

        # Split data
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                            frac_valid=0.1, frac_test=0.2, random_state=0)
        X_train = df_train[cat_features+num_features]
        X_valid = df_valid[cat_features+num_features]
        X_test = df_test[cat_features+num_features]
        n_train, n_valid, n_test = len(X_train), len(X_valid), len(X_test)
        e_train_pct = df_train["event"].mean() * 100
        e_valid_pct = df_valid["event"].mean() * 100
        e_test_pct = df_test["event"].mean() * 100
        print(f"  Data layout: 70% train / 10% valid / 20% test (stratified by time+event)")
        print(f"  Split: train={n_train} ({e_train_pct:.1f}% events) | valid={n_valid} ({e_valid_pct:.1f}% events) | test={n_test} ({e_test_pct:.1f}% events)")
        y_train = convert_to_structured(df_train["time"], df_train["event"])
        y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
        y_test = convert_to_structured(df_test["time"], df_test["event"])

        # Scale data
        X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
        
        # Convert to array
        X_train = np.array(X_train)
        X_valid = np.array(X_valid)
        X_test = np.array(X_test)

        # Make time/event split
        t_train, e_train = split_time_event(y_train)
        t_valid, e_valid = split_time_event(y_valid)
        t_test, e_test = split_time_event(y_test)

        # Make event times
        event_times = calculate_event_times(t_train, e_train)
        
        # Calculate quantiles
        event_times_pct = calculate_percentiles(event_times)

        # Make data loaders
        train_ds = InputFunction(X_train, t_train, e_train, batch_size=batch_size, drop_last=True, shuffle=True)()
        valid_ds = InputFunction(X_valid, t_valid, e_valid, batch_size=batch_size)()
        test_ds = InputFunction(X_test, t_test, e_test, batch_size=batch_size)()

        # Make models
        
        all_calib_data = {}
        trained_models = []

        for i, model_name in enumerate(MODELS):
            print(f"\n[{dataset_name}] ({i+1}/{len(MODELS)}) Training {model_name} ...", flush=True)

            # Aggressive cleanup before each model to avoid OOM on small GPU (e.g. MIG 1g.10gb)
            tf.keras.backend.clear_session()
            gc.collect()

            if USE_WANDB:
                wandb.init(
                    project=args.wandb_project,
                    name=f"{dataset_name}_{model_name}",
                    group=dataset_name,
                    tags=["bnn", model_name, dataset_name],
                    config={"model": model_name, "dataset": dataset_name,
                            "n_train": len(X_train), "n_features": X_train.shape[1],
                            "layers": layers, "activation": activation_fn,
                            "batch_size": batch_size, "epochs": N_EPOCHS,
                            "l2_reg": l2_reg, "early_stop": early_stop,
                            "patience": patience, "lr": float(optimizer.learning_rate.numpy())},
                    reinit=True,
                )

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
            elif model_name == "mcd1":
                dropout_rate = 0.1
                model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                                       layers=layers, activation_fn=activation_fn,
                                       dropout_rate=dropout_rate, regularization_pen=l2_reg)
            elif model_name == "mcd2":
                dropout_rate = 0.2
                model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                                       layers=layers, activation_fn=activation_fn,
                                       dropout_rate=dropout_rate, regularization_pen=l2_reg)
            elif model_name == "mcd3":
                dropout_rate = 0.5
                model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                                       layers=layers, activation_fn=activation_fn,
                                       dropout_rate=dropout_rate, regularization_pen=l2_reg)
            elif model_name == "transformer_mcd":
                dropout_rate = config['dropout_rate']
                model = make_transformer_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                                                   layers=layers, activation_fn=activation_fn,
                                                   dropout_rate=dropout_rate, regularization_pen=l2_reg)
            else:
                raise ValueError("Model not found")
            
            # Create a fresh optimizer per model run (Keras 3 tracks variables per optimizer instance)
            optimizer = tf.keras.optimizers.deserialize(optimizer_config)

            # Train model (paper: same hyperparams for all BNN models)
            trainer = Trainer(model=model, model_name=model_name,
                              train_dataset=train_ds, valid_dataset=valid_ds,
                              test_dataset=None, optimizer=optimizer,
                              loss_function=loss_function, num_epochs=N_EPOCHS,
                              early_stop=early_stop, patience=patience,
                              n_samples_train=n_samples_train,
                              n_samples_valid=n_samples_valid,
                              n_samples_test=n_samples_test,
                              use_wandb=USE_WANDB,
                              checkpoint_dir=run.models_dir)
            train_start_time = time()
            trainer.train_and_evaluate()
            train_time = time() - train_start_time
            n_params = count_parameters(model)
            params_str = f" | params: {n_params:,}" if n_params else ""
            print(f"[{dataset_name}] {model_name} trained in {train_time:.2f}s (best epoch: {trainer.best_ep}){params_str}")

            # Get model for best epoch
            best_ep = trainer.best_ep
            status = trainer.checkpoint.restore(Path.joinpath(run.models_dir, f"ckpt-{best_ep}"))
            model = trainer.model

            # Compute survival function
            test_start_time = time()
            if model_name in ["mlp", "sngp"]:
                surv_preds = compute_deterministic_survival_curve(model, X_train, X_test,
                                                                  e_train, t_train, event_times, model_name)
            else:
                surv_preds = np.mean(compute_nondeterministic_survival_curve(
                    model, np.array(X_train), np.array(X_test),
                    e_train, t_train, event_times,
                    n_samples_train, n_samples_test), axis=0)
            test_time = time() - test_start_time
            
            # Make dataframe
            surv_preds = pd.DataFrame(surv_preds, columns=event_times)
            
            # Sanitize
            surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0)
            bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
            sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
            sanitized_y_test = np.delete(y_test, bad_idx, axis=0)
            sanitized_x_test = np.delete(X_test, bad_idx, axis=0)
            sanitized_t_test = np.delete(t_test, bad_idx, axis=0)
            sanitized_e_test = np.delete(e_test, bad_idx, axis=0)

            # Compute metrics
            lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_y_test["time"], sanitized_y_test["event"], t_train, e_train)
            ibs = lifelines_eval.integrated_brier_score()
            mae_hinge = lifelines_eval.mae(method="Hinge")
            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
            d_calib = lifelines_eval.d_calibration()[0] # 1 if lifelines_eval.d_calibration()[0] > 0.05 else 0
            km_mse = lifelines_eval.km_calibration()
            ev = EvalSurv(sanitized_surv_preds.T, sanitized_y_test["time"], sanitized_y_test["event"], censor_surv="km")
            inbll = ev.integrated_nbll(event_times)
            ci = ev.concordance_td()
            
            # Calculate C-cal for BNN models (single MC pass for C-cal + CrI plot when both needed)
            if model_name in ["vi", "mcd1", "mcd2", "mcd3"]:
                calib_samples = min(n_samples_test, MAX_FULL_MC_SAMPLES)
                cri_samples = min(CRI_SAMPLES, MAX_CRI_SAMPLES)
                n_use = max(calib_samples, cri_samples)
                if calib_samples < n_samples_test:
                    print(f"     MC samples capped for calibration: {calib_samples}/{n_samples_test} (OOM guard)")
                if cri_samples < CRI_SAMPLES:
                    print(f"     CrI samples capped: {cri_samples}/{CRI_SAMPLES} (OOM guard)")
                surv_probs = compute_nondeterministic_survival_curve(model, X_train, sanitized_x_test,
                                                                     e_train, t_train, event_times,
                                                                     n_samples_train, n_use)
                credible_region_sizes = np.arange(0.1, 1, 0.1)
                surv_times = torch.from_numpy(surv_probs)
                coverage_stats = {}
                for percentage in credible_region_sizes:
                    drop_num = math.floor(0.5 * n_use * (1 - percentage))
                    lower_outputs = torch.kthvalue(surv_times, k=1 + drop_num, dim=0)[0]
                    upper_outputs = torch.kthvalue(surv_times, k=n_use - drop_num, dim=0)[0]
                    coverage_stats[percentage] = coverage(event_times, upper_outputs, lower_outputs,
                                                          sanitized_t_test, sanitized_e_test)
                expected_percentages = coverage_stats.keys()
                observed_percentages = coverage_stats.values()
                expected = [x / sum(expected_percentages) * 100 for x in expected_percentages]
                observed = [x / sum(observed_percentages) * 100 for x in observed_percentages]
                _, p_value = chisquare(f_obs=observed, f_exp=expected)
                c_calib = p_value

                # Save CrI plot(s) (Figure 2 from paper). Generic for vi, mcd1, mcd2, mcd3.
                try:
                    cri_surv_probs = surv_probs[:cri_samples] if cri_samples <= n_use else surv_probs
                    n_test = cri_surv_probs.shape[1]
                    if CRI_PLOT_ALL:
                        sample_indices = list(range(n_test))
                    elif CRI_PLOT_SAMPLES:
                        sample_indices = [int(x.strip()) for x in CRI_PLOT_SAMPLES.split(",")]
                    else:
                        idx = random.randint(0, n_test - 1) if CRI_PLOT_RANDOM else min(42, n_test - 1)
                        sample_indices = [idx]
                    sample_indices = [i for i in sample_indices if 0 <= i < n_test]
                    if not sample_indices:
                        sample_indices = [0]
                    saved_paths = []
                    for sample_idx in sample_indices:
                        path = plot_credible_interval(
                            event_times, cri_surv_probs,
                            sanitized_t_test[sample_idx], sanitized_e_test[sample_idx],
                            model_name, dataset_name, sample_idx, run.run_dir)
                        saved_paths.append(path)
                    print(f"     CrI plot(s) saved ({len(saved_paths)} file(s), {CRI_SAMPLES} MC samples): {saved_paths[0]}" + (f" ... +{len(saved_paths)-1} more" if len(saved_paths) > 1 else ""))
                except Exception as e:
                    print(f"     CrI plot skipped: {e}")
            else:
                c_calib = 0
            
            # Compute calibration curves
            deltas = dict()
            for t0 in event_times_pct.values():
                _, _, _, deltas_t0 = survival_probability_calibration(sanitized_surv_preds,
                                                                      sanitized_y_test["time"],
                                                                      sanitized_y_test["event"],
                                                                      t0)
                deltas[t0] = deltas_t0
            ici = deltas[t0].mean()
            
            # Save to df
            metrics = [ci, ibs, mae_hinge, mae_pseudo, d_calib, km_mse, inbll, c_calib, ici, train_time, test_time]
            res_df = pd.DataFrame(np.column_stack(metrics), columns=["CI", "IBS", "MAEHinge", "MAEPseudo", "DCalib", "KM",
                                                                     "INBLL", "CCalib", "ICI", "TrainTime", "TestTime"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            res_df['NParams'] = n_params if n_params else 0

            dcal_str = f"{d_calib:.3f}" if d_calib > 0.05 else f"{d_calib:.3f}*"
            ccal_str = f"{c_calib:.3f}" if model_name in ['vi', 'mcd1', 'mcd2', 'mcd3'] else "-"
            print(f"  -> Inference: {test_time:.2f}s")
            print(f"     CI: {ci:.4f} | IBS: {ibs:.4f} | MAE_H: {mae_hinge:.2f} | MAE_PO: {mae_pseudo:.2f}")
            print(f"     ICI: {ici:.4f} | D-Cal: {dcal_str} | C-Cal: {ccal_str}")

            times_for_pct = np.concatenate([t_train, t_test])
            ext_metrics, calib_data = rg.generate_all(
                evaluator=lifelines_eval,
                event_times=event_times,
                t_test=sanitized_t_test,
                e_test=sanitized_e_test,
                dataset_name=dataset_name,
                model_name=model_name,
                surv_preds_df=sanitized_surv_preds,
                event_times_pct=event_times_pct,
                event_times_for_pct=times_for_pct,
            )
            for k, v in ext_metrics.items():
                res_df[k] = v
            all_calib_data[model_name] = calib_data
            trained_models.append(model_name)
            test_results = pd.concat([test_results, res_df], axis=0)

            if USE_WANDB:
                wandb.log({
                    "CI": ci, "IBS": ibs, "INBLL": inbll,
                    "MAE_Hinge": mae_hinge, "MAE_Pseudo": mae_pseudo,
                    "DCalib": d_calib, "KM_MSE": km_mse,
                    "CCalib": c_calib, "ICI": ici,
                    "train_time": train_time, "test_time": test_time,
                    "best_epoch": trainer.best_ep,
                })
                wandb.finish()

            # Save loss and variance from training
            train_loss = trainer.train_loss
            train_variance = trainer.train_variance
            valid_loss = trainer.valid_loss
            valid_variance = trainer.valid_variance
            res_df = pd.DataFrame(np.column_stack([train_loss, train_variance, valid_loss, valid_variance]),
                                  columns=["TrainLoss", "TrainVariance", "ValidLoss", "ValidVariance"])
            res_df['ModelName'] = model_name
            res_df['DatasetName'] = dataset_name
            training_results = pd.concat([training_results, res_df], axis=0)
            
            # Save model
            weights_dir = run.models_dir / f"{dataset_name.lower()}_{model_name.lower()}"
            weights_dir.mkdir(parents=True, exist_ok=True)
            path = weights_dir / "weights.weights.h5"
            model.save_weights(path)
            print(f"  -> Model saved: {path}")

            # Log run metadata
            model_config = {
                "network_layers": layers, "activation_fn": activation_fn,
                "dropout_rate": dropout_rate, "l2_reg": l2_reg,
                "batch_size": batch_size, "learning_rate": float(optimizer.learning_rate.numpy()),
                "n_samples_train": n_samples_train, "n_samples_valid": n_samples_valid,
                "n_samples_test": n_samples_test,
            }
            model_metrics = {
                "CI": ci, "IBS": ibs, "MAEHinge": mae_hinge, "MAEPseudo": mae_pseudo,
                "DCalib": d_calib, "CCalib": c_calib, "ICI": ici,
                "KM": km_mse, "INBLL": inbll,
            }
            run.log_model_result(dataset_name, model_name,
                config=model_config, metrics=model_metrics,
                extra={
                    "n_params": n_params if n_params else 0,
                    "best_epoch": best_ep,
                    "early_stopped": early_stop and (best_ep < N_EPOCHS),
                    "train_time_s": round(train_time, 2),
                    "test_time_s": round(test_time, 2),
                })

            # Clean up to free GPU memory before next model (critical for MIG 1g.10gb)
            del trainer.checkpoint, trainer.manager
            del trainer
            del model
            tf.keras.backend.clear_session()
            gc.collect()

            # Save results
            training_results.to_csv(run.run_dir / "baysurv_training_results.csv", index=False)
            test_results.to_csv(run.run_dir / "baysurv_test_results.csv", index=False)

        # After all models for this dataset: joint calibration + loss curves
        rg.plot_calibration_curves_all(all_calib_data, event_times_pct,
                                        trained_models, dataset_name)
        rg.plot_training_loss_curves(training_results, dataset_name,
                                      trained_models)

    logger.close()
    run.finalize()

