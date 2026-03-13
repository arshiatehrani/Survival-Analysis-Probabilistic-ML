import atexit, gc
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import tensorflow as tf

atexit.register(gc.collect)
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from tools.baysurv_trainer import Trainer
from utility.config import load_config
from utility.training import get_data_loader, scale_data, split_time_event
from tools.baysurv_builder import make_mlp_model, make_vi_model, make_mcd_model, make_sngp_model
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def plot_credible_interval(event_times, surv_ensemble, actual_time, actual_event,
                           model_name, dataset_name, sample_idx, save_dir):
    """Plot individual survival curve with 90% credible interval (Figure 2 from paper)."""
    surv_sample = surv_ensemble[:, sample_idx, :]  # (n_samples, n_times)
    mean_surv = np.mean(surv_sample, axis=0)
    lower = np.percentile(surv_sample, 5, axis=0)
    upper = np.percentile(surv_sample, 95, axis=0)
    median_time_idx = np.searchsorted(-mean_surv, -0.5)
    median_time = event_times[min(median_time_idx, len(event_times) - 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(event_times, mean_surv, 'b-', linewidth=2, label='Mean S(t)')
    ax1.fill_between(event_times, lower, upper, alpha=0.3, color='blue', label='90% CrI')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=median_time, color='red', linestyle='--', alpha=0.7, label=f'Median: {median_time:.0f}')
    if actual_event == 1:
        ax1.axvline(x=actual_time, color='green', linestyle='-', alpha=0.7, label=f'Actual: {actual_time:.0f}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Survival Probability')
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f'{model_name.upper()} - Individual Survival Function')
    ax1.legend(fontsize=8)

    median_times = []
    for s in range(surv_sample.shape[0]):
        idx = np.searchsorted(-surv_sample[s], -0.5)
        median_times.append(event_times[min(idx, len(event_times) - 1)])
    ax2.hist(median_times, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(x=np.mean(median_times), color='red', linestyle='--', label=f'Mean: {np.mean(median_times):.0f}')
    ax2.set_xlabel('Predicted Survival Time')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{model_name.upper()} - Predicted Time Distribution')
    ax2.legend(fontsize=8)

    fig.suptitle(f'{dataset_name} - Sample #{sample_idx}', fontsize=12)
    plt.tight_layout()
    save_path = Path(save_dir) / f"{dataset_name.lower()}_{model_name}_cri_sample{sample_idx}.pdf"
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path

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
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (overrides config files)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb experiment tracking")
    parser.add_argument("--wandb-project", default="baysurv-bnn", help="wandb project name")
    args = parser.parse_args()

    DATASETS = args.datasets
    MODELS = args.models
    N_EPOCHS = args.epochs
    CRI_SAMPLES = args.cri_samples
    DISABLE_EARLY_STOP = args.no_early_stop
    USE_WANDB = args.wandb

    if USE_WANDB:
        import wandb
        os.environ["WANDB_SILENT"] = "true"

    logger = TeeLogger.start(Path.joinpath(pt.RESULTS_DIR, "bnn_training_log.txt"))
    rg = ResultsGenerator(pt.RESULTS_DIR)

    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Early stopping: {'disabled (--no-early-stop)' if DISABLE_EARLY_STOP else 'per config'}")
    print(f"CrI samples: {CRI_SAMPLES}")
    print(f"Wandb: {'enabled' if USE_WANDB else 'disabled'}")
    # For each dataset, train models and plot scores
    for dataset_name in DATASETS:
        
        # Load training parameters
        config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
        optimizer_config = config['optimizer']
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
        print(f"  Split: train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}")
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
            else:
                raise ValueError("Model not found")
            
            # Create a fresh optimizer per model run (Keras 3 tracks variables per optimizer instance)
            optimizer = tf.keras.optimizers.deserialize(optimizer_config)

            # Train model
            trainer = Trainer(model=model, model_name=model_name,
                              train_dataset=train_ds, valid_dataset=valid_ds,
                              test_dataset=None, optimizer=optimizer,
                              loss_function=loss_function, num_epochs=N_EPOCHS,
                              early_stop=early_stop, patience=patience,
                              n_samples_train=n_samples_train,
                              n_samples_valid=n_samples_valid,
                              n_samples_test=n_samples_test,
                              use_wandb=USE_WANDB)
            train_start_time = time()
            trainer.train_and_evaluate()
            train_time = time() - train_start_time
            n_params = count_parameters(model)
            params_str = f" | params: {n_params:,}" if n_params else ""
            print(f"[{dataset_name}] {model_name} trained in {train_time:.2f}s (best epoch: {trainer.best_ep}){params_str}")

            # Get model for best epoch
            best_ep = trainer.best_ep
            status = trainer.checkpoint.restore(Path.joinpath(pt.MODELS_DIR, f"ckpt-{best_ep}"))
            model = trainer.model

            # Compute survival function
            test_start_time = time()
            if model_name in ["mlp", "sngp"]:
                surv_preds = compute_deterministic_survival_curve(model, X_train, X_test,
                                                                  e_train, t_train, event_times, model_name)
            else:
                surv_preds = np.mean(compute_nondeterministic_survival_curve(model, np.array(X_train), np.array(X_test),
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
            
            # Calculate C-cal for BNN models
            if model_name in ["vi", "mcd1", "mcd2", "mcd3"]:
                surv_probs = compute_nondeterministic_survival_curve(model, X_train, sanitized_x_test,
                                                                     e_train, t_train, event_times,
                                                                     n_samples_train, n_samples_test)
                credible_region_sizes = np.arange(0.1, 1, 0.1)
                surv_times = torch.from_numpy(surv_probs)
                coverage_stats = {}
                for percentage in credible_region_sizes:
                    drop_num = math.floor(0.5 * n_samples_test * (1 - percentage))
                    lower_outputs = torch.kthvalue(surv_times, k=1 + drop_num, dim=0)[0]
                    upper_outputs = torch.kthvalue(surv_times, k=n_samples_test - drop_num, dim=0)[0]
                    coverage_stats[percentage] = coverage(event_times, upper_outputs, lower_outputs,
                                                          sanitized_t_test, sanitized_e_test)
                expected_percentages = coverage_stats.keys()
                observed_percentages = coverage_stats.values()
                expected = [x / sum(expected_percentages) * 100 for x in expected_percentages]
                observed = [x / sum(observed_percentages) * 100 for x in observed_percentages]
                _, p_value = chisquare(f_obs=observed, f_exp=expected)
                c_calib = p_value

                # Save CrI plot for a random test sample (Figure 2 from paper)
                try:
                    if CRI_SAMPLES > n_samples_test:
                        cri_surv_probs = compute_nondeterministic_survival_curve(
                            model, X_train, sanitized_x_test,
                            e_train, t_train, event_times,
                            n_samples_train, CRI_SAMPLES)
                    else:
                        cri_surv_probs = surv_probs
                    sample_idx = min(42, cri_surv_probs.shape[1] - 1)
                    cri_path = plot_credible_interval(
                        event_times, cri_surv_probs,
                        sanitized_t_test[sample_idx], sanitized_e_test[sample_idx],
                        model_name, dataset_name, sample_idx, pt.RESULTS_DIR)
                    print(f"     CrI plot saved ({CRI_SAMPLES} MC samples): {cri_path}")
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
            weights_dir = Path.joinpath(pt.MODELS_DIR, f"{dataset_name.lower()}_{model_name.lower()}")
            weights_dir.mkdir(parents=True, exist_ok=True)
            path = weights_dir / "weights.weights.h5"
            model.save_weights(path)
            print(f"  -> Model saved: {path}")

            # Clean up checkpoint references to avoid TF shutdown errors
            del trainer.checkpoint, trainer.manager
            del trainer

            # Save results
            training_results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"baysurv_training_results.csv"), index=False)
            test_results.to_csv(Path.joinpath(pt.RESULTS_DIR, f"baysurv_test_results.csv"), index=False)

        # After all models for this dataset: joint calibration + loss curves
        rg.plot_calibration_curves_all(all_calib_data, event_times_pct,
                                        trained_models, dataset_name)
        rg.plot_training_loss_curves(training_results, dataset_name,
                                      trained_models)

    logger.close()

