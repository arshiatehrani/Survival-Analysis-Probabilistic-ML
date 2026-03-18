"""MC Sample Efficiency Experiment Script.

Loads trained MCD models from a calibration-loss experiment,
and evaluates performance (CI, IBS, D-Cal) and inference time
across different numbers of Monte Carlo samples during inference.

Usage:
    python experiments/exp_mc_samples.py \
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
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*add_variable.*deprecated.*')


# Monkey-patch: scipy >= 1.14 removed simps, but pycox still uses it
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import paths as pt
from utility.config import load_config
from utility.training import get_data_loader, make_stratified_split, scale_data, split_time_event
from utility.survival import (calculate_event_times, calculate_percentiles, convert_to_structured,
                              compute_nondeterministic_survival_curve)
from tools.baysurv_builder import make_mcd_model
from experiments.exp_posthoc_calibration import evaluate_survival_preds # reuse existing evaluator

ALL_DATASETS = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
ALL_MODELS = ["mcd1", "mcd2", "mcd3", "vi"]
MC_SAMPLES_LIST = [1, 5, 10, 25, 50, 100, 200, 500]

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
        description="MC Sample Efficiency Experiment"
    )
    parser.add_argument("--source-experiment", required=True,
                        help="Name of the calibration-loss experiment to read checkpoints from")
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--model", default="mcd1", choices=ALL_MODELS)
    parser.add_argument("--loss-config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Output experiment name (default: auto appended with _mc_samples)")
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    loss_config = args.loss_config
    seed = args.seed

    # ---- Set seeds ----
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ---- Locate source checkpoint ----
    source_dir = Path(pt.RESULTS_DIR) / args.source_experiment / dataset_name / loss_config / f"seed_{seed}"
    models_dir = source_dir / "models"
    config_path = source_dir / "config.json"

    if not config_path.exists():
        print(f"ERROR: Source config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        run_config = json.load(f)

    best_ep = run_config.get("best_epoch", 1)

    # ---- Output directory ----
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = f"{args.source_experiment}_mc_samples"

    out_dir = Path(pt.RESULTS_DIR) / experiment_name / dataset_name / loss_config / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"MC Sample Efficiency Experiment")
    print(f"  Source: {args.source_experiment}/{dataset_name}/{loss_config}/seed_{seed}")
    print(f"  Output: {out_dir}")
    print(f"  MC Samples sequence: {MC_SAMPLES_LIST}")
    print(f"{'='*60}")

    # ---- Load Config ----
    config = load_config(pt.MLP_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    l2_reg = config['l2_reg']
    n_samples_train = config['n_samples_train']

    # ---- Load & Split Data (same seed = same split) ----
    dl = get_data_loader(dataset_name).load_data()
    num_features, cat_features = dl.get_features()
    df = dl.get_data()

    df_train, df_valid, df_test = make_stratified_split(
        df, stratify_colname='both', frac_train=0.7,
        frac_valid=0.1, frac_test=0.2, random_state=seed
    )
    X_train = df_train[cat_features + num_features]
    X_valid = df_valid[cat_features + num_features]
    X_test = df_test[cat_features + num_features]

    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])

    X_train, _, X_test = scale_data(X_train, X_valid=X_valid, X_test=X_test, 
                                     cat_features=cat_features, num_features=num_features)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    t_train, e_train = split_time_event(y_train)
    t_test, e_test = split_time_event(y_test)

    event_times = calculate_event_times(t_train, e_train)
    event_times_pct = calculate_percentiles(event_times)

    # ---- Rebuild model and restore checkpoint ----
    tf.keras.backend.clear_session()
    gc.collect()

    if model_name in ["mcd1", "mcd2", "mcd3"]:
        dropout_rates = {"mcd1": 0.1, "mcd2": 0.2, "mcd3": 0.5}
        dropout_rate = dropout_rates[model_name]
        model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                               layers=layers, activation_fn=activation_fn,
                               dropout_rate=dropout_rate, regularization_pen=l2_reg)
    else:
        raise ValueError(f"Only implemented for MCD models. Requested: {model_name}")

    # Dummy forward pass and restore
    _ = model(tf.zeros([1] + list(X_train.shape[1:])))
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(str(models_dir / f"ckpt-{best_ep}"))

    # Warmup pass (TensorFlow compilation can take a while)
    print("  Warming up TF graph...")
    _ = compute_nondeterministic_survival_curve(
        model, X_train, X_test[:5], e_train, t_train, event_times,
        1, n_samples_test=1
    )

    results = []

    # ---- Run Inference loops ----
    for mc_samples in MC_SAMPLES_LIST:
        print(f"\n  Evaluating S = {mc_samples}...")
        
        # Timing ONLY the inference logic itself
        start_time = time.time()
        
        # Shape: (S, num_patients, num_times)
        raw_preds = compute_nondeterministic_survival_curve(
            model, X_train, X_test, e_train, t_train, event_times,
            n_samples_train, n_samples_test=mc_samples
        )
        
        # Mean across MC samples -> (num_patients, num_times)
        surv_preds_test = np.mean(raw_preds, axis=0)
        
        inference_time = time.time() - start_time
        time_per_patient_ms = (inference_time / len(X_test)) * 1000

        print(f"    Inference took {inference_time:.3f}s ({time_per_patient_ms:.2f} ms/patient)")

        surv_preds_test = pd.DataFrame(surv_preds_test, columns=event_times)

        # Evaluate resulting curve
        metrics = evaluate_survival_preds(
            surv_preds_test, y_test, t_test, e_test,
            t_train, e_train, event_times, event_times_pct,
            label=f"S={mc_samples}"
        )

        results.append({
            "DatasetName": dataset_name,
            "ModelName": model_name,
            "LossConfig": loss_config,
            "Seed": seed,
            "MC_Samples": mc_samples,
            "InferenceTime_sec": inference_time,
            "TimePerPatient_ms": time_per_patient_ms,
            **metrics
        })

    # ---- Save Results ----
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "mc_efficiency_metrics.csv", index=False)
    print(f"\n  Saved {len(results)} rows to {out_dir / 'mc_efficiency_metrics.csv'}")
