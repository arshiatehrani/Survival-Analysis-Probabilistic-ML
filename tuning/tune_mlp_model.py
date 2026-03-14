"""
tune_mlp_model.py
====================================
Bayesian optimization for MLP/BNN hyperparameters (paper: 10 iterations per dataset).
Uses wandb sweeps (method: bayes) to maximize validation concordance index.
Config is shared by MLP, VI, MCD×3, SNGP for fair comparison.
"""

import numpy as np
import os
import tensorflow as tf
import yaml
from pathlib import Path
from tools.baysurv_builder import make_mlp_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from tools import baysurv_trainer, data_loader
from utility.tuning import get_mlp_sweep_config
import argparse
from tools import data_loader
import pandas as pd
from utility.training import split_time_event
from utility.survival import calculate_event_times, compute_deterministic_survival_curve
import config as cfg
from utility.training import make_stratified_split
from utility.survival import convert_to_structured
from utility.training import scale_data
from pycox.evaluation import EvalSurv
import paths as pt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["WANDB_SILENT"] = "true"
import wandb

PROJECT_NAME = "baysurv_bo_mlp"

# Track best config across sweep runs (used when --save-config)
best_ci = -1.0
best_config = None


def _config_to_yaml_format(c):
    """Convert sweep config dict to YAML format expected by train_bnn_models."""
    wd = c.get('weight_decay')
    return {
        "network_layers": c.get('network_layers', [32]),
        "activiation_fn": c.get('activation_fn', 'relu'),
        "dropout_rate": c.get('dropout', 0.25),
        "l2_reg": c.get('l2_reg', 0.001),
        "batch_size": c.get('batch_size', 32),
        "early_stop": c.get('early_stop', True),
        "patience": c.get('patience', 5),
        "n_samples_train": c.get('n_samples_train', 10),
        "n_samples_valid": c.get('n_samples_valid', 10),
        "n_samples_test": c.get('n_samples_test', 100),
        "optimizer": {
            "class_name": "Adam",
            "config": {
                "learning_rate": float(c.get('learning_rate', 0.001)),
                "decay": float(wd) if wd is not None else 0.001,
            }
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name (SUPPORT, SEER, METABRIC, MIMIC, etc.)")
    parser.add_argument('--iterations', type=int, default=10,
                        help="Bayesian optimization iterations (paper: 10)")
    parser.add_argument('--save-config', action='store_true',
                        help="Save best config to configs/mlp/{dataset}.yaml for training")
    args = parser.parse_args()
    global dataset_name, best_ci, best_config
    dataset_name = args.dataset
    best_ci = -1.0
    best_config = None

    sweep_config = get_mlp_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, train_model, count=args.iterations)

    if args.save_config and best_config is not None:
        out_path = Path(pt.MLP_CONFIGS_DIR) / f"{dataset_name.lower()}.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_cfg = _config_to_yaml_format(best_config)
        with open(out_path, 'w') as f:
            yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n[Bayesian optimization] Best config saved to {out_path}")
        print("[Bayesian optimization] Hyperparameters chosen:")
        for k, v in yaml_cfg.items():
            if k != "optimizer":
                print(f"  {k}: {v}")
            else:
                print(f"  optimizer: {v}")
    elif args.save_config and best_config is None:
        print("\n[Bayesian optimization] No valid runs; config not saved.")

def train_model():
    global best_ci, best_config
    config_defaults = cfg.MLP_DEFAULT_PARAMS

    # Initialize a new wandb run
    wandb.init(config=config_defaults, group=dataset_name)
    config = wandb.config
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    early_stop = config['early_stop']
    patience = config['patience']
    l2_reg = config['l2_reg']
    n_samples_train = config['n_samples_train']
    n_samples_valid = config['n_samples_valid']
    n_samples_test = config['n_samples_test']

    # Load data
    if dataset_name == "SUPPORT":
        dl = data_loader.SupportDataLoader().load_data()
    elif dataset_name == "GBSG2":
        dl = data_loader.GbsgDataLoader().load_data()
    elif dataset_name == "WHAS500":
        dl = data_loader.WhasDataLoader().load_data()
    elif dataset_name == "FLCHAIN":
        dl = data_loader.FlchainDataLoader().load_data()
    elif dataset_name == "METABRIC":
        dl = data_loader.MetabricDataLoader().load_data()
    elif dataset_name == "SEER":
        dl = data_loader.SeerDataLoader().load_data()
    elif dataset_name == "MIMIC":
        dl = data_loader.MimicDataLoader().load_data()
    else:
        raise ValueError("Dataset not found")

    num_features, cat_features = dl.get_features()
    df = dl.get_data()
    
    # Split data
    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2, random_state=0)
    X_train = df_train[cat_features+num_features]
    X_valid = df_valid[cat_features+num_features]
    X_test = df_test[cat_features+num_features]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])
    
    # Scale data
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
    
    # Convert to array
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)

    # Make time/event split
    t_train, e_train = split_time_event(y_train)
    t_valid, e_valid = split_time_event(y_valid)

    # Calculate event times
    event_times = calculate_event_times(t_train, e_train)
    
    # Make datasets
    train_ds = InputFunction(X_train, t_train, e_train, batch_size=batch_size,
                             drop_last=True, shuffle=True)()
    valid_ds = InputFunction(X_valid, t_valid, e_valid, batch_size=batch_size)()
    
    model_name = "MLP"
    model = make_mlp_model(input_shape=X_train.shape[1:],
                           output_dim=1,
                           layers=config['network_layers'],
                           activation_fn=config['activation_fn'],
                           dropout_rate=config['dropout'],
                           regularization_pen=l2_reg)
    
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate,
                                            weight_decay=wandb.config.weight_decay)
    
    # Train model
    trainer = baysurv_trainer.Trainer(model=model,
                                      model_name=model_name,
                                      train_dataset=train_ds,
                                      valid_dataset=valid_ds,
                                      test_dataset=None,
                                      optimizer=optimizer,
                                      loss_function=CoxPHLoss(),
                                      num_epochs=num_epochs,
                                      early_stop=early_stop,
                                      patience=patience,
                                      n_samples_train=n_samples_train,
                                      n_samples_valid=n_samples_valid,
                                      n_samples_test=n_samples_test)
    trainer.train_and_evaluate()

    # Compute survival function
    surv_preds = pd.DataFrame(compute_deterministic_survival_curve(
        model, X_train, X_valid, e_train, t_train, event_times, model_name), columns=event_times)
    
    # Compute CI
    try:
        ev = EvalSurv(surv_preds.T, t_valid, e_valid, censor_surv="km")
        ci = ev.concordance_td()
    except Exception:
        ci = np.nan

    # Log to wandb
    wandb.log({"val_ci": ci})

    # Track best config for --save-config
    if not np.isnan(ci) and ci > best_ci:
        best_ci = float(ci)
        best_config = dict(wandb.config)

if __name__ == "__main__":
    main()