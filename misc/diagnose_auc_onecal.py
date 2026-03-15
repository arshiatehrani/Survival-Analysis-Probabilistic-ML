#!/usr/bin/env python3
"""
Diagnose why AUC@25/50/75 and 1-Cal@25/50/75 return N/A on SUPPORT (or any dataset).

Run: conda activate p && python misc/diagnose_auc_onecal.py --dataset SUPPORT

Checks:
1. Percentile times (t25, t50, t75) from train+test
2. For each t: test set binary status (event_time > t) after excluding early-censored
3. AUC requires both classes; reports if single-class
4. One-calibration: runs and reports any exception or NaN
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utility.training import get_data_loader, scale_data, split_time_event, make_stratified_split
from utility.survival import calculate_event_times, calculate_percentiles, convert_to_structured
from tools.Evaluations.AreaUnderCurve import auc as compute_auc
from tools.Evaluations.OneCalibration import one_calibration


def main():
    parser = argparse.ArgumentParser(description="Diagnose AUC/1-Cal N/A on a dataset")
    parser.add_argument("--dataset", default="SUPPORT", help="Dataset name (SUPPORT, SEER, METABRIC, MIMIC)")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"\n{'='*60}")
    print(f"Diagnosing AUC / 1-Cal on {dataset_name}")
    print(f"{'='*60}\n")

    # Load and split data (same as train_bnn_models)
    dl = get_data_loader(dataset_name).load_data()
    num_features, cat_features = dl.get_features()
    df = dl.get_data()

    df_train, df_valid, df_test = make_stratified_split(
        df, stratify_colname='both', frac_train=0.7, frac_valid=0.1, frac_test=0.2, random_state=0
    )
    X_train = df_train[cat_features + num_features]
    X_test = df_test[cat_features + num_features]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])

    t_train, e_train = split_time_event(y_train)
    t_test, e_test = split_time_event(y_test)
    t_train = np.asarray(t_train).ravel()
    t_test = np.asarray(t_test).ravel()
    e_train = np.asarray(e_train).ravel()
    e_test = np.asarray(e_test).ravel()

    # Percentiles (same as compute_extended_metrics)
    times_for_pct = np.concatenate([t_train, t_test])
    event_times_pct = calculate_percentiles(times_for_pct)

    print("1. PERCENTILE TIMES (from train+test)")
    print("-" * 40)
    for q, t0 in event_times_pct.items():
        print(f"   t{q}: {t0}")

    # Simulate AUC logic (from AreaUnderCurve.py)
    print("\n2. AUC DIAGNOSTIC (test set, at each percentile)")
    print("-" * 40)
    print("   AUC requires both classes (0 and 1) after excluding early-censored.\n")

    # Use dummy predictions (all 0.5) - we only care about binary_status
    dummy_preds = np.full(len(t_test), 0.5)

    for q, t0 in event_times_pct.items():
        exclude = np.logical_and(t_test < t0, e_test == 0)
        t_used = t_test[~exclude]
        pred_used = dummy_preds[~exclude]
        binary_status = (t_used > t0).astype(int)

        n_excluded = exclude.sum()
        n_kept = len(t_used)
        n_class0 = (binary_status == 0).sum()
        n_class1 = (binary_status == 1).sum()
        n_unique = len(np.unique(binary_status))

        status = "OK (both classes)" if n_unique >= 2 else "N/A (single class)"
        print(f"   @{q}: t0={t0}")
        print(f"         Excluded (censored, time<{t0}): {n_excluded}")
        print(f"         Kept: {n_kept} | class0 (event before t0): {n_class0} | class1 (event after t0): {n_class1}")
        print(f"         -> AUC: {status}")
        print()

    # One-calibration diagnostic
    print("3. ONE-CALIBRATION DIAGNOSTIC")
    print("-" * 40)
    print("   One-cal needs predictions at target time. Using dummy S(t)=0.5 -> P(event)=0.5.\n")

    for q, t0 in event_times_pct.items():
        # Predictions = survival prob at t0. 1 - S(t0) = P(T<=t0) = event prob
        # Dummy: assume everyone has S(t0)=0.5 -> event prob = 0.5
        dummy_surv_at_t0 = np.full(len(t_test), 0.5)
        predictions = 1 - dummy_surv_at_t0  # event probability

        try:
            p_val, obs, exp = one_calibration(
                predictions, t_test, e_test, target_time=t0, num_bins=10, method="DN"
            )
            if np.isnan(p_val):
                print(f"   @{q}: t0={t0} -> p_value=NaN (possible numerical issue)")
            else:
                print(f"   @{q}: t0={t0} -> p_value={p_val:.4f} OK")
        except Exception as ex:
            print(f"   @{q}: t0={t0} -> EXCEPTION: {ex}")

    # Check if evaluator would get different predictions
    print("\n4. ROOT CAUSE SUMMARY")
    print("-" * 40)
    single_class = []
    for q, t0 in event_times_pct.items():
        exclude = np.logical_and(t_test < t0, e_test == 0)
        t_used = t_test[~exclude]
        binary_status = (t_used > t0).astype(int)
        if len(np.unique(binary_status)) < 2:
            single_class.append(q)
    if single_class:
        print(f"   AUC N/A: percentiles {single_class} have single class in test set after exclusion.")
        print("   -> Try using different percentile times, or a larger test set.")
    else:
        print("   AUC: All percentiles have both classes. N/A may come from evaluator/predictions.")

    print("\n   For 1-Cal: If no exceptions above, N/A may come from:")
    print("   - evaluator.one_calibration() receiving invalid survival curves")
    print("   - NaN in predictions (e.g. extrapolation outside curve range)")
    print()


if __name__ == "__main__":
    main()
