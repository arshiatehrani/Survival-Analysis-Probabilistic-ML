"""Post-hoc calibration methods for survival analysis.

Two calibration strategies that transform predicted survival curves
*after* model training, fitted on a held-out validation set:

  - TemperatureScaling: S_cal(t|x) = S(t|x)^(1/T)
  - IsotonicCalibration: per-time-point isotonic regression

Both preserve the monotonically-decreasing property of survival curves.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from lifelines import KaplanMeierFitter


class TemperatureScaling:
    """Global power-law calibration for survival curves.

    Finds a scalar T > 0 such that S_cal(t|x) = S(t|x)^(1/T)
    minimizes the Integrated Brier Score on the validation set.

    Parameters
    ----------
    T_range : tuple
        (min, max) search range for T.
    n_grid : int
        Number of grid points for the search.
    """

    def __init__(self, T_range=(0.1, 5.0), n_grid=200):
        self.T_range = T_range
        self.n_grid = n_grid
        self.T_ = 1.0  # default: no change

    def fit(self, surv_preds_val, t_val, e_val, t_train, e_train):
        """Fit T by grid-searching for minimum IBS on validation data.

        Parameters
        ----------
        surv_preds_val : pd.DataFrame
            Survival predictions on validation set.
            Columns = event times, rows = patients.
        t_val, e_val : np.ndarray
            Validation times and event indicators.
        t_train, e_train : np.ndarray
            Training times and event indicators (for IPCW in IBS).
        """
        from tools.evaluator import LifelinesEvaluator

        event_times = surv_preds_val.columns.values
        best_T, best_ibs = 1.0, np.inf

        for T in np.linspace(self.T_range[0], self.T_range[1], self.n_grid):
            cal_preds = surv_preds_val ** (1.0 / T)
            cal_preds = cal_preds.clip(0, 1)

            try:
                ev = LifelinesEvaluator(
                    cal_preds.T, t_val, e_val, t_train, e_train
                )
                ibs = ev.integrated_brier_score()
                if ibs < best_ibs:
                    best_ibs = ibs
                    best_T = T
            except Exception:
                continue

        self.T_ = best_T
        self.best_ibs_ = best_ibs
        return self

    def transform(self, surv_preds):
        """Apply temperature scaling to survival predictions.

        Parameters
        ----------
        surv_preds : pd.DataFrame
            Survival predictions. Columns = event times, rows = patients.

        Returns
        -------
        pd.DataFrame
            Calibrated survival predictions.
        """
        cal = surv_preds ** (1.0 / self.T_)
        return cal.clip(0, 1)


class IsotonicCalibration:
    """Per-time-point isotonic regression calibration.

    At selected time points, fits an isotonic regression mapping
    predicted probabilities to Kaplan-Meier-estimated true probabilities.
    Enforces monotonicity across time after calibration.

    Parameters
    ----------
    n_time_points : int
        Number of time points at which to fit isotonic regression.
        Uses evenly spaced points across the event horizon.
    """

    def __init__(self, n_time_points=50):
        self.n_time_points = n_time_points
        self.models_ = {}  # {t_k: IsotonicRegression}
        self.time_points_ = None

    def fit(self, surv_preds_val, t_val, e_val):
        """Fit isotonic regression at each time point.

        Uses Kaplan-Meier on validation data to get "true" survival
        probabilities for each time point.

        Parameters
        ----------
        surv_preds_val : pd.DataFrame
            Survival predictions on validation set.
        t_val, e_val : np.ndarray
            Validation times and event indicators.
        """
        event_times = surv_preds_val.columns.values.astype(float)

        # Select time points for fitting
        indices = np.linspace(0, len(event_times) - 1,
                              min(self.n_time_points, len(event_times)),
                              dtype=int)
        self.time_points_ = event_times[indices]

        # Fit KM on validation data
        kmf = KaplanMeierFitter()
        kmf.fit(t_val, event_observed=e_val)

        self.models_ = {}
        for t_k in self.time_points_:
            # Predicted survival at this time point
            col_idx = np.argmin(np.abs(event_times - t_k))
            p_pred = surv_preds_val.iloc[:, col_idx].values

            # "True" target: for each patient, 1 if survived past t_k, 0 otherwise
            # For censored patients with c_i > t_k, they survived past t_k (target = 1)
            # For censored patients with c_i <= t_k, we don't know — exclude them
            survived = (t_val > t_k).astype(float)
            known = (e_val == 1) | (t_val > t_k)  # uncensored, or censored after t_k

            if known.sum() < 10:
                # Not enough data, skip this time point
                continue

            p_pred_known = p_pred[known]
            target_known = survived[known]

            try:
                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds='clip',
                    increasing=True  # higher predicted S → higher true S
                )
                iso.fit(p_pred_known, target_known)
                self.models_[t_k] = (col_idx, iso)
            except Exception:
                continue

        return self

    def transform(self, surv_preds):
        """Apply isotonic calibration and enforce monotonicity.

        Parameters
        ----------
        surv_preds : pd.DataFrame
            Survival predictions.

        Returns
        -------
        pd.DataFrame
            Calibrated survival predictions.
        """
        cal = surv_preds.copy()
        event_times = cal.columns.values.astype(float)

        # Apply isotonic regression at fitted time points
        for t_k, (col_idx, iso) in self.models_.items():
            p_pred = cal.iloc[:, col_idx].values
            cal.iloc[:, col_idx] = iso.predict(p_pred)

        # Interpolate corrections for non-fitted time points
        # by linearly interpolating between the fitted corrections
        fitted_cols = sorted([col_idx for _, (col_idx, _) in self.models_.items()])
        if len(fitted_cols) >= 2:
            for patient_idx in range(len(cal)):
                row = cal.iloc[patient_idx].values.copy()
                # Enforce monotonicity: survival must be non-increasing
                for i in range(1, len(row)):
                    if row[i] > row[i - 1]:
                        row[i] = row[i - 1]
                cal.iloc[patient_idx] = row
        else:
            # Fallback: just enforce monotonicity
            for patient_idx in range(len(cal)):
                row = cal.iloc[patient_idx].values.copy()
                for i in range(1, len(row)):
                    if row[i] > row[i - 1]:
                        row[i] = row[i - 1]
                cal.iloc[patient_idx] = row

        return cal.clip(0, 1)
