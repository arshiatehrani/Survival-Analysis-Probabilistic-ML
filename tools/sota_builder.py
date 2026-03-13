import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from utility.bnn_isd_models import BayesCox
from utility.bnn_isd_models import BayesMtlr
from pycox.models import DeepHitSingle, CoxPH, LogisticHazard
import torchtuples as tt
import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def make_baycox_model(num_features, config):
    return BayesCox(in_features=num_features, config=config)
    
def make_baymtlr_model(num_features, time_bins, config):
    num_time_bins = len(time_bins)
    return BayesMtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)

def make_cox_model(config):
    n_iter = config['n_iter']
    tol = config['tol']
    return CoxPHSurvivalAnalysis(alpha=0.0001, n_iter=n_iter, tol=tol)

def make_rsf_model(config):
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    min_samples_split = config['min_samples_split']
    min_samples_leaf =  config['min_samples_leaf']
    max_features = config['max_features']
    return RandomSurvivalForest(random_state=0,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features)

def make_coxnet_model(config):
    l1_ratio = config['l1_ratio']
    alpha_min_ratio = config['alpha_min_ratio']
    n_alphas = config['n_alphas']
    normalize = config['normalize']
    tol = config['tol']
    max_iter = config['max_iter']
    return CoxnetSurvivalAnalysis(fit_baseline_model=True,
                                  l1_ratio=l1_ratio,
                                  alpha_min_ratio=alpha_min_ratio,
                                  n_alphas=n_alphas,
                                  normalize=normalize,
                                  tol=tol,
                                  max_iter=max_iter)
    
def make_coxboost_model(config):
    n_estimators = config['n_estimators']
    learning_rate = config['learning_rate']
    max_depth = config['max_depth']
    loss = config['loss']
    min_samples_split = config['min_samples_split']
    min_samples_leaf = config['min_samples_leaf']
    max_features = config['max_features']
    dropout_rate = config['dropout_rate']
    subsample = config['subsample']
    return GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            max_depth=max_depth,
                                            loss=loss,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            dropout_rate=dropout_rate,
                                            subsample=subsample,
                                            random_state=0)

def _make_pycox_net(in_features, layers, out_features, dropout=0.1):
    """Build a simple MLP with ReLU + BatchNorm + Dropout for pycox models."""
    modules = []
    prev = in_features
    for units in layers:
        modules.append(torch.nn.Linear(prev, units))
        modules.append(torch.nn.ReLU())
        modules.append(torch.nn.BatchNorm1d(units))
        modules.append(torch.nn.Dropout(dropout))
        prev = units
    modules.append(torch.nn.Linear(prev, out_features))
    return torch.nn.Sequential(*modules)


class _PyCoxWrapper:
    """Base class providing auton-survival-compatible .fit()/.predict_survival() API."""

    @staticmethod
    def _parse_targets(y_df_or_t, e=None):
        if isinstance(y_df_or_t, pd.DataFrame):
            return (y_df_or_t['time'].values.astype('float64'),
                    y_df_or_t['event'].values.astype('float64'))
        return (np.asarray(y_df_or_t, dtype='float64'),
                np.asarray(e, dtype='float64'))

    @staticmethod
    def _interpolate_surv(surv_df, query_times):
        """Interpolate survival DataFrame (index=model_times, cols=samples) to query_times.
        Returns ndarray of shape (n_samples, len(query_times))."""
        model_times = surv_df.index.values.astype('float64')
        query_times = np.asarray(query_times, dtype='float64')
        result = np.empty((surv_df.shape[1], len(query_times)))
        for i in range(surv_df.shape[1]):
            vals = surv_df.iloc[:, i].values
            fn = interp1d(model_times, vals, kind='previous',
                          bounds_error=False,
                          fill_value=(1.0, float(vals[-1])))
            result[i] = fn(query_times)
        return result


class DeepHitDSMWrapper(_PyCoxWrapper):
    """DeepHitSingle-based replacement for auton-survival DSM."""

    def __init__(self, in_features, config, num_durations=100, device=None):
        self.in_features = in_features
        self.layers = config['network_layers']
        self.lr = config.get('learning_rate', 0.01)
        self.n_epochs = config.get('n_iter', 200)
        self.num_durations = num_durations
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.labtrans = None

    def fit(self, X_train, y_df_or_t, e=None, val_data=None, **kwargs):
        t_train, e_train = self._parse_targets(y_df_or_t, e)

        self.labtrans = DeepHitSingle.label_transform(self.num_durations)
        y_train_trans = self.labtrans.fit_transform(t_train, e_train)

        net = _make_pycox_net(self.in_features, self.layers, self.labtrans.out_features)
        net.to(self.device)
        self.model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1,
                                   duration_index=self.labtrans.cuts)
        self.model.optimizer.set_lr(self.lr)

        val = None
        if val_data is not None:
            val = self._prepare_val(val_data)

        x_train = np.asarray(X_train).astype('float32')
        self.model.fit(x_train, y_train_trans, batch_size=32,
                       epochs=self.n_epochs, val_data=val, verbose=False)
        return self

    def predict_survival(self, X_test, times=None, t=None):
        query_times = times if times is not None else t
        x_test = np.asarray(X_test).astype('float32')
        surv_df = self.model.predict_surv_df(x_test)
        if query_times is not None:
            return self._interpolate_surv(surv_df, query_times)
        return surv_df.T.values

    def _prepare_val(self, val_data):
        if len(val_data) == 2:
            X_val, y_val = val_data
            t_val, e_val = self._parse_targets(y_val)
        else:
            X_val, t_val, e_val = val_data[0], val_data[1], val_data[2]
            t_val = np.asarray(t_val, dtype='float64')
            e_val = np.asarray(e_val, dtype='float64')
        y_val_trans = self.labtrans.transform(t_val, e_val)
        return (np.asarray(X_val).astype('float32'), y_val_trans)


class CoxPHWrapper(_PyCoxWrapper):
    """pycox CoxPH wrapper replacing auton-survival DCPH."""

    def __init__(self, in_features, config, device=None):
        self.in_features = in_features
        self.layers = config['network_layers']
        self.lr = config.get('learning_rate', 0.001)
        self.n_epochs = config.get('n_iter', config.get('iters', 500))
        self.batch_size = config.get('batch_size', 32)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X_train, y_df_or_t, e=None, val_data=None, **kwargs):
        t_train, e_train = self._parse_targets(y_df_or_t, e)

        net = _make_pycox_net(self.in_features, self.layers, out_features=1)
        net.to(self.device)
        self.model = CoxPH(net, tt.optim.Adam)
        self.model.optimizer.set_lr(self.lr)

        y_train_pycox = (t_train.astype('float32'), e_train.astype('float32'))

        val = None
        if val_data is not None:
            val = self._prepare_val(val_data)

        x_train = np.asarray(X_train).astype('float32')
        self.model.fit(x_train, y_train_pycox, batch_size=self.batch_size,
                       epochs=self.n_epochs, val_data=val, verbose=False)
        self.model.compute_baseline_hazards()
        return self

    def predict_survival(self, X_test, times=None, t=None):
        query_times = times if times is not None else t
        x_test = np.asarray(X_test).astype('float32')
        surv_df = self.model.predict_surv_df(x_test)
        if query_times is not None:
            return self._interpolate_surv(surv_df, query_times)
        return surv_df.T.values

    def _prepare_val(self, val_data):
        if len(val_data) == 2:
            X_val, y_val = val_data
            t_val, e_val = self._parse_targets(y_val)
        else:
            X_val, t_val, e_val = val_data[0], val_data[1], val_data[2]
            t_val = np.asarray(t_val, dtype='float64')
            e_val = np.asarray(e_val, dtype='float64')
        y_val_pycox = (t_val.astype('float32'), e_val.astype('float32'))
        return (np.asarray(X_val).astype('float32'), y_val_pycox)


class LogisticHazardDCMWrapper(_PyCoxWrapper):
    """LogisticHazard-based replacement for auton-survival DCM."""

    def __init__(self, in_features, config, num_durations=100, device=None):
        self.in_features = in_features
        self.layers = config['network_layers']
        self.lr = config.get('learning_rate', 0.001)
        self.n_epochs = config.get('n_iter', 1000)
        self.num_durations = num_durations
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.labtrans = None

    def fit(self, X_train, y_df_or_t, e=None, val_data=None, **kwargs):
        t_train, e_train = self._parse_targets(y_df_or_t, e)

        self.labtrans = LogisticHazard.label_transform(self.num_durations)
        y_train_trans = self.labtrans.fit_transform(t_train, e_train)

        net = _make_pycox_net(self.in_features, self.layers, self.labtrans.out_features)
        net.to(self.device)
        self.model = LogisticHazard(net, tt.optim.Adam,
                                    duration_index=self.labtrans.cuts)
        self.model.optimizer.set_lr(self.lr)

        val = None
        if val_data is not None:
            val = self._prepare_val(val_data)

        x_train = np.asarray(X_train).astype('float32')
        self.model.fit(x_train, y_train_trans, batch_size=32,
                       epochs=self.n_epochs, val_data=val, verbose=False)
        return self

    def predict_survival(self, X_test, times=None, t=None):
        query_times = times if times is not None else t
        x_test = np.asarray(X_test).astype('float32')
        surv_df = self.model.predict_surv_df(x_test)
        if query_times is not None:
            return self._interpolate_surv(surv_df, query_times)
        return surv_df.T.values

    def _prepare_val(self, val_data):
        if len(val_data) == 2:
            X_val, y_val = val_data
            t_val, e_val = self._parse_targets(y_val)
        else:
            X_val, t_val, e_val = val_data[0], val_data[1], val_data[2]
            t_val = np.asarray(t_val, dtype='float64')
            e_val = np.asarray(e_val, dtype='float64')
        y_val_trans = self.labtrans.transform(t_val, e_val)
        return (np.asarray(X_val).astype('float32'), y_val_trans)


def make_dsm_model(config, in_features, num_durations=100):
    return DeepHitDSMWrapper(in_features, config, num_durations)

def make_dcph_model(config, in_features):
    return CoxPHWrapper(in_features, config)

def make_dcm_model(config, in_features, num_durations=100):
    return LogisticHazardDCMWrapper(in_features, config, num_durations)