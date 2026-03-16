import tensorflow as tf
from typing import Optional, Sequence
import tensorflow_probability as tfp
import numpy as np
import torch

class CoxPHLossGaussian(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, y_true: Sequence[tf.Tensor], y_pred: tf.Tensor) -> tf.Tensor:
        '''
        runs = 100
        logits_cpd = tf.zeros((runs, y_pred.shape[0]), dtype=np.float32)
        output_list = []
        tensor_shape = logits_cpd.get_shape()
        for i in range(tensor_shape[0]):
            output_list.append(tf.reshape(y_pred.sample(), y_pred.shape[0]))
        logits_cpd = tf.stack(output_list)
        '''
        #log_dist_var = -tf.math.log(tf.math.reduce_std(logits_cpd, axis=0, keepdims=True) ** 2)
        #return coxloss(y_true, y_pred) + tf.reduce_mean(log_dist_var)
        
        y_var = tf.transpose(tf.math.reduce_variance(y_pred, axis=0, keepdims=True))
        y_pred = tf.transpose(tf.reduce_mean(y_pred, axis=0, keepdims=True))
        
        # Perform Cox loss
        event, riskset = y_true
        
        pred_shape = y_pred.shape
        if pred_shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "be 2." % pred_shape.ndims)

        if pred_shape[1] is None:
            raise ValueError("Last dimension of predictions must be known.")

        if pred_shape[1] != 1:
            raise ValueError("Dimension mismatch: Last dimension of predictions "
                             "(received %s) must be 1." % pred_shape[1])

        if event.shape.ndims != pred_shape.ndims:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "equal ranvk of event (received %s)" % (
                pred_shape.ndims, event.shape.ndims))

        if riskset.shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of riskset (received %s) should "
                             "be 2." % riskset.shape.ndims)

        event = tf.cast(event, y_pred.dtype)
        riskset = tf.cast(riskset, tf.bool)
        y_pred = safe_normalize(y_pred)
        
        with tf.name_scope("assertions"):
            assertions = (
                tf.debugging.assert_less_equal(event, 1.),
                tf.debugging.assert_greater_equal(event, 0.),
                tf.debugging.assert_type(riskset, tf.bool)
            )

        # move batch dimension to the end so predictions get broadcast
        # row-wise when multiplying by riskset
        pred_t = tf.transpose(y_pred)

        # compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        assert rr.shape.as_list() == y_pred.shape.as_list()
        
        losses = (tf.math.multiply(event, (rr - y_pred)/(0.5*(np.exp(-np.log(y_var))))))
        variances = tf.math.multiply(event, 0.5*np.log(y_var))
        
        return losses + variances

class CoxPHLoss(tf.keras.losses.Loss):
    """Negative partial log-likelihood of Cox's proportional hazards model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self,
             y_true: Sequence[tf.Tensor],
             y_pred: tf.Tensor) -> tf.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_true : list|tuple of tf.Tensor
            The first element holds a binary vector where 1
            indicates an event 0 censoring.
            The second element holds the riskset, a
            boolean matrix where the `i`-th row denotes the
            risk set of the `i`-th instance, i.e. the indices `j`
            for which the observer time `y_j >= y_i`.
            Both must be rank 2 tensors.
        y_pred : tf.Tensor
            The predicted outputs. Must be a rank 2 tensor.

        Returns
        -------
        loss : tf.Tensor
            Loss for each instance in the batch.
        """
        event, riskset = y_true
        predictions = y_pred
        
        pred_shape = predictions.shape
        if pred_shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "be 2." % pred_shape.ndims)

        if pred_shape[1] is None:
            raise ValueError("Last dimension of predictions must be known.")

        if pred_shape[1] != 1:
            raise ValueError("Dimension mismatch: Last dimension of predictions "
                             "(received %s) must be 1." % pred_shape[1])

        if event.shape.ndims != pred_shape.ndims:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "equal ranvk of event (received %s)" % (
                pred_shape.ndims, event.shape.ndims))

        if riskset.shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of riskset (received %s) should "
                             "be 2." % riskset.shape.ndims)

        event = tf.cast(event, predictions.dtype)
        riskset = tf.cast(riskset, tf.bool)
        predictions = safe_normalize(predictions)

        with tf.name_scope("assertions"):
            assertions = (
                tf.debugging.assert_less_equal(event, 1.),
                tf.debugging.assert_greater_equal(event, 0.),
                tf.debugging.assert_type(riskset, tf.bool)
            )

        # move batch dimension to the end so predictions get broadcast
        # row-wise when multiplying by riskset
        pred_t = tf.transpose(predictions)

        # compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        assert rr.shape.as_list() == predictions.shape.as_list()

        losses = tf.math.multiply(event, rr - predictions)

        return losses

def safe_normalize(x: tf.Tensor) -> tf.Tensor:
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm

def logsumexp_masked(risk_scores: tf.Tensor,
                     mask: tf.Tensor,
                     axis: int = 0,
                     keepdims: Optional[bool] = None) -> tf.Tensor:
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    risk_scores.shape.assert_same_rank(mask.shape)

    with tf.name_scope("logsumexp_masked"):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)

        # for numerical stability, substract the maximum value
        # before taking the exponential
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax

        exp_masked = tf.math.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        output = amax + tf.math.log(exp_sum)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
    return output

def masked_logsumexp(
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1
) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp((x - max_val.unsqueeze(dim)) * mask) * mask,
                  dim=dim)) + max_val

def mtlr_nll(
        logits: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        C1: float,
        average: bool = False
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.
    average
        Whether to compute the average log likelihood instead of sum
        (useful for minibatch training).

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()

    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)

    # L2 regularization
    for k, v in model.named_parameters():
        if "mtlr_weight" in k:
            nll_total += C1/2 * torch.sum(v**2)

    return nll_total

def cox_nll(
        risk_pred: torch.Tensor,
        true_times: torch.Tensor,
        true_indicator: torch.Tensor,
        model: torch.nn.Module,
        C1: float
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    risk_pred : torch.Tensor, shape (num_samples, )
        Risk prediction from Cox-based model. It means the relative hazard ratio: \beta * x.
    true_times : torch.Tensor, shape (num_samples, )
        Tensor with the censor/event time.
    true_indicator : torch.Tensor, shape (num_samples, )
        Tensor with the censor indicator.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    eps = 1e-20
    risk_pred = risk_pred.reshape(-1, 1)
    true_times = true_times.reshape(-1, 1)
    true_indicator = true_indicator.reshape(-1, 1)
    mask = torch.ones(true_times.shape[0], true_times.shape[0]).to(true_times.device)
    mask[(true_times.T - true_times) > 0] = 0
    max_risk = risk_pred.max()
    log_loss = torch.exp(risk_pred - max_risk) * mask
    log_loss = torch.sum(log_loss, dim=0)
    log_loss = torch.log(log_loss + eps).reshape(-1, 1) + max_risk
    # Sometimes in the batch we got all censoring data, so the denominator gets 0 and throw nan.
    # Solution: Consider increase the batch size. Afterall the nll should performed on the whole dataset.
    # Based on equation 2&3 in https://arxiv.org/pdf/1606.00931.pdf
    neg_log_loss = -torch.sum((risk_pred - log_loss) * true_indicator) / torch.sum(true_indicator)

    # L2 regularization
    for k, v in model.named_parameters():
        if "weight" in k:
            neg_log_loss += C1/2 * torch.norm(v, p=2)

    return neg_log_loss


# ---------------------------------------------------------------------------
# Calibration-Aware Loss Functions (TensorFlow)
# ---------------------------------------------------------------------------

def breslow_survival_tf(logits, event, time, time_grid):
    """Differentiable Breslow estimator: logits → S(t|x) at each grid point.

    Parameters
    ----------
    logits : tf.Tensor, shape (batch, 1)
        Predicted log-hazard ratio from the model.
    event : tf.Tensor, shape (batch, 1) or (batch,)
        Binary event indicator (1 = event, 0 = censored).
    time : tf.Tensor, shape (batch,)
        Observed time for each sample.
    time_grid : tf.Tensor, shape (K,)
        Time points at which to evaluate the survival function.

    Returns
    -------
    surv : tf.Tensor, shape (batch, K)
        Predicted S(t_k | x_i) for each sample i and grid point k.
    """
    logits_flat = tf.squeeze(logits, axis=-1)          # (batch,)
    event_flat = tf.cast(tf.reshape(event, [-1]), tf.float32)
    time_flat = tf.cast(tf.reshape(time, [-1]), tf.float32)

    # Sort by ascending time
    order = tf.argsort(time_flat)
    sorted_time = tf.gather(time_flat, order)
    sorted_event = tf.gather(event_flat, order)
    sorted_logits = tf.gather(logits_flat, order)

    # Risk scores
    risk = tf.exp(sorted_logits)                       # (batch,)

    # Risk-set sums (cumulative from the end = sum of risk[j] for j>=i)
    risk_rev = tf.reverse(risk, axis=[0])
    risk_set_sum = tf.reverse(tf.cumsum(risk_rev), axis=[0])  # (batch,)
    risk_set_sum = tf.maximum(risk_set_sum, 1e-10)

    # Baseline hazard increments: d_i / R_i only for events
    hazard_inc = sorted_event / risk_set_sum           # (batch,)

    # Cumulative baseline hazard at each sorted observation
    cum_hazard = tf.cumsum(hazard_inc)                 # (batch,)

    # Interpolate: for each t in time_grid find H_0(t)
    grid_idx = tf.searchsorted(sorted_time, time_grid, side='right')
    grid_idx = tf.clip_by_value(grid_idx, 1, tf.shape(cum_hazard)[0]) - 1
    H0_grid = tf.gather(cum_hazard, grid_idx)          # (K,)

    # Survival: S(t_k | x_i) = exp( -H_0(t_k) * exp(logit_i) )
    all_risk = tf.exp(logits_flat)                     # (batch,)
    surv = tf.exp(
        -tf.expand_dims(all_risk, 1) * tf.expand_dims(H0_grid, 0)
    )                                                  # (batch, K)

    return surv


class CRPSLoss(tf.keras.losses.Loss):
    """Right-censored CRPS as a differentiable training loss.

    CRPS(S, t_obs, δ) = ∫₀^{t_obs} (1-S(u))² du + δ · ∫_{t_obs}^∞ S(u)² du

    Uses trapezoidal integration over a fixed time grid.
    y_true must be [event, riskset, time] (3-element list).
    """

    def __init__(self, time_grid_np, **kwargs):
        """
        Parameters
        ----------
        time_grid_np : np.ndarray, shape (K,)
            Sorted array of time points covering the observation range.
            Typically np.linspace(0, t_max, 100) from training data.
        """
        super().__init__(**kwargs)
        self.time_grid = tf.constant(time_grid_np, dtype=tf.float32)
        # Precompute Δt for trapezoidal rule
        dt = np.diff(time_grid_np, prepend=0.0)
        self.dt = tf.constant(dt, dtype=tf.float32)  # (K,)

    def call(self, y_true, y_pred):
        event = y_true[0]          # (batch, 1) or (batch,)
        time = y_true[2]           # (batch,)
        event_f = tf.cast(tf.reshape(event, [-1]), tf.float32)   # (batch,)
        time_f = tf.cast(tf.reshape(time, [-1]), tf.float32)

        # Compute survival curves at grid points
        surv = breslow_survival_tf(y_pred, event, time, self.time_grid)  # (batch, K)

        # Masks: before_obs[i,k] = 1(grid_k < t_obs_i)
        #        after_obs[i,k]  = 1(grid_k >= t_obs_i)
        before_obs = tf.cast(
            tf.expand_dims(self.time_grid, 0) < tf.expand_dims(time_f, 1), tf.float32
        )  # (batch, K)
        after_obs = 1.0 - before_obs

        # CRPS integrand
        # Before t_obs: (1 - S(t))²
        term1 = (1.0 - surv) ** 2 * before_obs * self.dt        # (batch, K)
        # After t_obs (uncensored only): S(t)²
        term2 = surv ** 2 * after_obs * self.dt                 # (batch, K)
        term2 = term2 * tf.expand_dims(event_f, 1)              # mask by event

        crps_per_sample = tf.reduce_sum(term1 + term2, axis=1)  # (batch,)
        return tf.reduce_mean(crps_per_sample)


class BrierScoreLoss(tf.keras.losses.Loss):
    """IPCW-weighted Integrated Brier Score as a differentiable training loss.

    BS(t) = (1/N) Σ [ 1(T≤t,δ=1)·S(t|x)²/G(T) + 1(T>t)·(1-S(t|x))²/G(t) ]
    IBS = mean over time grid

    y_true must be [event, riskset, time] (3-element list).
    """

    def __init__(self, time_grid_np, km_times_np, km_surv_np, **kwargs):
        """
        Parameters
        ----------
        time_grid_np : np.ndarray, shape (K,)
            Sorted time points at which to evaluate the Brier Score.
        km_times_np : np.ndarray
            Unique times of the KM estimate for the **censoring** distribution.
        km_surv_np : np.ndarray
            Corresponding KM survival probabilities G(t) = P(C > t).
        """
        super().__init__(**kwargs)
        self.time_grid = tf.constant(time_grid_np, dtype=tf.float32)
        self.km_times = tf.constant(km_times_np, dtype=tf.float32)
        self.km_surv = tf.constant(km_surv_np, dtype=tf.float32)

    def _G(self, t):
        """Censoring survival probability G(t) = P(C > t)."""
        idx = tf.searchsorted(self.km_times, t, side='right')
        idx = tf.clip_by_value(idx, 1, tf.shape(self.km_surv)[0]) - 1
        return tf.maximum(tf.gather(self.km_surv, idx), 1e-8)

    def call(self, y_true, y_pred):
        event = y_true[0]
        time = y_true[2]
        event_f = tf.cast(tf.reshape(event, [-1]), tf.float32)
        time_f = tf.cast(tf.reshape(time, [-1]), tf.float32)

        surv = breslow_survival_tf(y_pred, event, time, self.time_grid)  # (batch, K)

        # G(T_i) for each sample
        G_Ti = self._G(time_f)                       # (batch,)
        # G(t_k) for each grid point
        G_tk = self._G(self.time_grid)                # (K,)

        # Masks
        died_before = tf.cast(
            tf.expand_dims(time_f, 1) <= tf.expand_dims(self.time_grid, 0), tf.float32
        )  # (batch, K): 1 if T_i <= t_k
        alive = 1.0 - died_before

        # Case 1: died before t_k (uncensored only)
        case1 = (
            tf.expand_dims(event_f, 1) * died_before * surv ** 2
            / tf.expand_dims(G_Ti, 1)
        )  # (batch, K)

        # Case 2: alive at t_k
        case2 = alive * (1.0 - surv) ** 2 / tf.expand_dims(G_tk, 0)

        bs_grid = tf.reduce_mean(case1 + case2, axis=0)  # (K,)
        return tf.reduce_mean(bs_grid)


class MarginalCalibrationLoss(tf.keras.losses.Loss):
    """Marginal calibration: penalizes divergence of batch-mean survival
    from the Kaplan-Meier estimate.

    L = (1/K) Σ_k (S_model_avg(t_k) - S_KM(t_k))²

    y_true must be [event, riskset, time] (3-element list).
    """

    def __init__(self, time_grid_np, km_surv_at_grid_np, **kwargs):
        """
        Parameters
        ----------
        time_grid_np : np.ndarray, shape (K,)
            Time points for the grid.
        km_surv_at_grid_np : np.ndarray, shape (K,)
            KM survival probabilities at each grid point.
        """
        super().__init__(**kwargs)
        self.time_grid = tf.constant(time_grid_np, dtype=tf.float32)
        self.km_surv = tf.constant(km_surv_at_grid_np, dtype=tf.float32)

    def call(self, y_true, y_pred):
        event = y_true[0]
        time = y_true[2]

        surv = breslow_survival_tf(y_pred, event, time, self.time_grid)  # (batch, K)
        surv_mean = tf.reduce_mean(surv, axis=0)  # (K,)

        return tf.reduce_mean((surv_mean - self.km_surv) ** 2)


class JointCoxCalibrationLoss(tf.keras.losses.Loss):
    """Combined loss: (1-λ)·CoxPH + λ·CalibrationLoss [+ μ·MarginalLoss].

    y_true must be [event, riskset, time] (3-element list).
    """

    def __init__(self, calibration_loss, lam=0.3, marginal_loss=None, mu=0.0,
                 **kwargs):
        """
        Parameters
        ----------
        calibration_loss : CRPSLoss or BrierScoreLoss
            The calibration loss component.
        lam : float
            Weight for calibration loss (0 = pure Cox, 1 = pure calibration).
        marginal_loss : MarginalCalibrationLoss or None
            Optional marginal calibration regularizer.
        mu : float
            Weight for marginal loss.
        """
        super().__init__(**kwargs)
        self.cox_loss = CoxPHLoss()
        self.cal_loss = calibration_loss
        self.lam = lam
        self.marginal_loss = marginal_loss
        self.mu = mu

    def call(self, y_true, y_pred):
        # Cox uses only [event, riskset]
        cox = self.cox_loss(y_true=[y_true[0], y_true[1]], y_pred=y_pred)
        cal = self.cal_loss(y_true=y_true, y_pred=y_pred)

        total = (1.0 - self.lam - self.mu) * tf.reduce_mean(cox) + self.lam * cal

        if self.marginal_loss is not None and self.mu > 0:
            marg = self.marginal_loss(y_true=y_true, y_pred=y_pred)
            total = total + self.mu * marg

        return total