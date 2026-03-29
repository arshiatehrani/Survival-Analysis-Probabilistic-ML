"""Uncertainty-Aware Trainer (Self-Paced Curriculum Learning for Survival Analysis).

Extends the base Trainer with two optional mechanisms activated after a warmup phase:

1. **Soft weighting**: Per-sample loss is scaled by exp(-T * u_i) where u_i is the
   normalised MC-dropout variance. High-uncertainty samples contribute less gradient.
2. **Hard curriculum**: Only the lowest-uncertainty fraction of training samples is
   included each epoch. The fraction ramps linearly from `curriculum_start` to
   `curriculum_end` over the remaining epochs.

Both mechanisms require a stochastic model (MCD or VI) and use the model's own
MC variance to estimate per-sample uncertainty.

Weight alignment note: We use WeightedInputFunction to embed per-sample weights
directly into the tf.data pipeline labels dict ("sample_weights"). This ensures
correct alignment even after internal shuffling.
"""

import sys
import numpy as np
import tensorflow as tf
from tools.baysurv_trainer import Trainer
from utility.risk import InputFunction, WeightedInputFunction
from utility.loss import (CRPSLossPerSample, BrierScoreLossPerSample,
                          JointCoxCalibrationLossPerSample)


class UncertaintyTrainer(Trainer):
    """Trainer with self-paced curriculum learning driven by MC-dropout uncertainty.

    Parameters (beyond base Trainer)
    --------------------------------
    uncertainty_mode : str
        One of {"none", "soft", "curriculum", "both"}.
    warmup_epochs : int
        Epochs of standard training before uncertainty kicks in.
    mc_passes : int
        Number of stochastic forward passes for uncertainty estimation.
    temperature : float
        Exponential weighting strength: w_i = exp(-T * u_i).
    curriculum_start : float
        Fraction of training data to keep at first post-warmup epoch.
    curriculum_end : float
        Fraction at the final epoch (typically 1.0).
    X_train, t_train, e_train : np.ndarray
        Raw training arrays needed to rebuild InputFunction each epoch.
    batch_size : int
        Batch size for rebuilding the dataset.
    """

    def __init__(self, *, uncertainty_mode="none", warmup_epochs=2,
                 mc_passes=5, temperature=2.0, curriculum_start=0.55,
                 curriculum_end=1.0, X_train=None, t_train=None,
                 e_train=None, batch_size=32, **kwargs):
        super().__init__(**kwargs)

        self.unc_mode = uncertainty_mode
        self.warmup_epochs = warmup_epochs
        self.mc_passes = mc_passes
        self.temperature = temperature
        self.curriculum_start = curriculum_start
        self.curriculum_end = curriculum_end

        self.X_train_full = X_train
        self.t_train_full = t_train
        self.e_train_full = e_train
        self._batch_size = batch_size

        self.use_soft = uncertainty_mode in ("soft", "both")
        self.use_curriculum = uncertainty_mode in ("curriculum", "both")

        self._loss_needs_time = self._loss_needs_time or isinstance(
            self.loss_fn,
            (CRPSLossPerSample, BrierScoreLossPerSample, JointCoxCalibrationLossPerSample)
        )

        self._n_train = len(X_train) if X_train is not None else 0
        self._current_uncertainty = np.zeros(self._n_train, dtype=np.float32)

        self.uncertainty_history = []

    def _estimate_uncertainty(self):
        """Run mc_passes stochastic forward passes to get per-sample log-hazard variance."""
        n = self._n_train
        logits_all = np.zeros((self.mc_passes, n), dtype=np.float32)

        for p in range(self.mc_passes):
            preds = []
            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                x_batch = tf.constant(self.X_train_full[start:end], dtype=tf.float32)
                output = self.model(x_batch, training=True)

                if isinstance(output, (tuple, list)):
                    logit = output[0]
                elif hasattr(output, "sample"):
                    logit = tf.expand_dims(output.sample(), -1) if output.shape[-1] != 1 else output.sample()
                elif tf.is_tensor(output) and output.shape.rank == 2 and output.shape[-1] == 2:
                    logit = output[:, 0:1]
                else:
                    logit = output

                preds.append(tf.squeeze(logit, axis=-1).numpy())
            logits_all[p] = np.concatenate(preds)

        var_per_sample = logits_all.var(axis=0)

        vmin, vmax = var_per_sample.min(), var_per_sample.max()
        if vmax > vmin:
            normed = (var_per_sample - vmin) / (vmax - vmin + 1e-8)
        else:
            normed = np.zeros_like(var_per_sample)

        self._current_uncertainty = normed

    def _rebuild_train_dataset(self, epoch):
        """Subset data (curriculum) and/or attach weights (soft), rebuild tf.data.Dataset."""
        if self.use_curriculum:
            total_post_warmup = max(1, self.num_epochs - self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / total_post_warmup
            progress = min(1.0, max(0.0, progress))
            keep_frac = self.curriculum_start + (self.curriculum_end - self.curriculum_start) * progress
            n_keep = max(1, int(self._n_train * keep_frac))
            keep_idx = np.argsort(self._current_uncertainty)[:n_keep]
        else:
            keep_idx = np.arange(self._n_train)

        X_sub = self.X_train_full[keep_idx]
        t_sub = self.t_train_full[keep_idx]
        e_sub = self.e_train_full[keep_idx]

        if self.use_soft:
            unc_sub = self._current_uncertainty[keep_idx]
            weights = np.exp(-self.temperature * unc_sub).astype(np.float32)
            self.train_ds = WeightedInputFunction(
                X_sub, t_sub, e_sub, sample_weights=weights,
                batch_size=self._batch_size,
                drop_last=True, shuffle=True,
            )()
        else:
            self.train_ds = InputFunction(
                X_sub, t_sub, e_sub,
                batch_size=self._batch_size,
                drop_last=True, shuffle=True,
            )()

        return keep_idx

    def train_and_evaluate(self):
        stop_training = False
        for epoch in range(1, self.num_epochs + 1):
            if epoch > 0 and self.model_name == "sngp":
                self.model.layers[-1].reset_covariance_matrix()

            is_active = (self.unc_mode != "none" and epoch > self.warmup_epochs)

            if is_active:
                self._estimate_uncertainty()
                keep_idx = self._rebuild_train_dataset(epoch)
                unc_kept = self._current_uncertainty[keep_idx]
                hist_entry = {
                    "epoch": epoch,
                    "unc_mean": float(self._current_uncertainty.mean()),
                    "unc_std": float(self._current_uncertainty.std()),
                    "unc_max": float(self._current_uncertainty.max()),
                    "kept_frac": len(keep_idx) / self._n_train,
                    "kept_n": len(keep_idx),
                    "kept_unc_mean": float(unc_kept.mean()),
                    "phase": "active",
                }
                self.uncertainty_history.append(hist_entry)
            else:
                self.uncertainty_history.append({
                    "epoch": epoch, "unc_mean": 0.0, "unc_std": 0.0,
                    "unc_max": 0.0, "kept_frac": 1.0, "kept_n": self._n_train,
                    "kept_unc_mean": 0.0, "phase": "warmup",
                })

            if is_active and self.use_soft:
                self._train_weighted(epoch)
            else:
                self.train(epoch)

            if self.valid_ds is not None:
                stop_training = self.validate(epoch)
            if self.test_ds is not None:
                self.test()

            self._progress_uncertainty(epoch)

            if self.use_wandb:
                import wandb
                log_dict = {"epoch": epoch, "train_loss": self.train_loss[-1]}
                if self.train_total:
                    log_dict["train_total"] = self.train_total[-1]
                if self.train_variance:
                    log_dict["train_variance"] = self.train_variance[-1]
                if self.valid_loss:
                    log_dict["valid_loss"] = self.valid_loss[-1]
                h = self.uncertainty_history[-1]
                log_dict["unc_mean"] = h["unc_mean"]
                log_dict["kept_frac"] = h["kept_frac"]
                wandb.log(log_dict)

            if stop_training:
                self.cleanup()
                break
            self.cleanup()
        print()

    def _train_weighted(self, epoch):
        """Training step with per-sample uncertainty weighting via labels dict."""
        batch_variances, batch_total, batch_nll, batch_kl = [], [], [], []
        runs = self.n_samples_train

        for x, y in self.train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            weights = y["sample_weights"]  # (batch,) — aligned by WeightedInputFunction

            with tf.GradientTape() as tape:
                logits, batch_var = self._predict_logits_and_variance(x, training=True, runs=runs)
                if self._loss_needs_time:
                    nll = self.loss_fn(y_true=[y_event, y["label_riskset"], y["label_time"]], y_pred=logits)
                else:
                    nll = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)

                if nll.shape.rank == 0:
                    weighted_nll = nll
                else:
                    w = tf.reshape(weights[:tf.shape(nll)[0]], tf.shape(nll))
                    weighted_nll = tf.reduce_mean(w * nll)

                kl = self._regularization_term()
                loss = weighted_nll + kl
                self.train_loss_metric.update_state(loss)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            batch_variances.append(float(batch_var.numpy()))
            batch_total.append(float(loss.numpy()))
            batch_nll.append(float(weighted_nll.numpy()) if weighted_nll.shape.rank == 0 else float(nll.numpy()))
            batch_kl.append(float(kl.numpy()))

        epoch_loss = self.train_loss_metric.result()
        self.train_loss.append(float(epoch_loss))
        self.train_total.append(float(np.mean(batch_total)) if batch_total else 0.0)
        self.train_nll.append(float(np.mean(batch_nll)) if batch_nll else 0.0)
        self.train_kl.append(float(np.mean(batch_kl)) if batch_kl else 0.0)
        if batch_variances:
            self.train_variance.append(float(np.mean(batch_variances)))
        self.manager.save()

    def _progress_uncertainty(self, epoch):
        """Extended progress bar showing uncertainty stats."""
        pct = min(100, epoch * 100 // self.num_epochs)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        t_total = self.train_total[-1] if self.train_total else 0.0
        t_reg = self.train_kl[-1] if self.train_kl else 0.0
        t_nll = self.train_nll[-1] if self.train_nll else 0.0
        parts = f"Train: Total={t_total:.4f}, Reg={t_reg:.4f}, Loss={t_nll:.4f}"
        if self.train_variance:
            parts += f" var={self.train_variance[-1]:.4f}"
        if self.valid_total:
            parts += f"; Val: Total={self.valid_total[-1]:.4f}, Loss={self.valid_nll[-1]:.4f}"

        h = self.uncertainty_history[-1]
        if h["phase"] == "active":
            parts += f" | unc={h['unc_mean']:.3f} kept={h['kept_frac']:.2f}"

        msg = f"  [{bar}] {epoch}/{self.num_epochs} {parts}"
        is_tty = getattr(sys.stdout, "isatty", lambda: False)()
        if is_tty:
            sys.stdout.write("\r" + msg)
        else:
            print(msg)
        sys.stdout.flush()
