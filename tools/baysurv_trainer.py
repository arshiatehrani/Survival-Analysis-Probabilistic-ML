import sys
import tensorflow as tf
import numpy as np
import paths as pt
from utility.loss import CoxPHLoss, CoxPHLossGaussian

class Trainer:
    def __init__(self, model, model_name, train_dataset, valid_dataset,
                 test_dataset, optimizer, loss_function, num_epochs, early_stop,
                 patience, n_samples_train, n_samples_valid, n_samples_test,
                 use_wandb=False):
        self.num_epochs = num_epochs
        self.model = model
        self.model_name = model_name
        self.use_wandb = use_wandb

        self.train_ds = train_dataset
        self.valid_ds = valid_dataset
        self.test_ds = test_dataset

        self.optimizer = optimizer
        self.loss_fn = loss_function
        
        self.n_samples_train = n_samples_train
        self.n_samples_valid = n_samples_valid
        self.n_samples_test = n_samples_test

        # Metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.valid_loss_metric = tf.keras.metrics.Mean(name="valid_loss")
        self.test_loss_metric = tf.keras.metrics.Mean(name="test_loss")
        self.train_loss, self.valid_loss = list(), list()
        self.train_variance, self.valid_variance = list(), list()
        self.train_total, self.train_nll, self.train_kl = list(), list(), list()
        self.valid_total, self.valid_nll, self.valid_kl = list(), list(), list()
        self.test_variance, self.test_loss_scores = list(), list()
                
        self.early_stop = early_stop
        self.patience = patience
        
        self.best_valid_nll = np.inf
        self.best_ep = -1
        
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=f"{pt.MODELS_DIR}", max_to_keep=num_epochs)

    def _regularization_term(self):
        if not self.model.losses:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.add_n([tf.cast(loss, tf.float32) for loss in self.model.losses])

    def _predict_logits_and_variance(self, x, training, runs):
        output = self.model(x, training=training)

        if isinstance(output, (tuple, list)):
            logits = output[0]
            variance = tf.constant(0.0, dtype=tf.float32)
            if len(output) > 1 and tf.is_tensor(output[1]):
                covmat = output[1]
                if covmat.shape.rank is not None and covmat.shape.rank >= 2:
                    variance = tf.reduce_mean(tf.linalg.diag_part(covmat))
            return logits, variance

        if hasattr(output, "sample"):
            logits_samples = tf.stack([tf.reshape(output.sample(), (-1,)) for _ in range(runs)])
            logits_mean = tf.expand_dims(tf.reduce_mean(logits_samples, axis=0), axis=1)
            variance = tf.reduce_mean(tf.math.reduce_variance(logits_samples, axis=0, keepdims=True))
            return logits_mean, variance

        if tf.is_tensor(output) and output.shape.rank is not None and output.shape.rank == 2 and output.shape[-1] == 2:
            loc = output[:, 0:1]
            raw_scale = output[:, 1:2]
            scale = 1e-3 + tf.nn.softplus(0.05 * raw_scale)
            eps = tf.random.normal(shape=(runs, tf.shape(loc)[0], 1), dtype=loc.dtype)
            logits_samples = loc[None, :, :] + scale[None, :, :] * eps
            logits_mean = tf.reduce_mean(logits_samples, axis=0)
            variance = tf.reduce_mean(tf.math.reduce_variance(logits_samples, axis=0, keepdims=True))
            return logits_mean, variance

        return output, tf.constant(0.0, dtype=tf.float32)
        
    def _progress(self, epoch):
        pct = min(100, epoch * 100 // self.num_epochs)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        t_total = self.train_total[-1] if self.train_total else 0.0
        t_kl = self.train_kl[-1] if self.train_kl else 0.0
        t_nll = self.train_nll[-1] if self.train_nll else 0.0
        parts = f"Train: Total={t_total:.4f}, KL={t_kl:.4f}, nll={t_nll:.4f}"
        if self.train_variance:
            parts += f" var={self.train_variance[-1]:.4f}"
        if self.valid_total:
            parts += f"; Val: Total={self.valid_total[-1]:.4f}, KL={self.valid_kl[-1]:.4f}, nll={self.valid_nll[-1]:.4f}"
            if self.valid_variance:
                parts += f" var={self.valid_variance[-1]:.4f}"
        msg = f"  [{bar}] {epoch}/{self.num_epochs} {parts}"
        # When stdout is a TTY (terminal), use \r for in-place updates. When redirected (e.g. SLURM log)
        # or wrapped (e.g. TeeLogger), \r doesn't work. Print every 10 epochs to avoid 100+ lines.
        is_tty = getattr(sys.stdout, "isatty", lambda: False)()
        if is_tty:
            sys.stdout.write("\r" + msg)
        else:
            if epoch % 10 == 0 or epoch == self.num_epochs:
                print(msg)
        sys.stdout.flush()

    def train_and_evaluate(self):
        stop_training = False
        for epoch in range(1, self.num_epochs+1):
            if epoch > 0 and self.model_name == "sngp":
                self.model.layers[-1].reset_covariance_matrix() # reset covmat for SNGP
            self.train(epoch)
            if self.valid_ds is not None:
                stop_training = self.validate(epoch)
            if self.test_ds is not None:
                self.test()

            self._progress(epoch)

            if self.use_wandb:
                import wandb
                log_dict = {"epoch": epoch, "train_loss": self.train_loss[-1]}
                if self.train_total:
                    log_dict["train_total"] = self.train_total[-1]
                if self.train_kl:
                    log_dict["train_kl"] = self.train_kl[-1]
                if self.train_nll:
                    log_dict["train_nll"] = self.train_nll[-1]
                if self.train_variance:
                    log_dict["train_variance"] = self.train_variance[-1]
                if self.valid_loss:
                    log_dict["valid_loss"] = self.valid_loss[-1]
                if self.valid_total:
                    log_dict["valid_total"] = self.valid_total[-1]
                if self.valid_kl:
                    log_dict["valid_kl"] = self.valid_kl[-1]
                if self.valid_nll:
                    log_dict["valid_nll"] = self.valid_nll[-1]
                if self.valid_variance:
                    log_dict["valid_variance"] = self.valid_variance[-1]
                wandb.log(log_dict)

            if stop_training:
                self.cleanup()
                break
            self.cleanup()
        print()  # newline after progress bar

    def train(self, epoch):
        batch_variances = list()
        batch_total = list()
        batch_nll = list()
        batch_kl = list()
        runs = self.n_samples_train
        for x, y in self.train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            with tf.GradientTape() as tape:
                logits, batch_var = self._predict_logits_and_variance(x, training=True, runs=runs)
                nll = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                kl = self._regularization_term()
                loss = nll + kl
                self.train_loss_metric.update_state(loss)
            with tf.name_scope("gradients"):
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            batch_variances.append(float(batch_var.numpy()))
            batch_total.append(float(loss.numpy()))
            batch_nll.append(float(nll.numpy()))
            batch_kl.append(float(kl.numpy()))
        epoch_loss = self.train_loss_metric.result()
        self.train_loss.append(float(epoch_loss))
        self.train_total.append(float(np.mean(batch_total)) if batch_total else 0.0)
        self.train_nll.append(float(np.mean(batch_nll)) if batch_nll else 0.0)
        self.train_kl.append(float(np.mean(batch_kl)) if batch_kl else 0.0)

        # Track variance
        if len(batch_variances) > 0:
            self.train_variance.append(float(np.mean(batch_variances)))
        
        self.manager.save()

    def validate(self, epoch):
        stop_training = False
        runs = self.n_samples_valid
        batch_variances = list()
        batch_total = list()
        batch_nll = list()
        batch_kl = list()
        for x, y in self.valid_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            logits, batch_var = self._predict_logits_and_variance(x, training=False, runs=runs)
            nll = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
            kl = self._regularization_term()
            loss = nll + kl
            self.valid_loss_metric.update_state(loss)
            batch_variances.append(float(batch_var.numpy()))
            batch_total.append(float(loss.numpy()))
            batch_nll.append(float(nll.numpy()))
            batch_kl.append(float(kl.numpy()))
        epoch_loss = self.valid_loss_metric.result()
        self.valid_loss.append(float(epoch_loss))
        self.valid_total.append(float(np.mean(batch_total)) if batch_total else 0.0)
        self.valid_nll.append(float(np.mean(batch_nll)) if batch_nll else 0.0)
        self.valid_kl.append(float(np.mean(batch_kl)) if batch_kl else 0.0)

        # Track variance
        if len(batch_variances) > 0:
            self.valid_variance.append(float(np.mean(batch_variances)))

        # Early stopping
        if self.early_stop:
            current_valid_nll = self.valid_nll[-1]
            if self.best_valid_nll > current_valid_nll:
                self.best_valid_nll = current_valid_nll
                self.best_ep = epoch
            if (epoch - self.best_ep) > self.patience:
                print(f"\n  Early stop at epoch {self.best_ep}, val_loss={float(self.best_valid_nll):.4f}")
                stop_training = True
            else:
                stop_training = False
                
        return stop_training

    def test(self):
        batch_variances = list()
        for x, y in self.test_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            if self.model_name in ["MLP-ALEA", "VI", "VI-EPI", "MCD-EPI", "MCD", "vi", "mcd1", "mcd2", "mcd3"]:
                runs = self.n_samples_test
                logits_cpd = np.zeros((runs, len(x)), dtype=np.float32)
                for i in range(0, runs):
                    if self.model_name in ["MLP-ALEA", "VI", "MCD", "vi", "mcd1", "mcd2", "mcd3"]:
                        logits_cpd[i,:] = np.reshape(self.model(x, training=False).sample(), len(x))
                    else:
                        logits_cpd[i,:] = np.reshape(self.model(x, training=False), len(x))
                logits_mean = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))
                batch_variances.append(np.mean(tf.math.reduce_variance(logits_cpd, axis=0, keepdims=True)))
                
                #logits = self.model(x, training=False)
                if isinstance(self.loss_fn, CoxPHLoss):
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_mean)
                else:
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_cpd)
                self.test_loss_metric.update_state(loss)
            elif self.model_name in ["SNGP", "sngp"]:
                logits, covmat = self.model(x, training=False)
                batch_variances.append(np.mean(tf.linalg.diag_part(covmat)[:, None]))
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                self.test_loss_metric.update_state(loss)
            else:
                logits = self.model(x, training=False)
                batch_variances.append(0) # zero variance for MLP
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                self.test_loss_metric.update_state(loss)
         
        # Track variance
        if len(batch_variances) > 0:
            self.test_variance.append(float(np.mean(batch_variances)))

        epoch_loss = self.test_loss_metric.result()
        self.test_loss_scores.append(float(epoch_loss))

    def cleanup(self):
        self.train_loss_metric.reset_state()
        self.valid_loss_metric.reset_state()
        self.test_loss_metric.reset_state()
