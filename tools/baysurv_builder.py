import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# TFP DenseFlipout requires Keras 2 (tf_keras); Keras 3 KerasTensor lacks .shape.rank
try:
    import tf_keras
except ImportError:
    tf_keras = tf.keras  # fallback if tf-keras not installed


class SpectralNormalization(tf.keras.layers.Wrapper):
    """Keras 3 compatible spectral normalization wrapper."""
    def __init__(self, layer, norm_multiplier=0.9, **kwargs):
        super().__init__(layer, **kwargs)
        self.norm_multiplier = norm_multiplier

    def build(self, input_shape):
        super().build(input_shape)
        kernel = self.layer.kernel
        self.u = self.add_weight(
            name='sn_u', shape=(1, kernel.shape[-1]),
            initializer='truncated_normal', trainable=False,
        )

    def call(self, inputs, training=None):
        self._normalize_kernel()
        return self.layer(inputs)

    def _normalize_kernel(self):
        kernel = self.layer.kernel
        w = tf.reshape(kernel, [-1, kernel.shape[-1]])
        u_hat = self.u
        v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w)))
        u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w))
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat)))
        self.u.assign(u_hat)
        self.layer.kernel.assign(
            kernel * (self.norm_multiplier / tf.maximum(sigma, self.norm_multiplier))
        )


class RandomFeatureGaussianProcess(tf.keras.layers.Layer):
    """Keras 3 compatible Random Feature Gaussian Process output layer.

    Uses Random Fourier Features to approximate a GP posterior, producing
    (logits, covariance_matrix) at each forward pass.
    """
    def __init__(self, units, num_inducing=1024, normalize_input=False,
                 scale_random_features=True, gp_cov_momentum=-1, ridge_penalty=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_inducing = num_inducing
        self.normalize_input = normalize_input
        self.scale_random_features = scale_random_features
        self.gp_cov_momentum = gp_cov_momentum
        self.ridge_penalty = ridge_penalty

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.normalize_input:
            self.input_norm = tf.keras.layers.LayerNormalization()

        self.rff_w = self.add_weight(
            name='rff_w', shape=(input_dim, self.num_inducing),
            initializer=tf.initializers.RandomNormal(stddev=1.0),
            trainable=False,
        )
        self.rff_b = self.add_weight(
            name='rff_b', shape=(self.num_inducing,),
            initializer=tf.initializers.RandomUniform(minval=0., maxval=2. * np.pi),
            trainable=False,
        )
        self.beta = self.add_weight(
            name='gp_beta', shape=(self.num_inducing, self.units),
            initializer='zeros', trainable=True,
        )
        self.output_bias = self.add_weight(
            name='gp_output_bias', shape=(self.units,),
            initializer='zeros', trainable=True,
        )
        self.precision = self.add_weight(
            name='gp_precision', shape=(self.num_inducing, self.num_inducing),
            initializer='identity', trainable=False,
        )
        self._rff_scale = tf.sqrt(2.0 / float(self.num_inducing))
        super().build(input_shape)

    def _random_features(self, inputs):
        if self.normalize_input:
            inputs = self.input_norm(inputs)
        phi = tf.cos(tf.matmul(inputs, self.rff_w) + self.rff_b)
        if self.scale_random_features:
            phi = phi * self._rff_scale
        return phi

    def call(self, inputs, training=None):
        phi = self._random_features(inputs)
        logits = tf.matmul(phi, self.beta) + self.output_bias

        if training:
            batch_prec = tf.matmul(phi, phi, transpose_a=True)
            if self.gp_cov_momentum < 0:
                self.precision.assign_add(batch_prec)
            else:
                self.precision.assign(
                    self.gp_cov_momentum * self.precision
                    + (1 - self.gp_cov_momentum) * batch_prec
                )

        prec_with_ridge = self.precision + self.ridge_penalty * tf.eye(self.num_inducing)
        covmat = tf.matmul(tf.matmul(phi, tf.linalg.inv(prec_with_ridge)), phi, transpose_b=True)
        return logits, covmat

    def reset_covariance_matrix(self):
        self.precision.assign(tf.eye(self.num_inducing))

class MonteCarloDropout(tf.keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)

tfd = tfp.distributions
tfb = tfp.bijectors
kl = tfd.kullback_leibler

def normal_loc(params):
    return tfd.Normal(loc=params[:,0:1], scale=1)

def normal_loc_scale(params):
    loc = params[:, 0:1]
    raw_scale = params[:, 1:2]
    scale = 1e-3 + tf.keras.ops.softplus(0.05 * raw_scale)
    return tfd.Normal(loc=loc, scale=scale)

def normal_fs(params):
    return tfd.Normal(loc=params[:,0:1], scale=1)

def make_mlp_model(input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    if output_dim == 2: # If 2, output Gaussian params [loc, raw_scale].
        params = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=params)
    else: # Do not model aleatoric uncertain
        output = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def make_vi_model(n_train_samples, input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    """VI model uses tf_keras (Keras 2) so TFP DenseFlipout receives compatible tensors."""
    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_train_samples * 1.0)
    bias_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_train_samples * 1.0)
    inputs = tf_keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                kernel_divergence_fn=kernel_divergence_fn,
                                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn,
                                                activity_regularizer=tf_keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                kernel_divergence_fn=kernel_divergence_fn,
                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(inputs)
            hidden = tf_keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf_keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                kernel_divergence_fn=kernel_divergence_fn,
                                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn,
                                                activity_regularizer=tf_keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                kernel_divergence_fn=kernel_divergence_fn,
                                bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(hidden)
            hidden = tf_keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf_keras.layers.Dropout(dropout_rate)(hidden)
                
    if output_dim == 2: # If 2, output Gaussian params [loc, raw_scale].
        params = tfp.layers.DenseFlipout(output_dim,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn)(hidden)
        model = tf_keras.Model(inputs=inputs, outputs=params)
    else: # model only epistemic uncertain.
        output = tf_keras.layers.Dense(output_dim, activation="linear")(hidden)
        model = tf_keras.Model(inputs=inputs, outputs=output)
    return model

def make_mcd_model(input_shape, output_dim, layers,
                   activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = MonteCarloDropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = MonteCarloDropout(dropout_rate)(hidden)
    if output_dim == 2: # If 2, output Gaussian params [loc, raw_scale].
        params = tf.keras.layers.Dense(output_dim)(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=params)
    else: # model only epistemic uncertain.
        output = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def make_sngp_model(input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(shape=input_shape)
    spec_norm_bound = 0.9
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                dense = tf.keras.layers.Dense(units, activation=activation_fn,
                                              activity_regularizer=tf.keras.regularizers.L2(regularization_pen))
            else:
                dense = tf.keras.layers.Dense(units, activation=activation_fn)
            dense = SpectralNormalization(dense, norm_multiplier=spec_norm_bound)
            hidden = dense(inputs)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                dense = tf.keras.layers.Dense(units, activation=activation_fn,
                                              activity_regularizer=tf.keras.regularizers.L2(regularization_pen))
            else:
                dense = tf.keras.layers.Dense(units, activation=activation_fn)
            dense = SpectralNormalization(dense, norm_multiplier=spec_norm_bound)
            hidden = dense(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
            
    output = RandomFeatureGaussianProcess(units=output_dim,
                                          num_inducing=1024,
                                          normalize_input=False,
                                          scale_random_features=True,
                                          gp_cov_momentum=-1)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    
def make_transformer_mcd_model(input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(shape=input_shape)
    embed_dim = layers[0] if layers else 32
    depth = len(layers) if layers else 3
    num_heads = 4
    
    # Feature Tokenization (batch, features, 1) -> (batch, features, embed_dim)
    x = tf.keras.layers.Reshape((input_shape[0], 1))(inputs)
    if regularization_pen is not None:
        x = tf.keras.layers.Dense(embed_dim, activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(x)
    else:
        x = tf.keras.layers.Dense(embed_dim)(x)
        
    for i in range(depth):
        # Self-Attention Block
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x, x)
        attn_out = MonteCarloDropout(dropout_rate)(attn_out)
        x = tf.keras.layers.Add()([x, attn_out])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feed-Forward Block
        if regularization_pen is not None:
            ffn_out = tf.keras.layers.Dense(embed_dim, activation=activation_fn, activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(x)
        else:
            ffn_out = tf.keras.layers.Dense(embed_dim, activation=activation_fn)(x)
        ffn_out = tf.keras.layers.Dense(embed_dim)(ffn_out)
        ffn_out = MonteCarloDropout(dropout_rate)(ffn_out)
        x = tf.keras.layers.Add()([x, ffn_out])
        x = tf.keras.layers.LayerNormalization()(x)
        
    # Global Mean Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    if output_dim == 2:
        params = tf.keras.layers.Dense(output_dim)(x)
        model = tf.keras.Model(inputs=inputs, outputs=params)
    else:
        output = tf.keras.layers.Dense(output_dim, activation="linear")(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
    