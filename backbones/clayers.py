# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:48 2024

@author: dimak
"""

import tensorflow as tf

# Complex Dropout"


class complex_Dropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(complex_Dropout, self).__init__(**kwargs)
        self.rate = rate  # Dropout probability

        self.real_dropped = tf.keras.layers.Dropout(rate=self.rate)
        self.imag_dropped = tf.keras.layers.Dropout(rate=self.rate)

    def call(self, x, training=False):

        real_outputs = self.real_dropped(x[..., 0], training=training)
        imag_outputs = self.imag_dropped(x[..., 1], training=training)

        out = tf.stack([real_outputs, imag_outputs], axis=-1)

        return out


# Complex Batch Normalization from DCCRN implementation"
# non working properly"


class ComplexBatchNorm(tf.keras.layers.Layer):
    def __init__(
        self,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        feature_axis=-2,
        **kwargs
    ):
        super(ComplexBatchNorm, self).__init__(name="complexbatch", **kwargs)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.feature_axis = feature_axis
        self.built = False

    def build(self, input_shape):
        dim = input_shape[self.feature_axis]
        self.num_features = dim

        if self.affine:
            self.Wrr = self.add_weight(
                name="Wrr", shape=(dim,), initializer="ones", trainable=True
            )
            self.Wri = self.add_weight(
                name="Wri",
                shape=(dim,),
                initializer=tf.initializers.RandomUniform(minval=-0.9, maxval=0.9),
                trainable=True,
            )
            self.Wii = self.add_weight(
                name="Wii", shape=(dim,), initializer="ones", trainable=True
            )
            self.Br = self.add_weight(
                name="Br", shape=(dim,), initializer="zeros", trainable=True
            )
            self.Bi = self.add_weight(
                name="Bi", shape=(dim,), initializer="zeros", trainable=True
            )

        if self.track_running_stats:
            self.RMr = self.add_weight(
                name="RMr", shape=(dim,), initializer="zeros", trainable=False
            )
            self.RMi = self.add_weight(
                name="RMi", shape=(dim,), initializer="zeros", trainable=False
            )
            self.RVrr = self.add_weight(
                name="RVrr", shape=(dim,), initializer="ones", trainable=False
            )
            self.RVri = self.add_weight(
                name="RVri", shape=(dim,), initializer="zeros", trainable=False
            )
            self.RVii = self.add_weight(
                name="RVii", shape=(dim,), initializer="ones", trainable=False
            )
            self.num_batches_tracked = self.add_weight(
                name="n_b_t",
                shape=(),
                initializer="zeros",
                trainable=False,
                dtype=tf.int64,
            )

    def call(self, inputs, training=False):

        xr, xi = inputs[..., 0], inputs[..., 1]

        if training:
            Mr, Mi = tf.reduce_mean(xr, axis=[1, 2, 3], keepdims=True), tf.reduce_mean(
                xi, axis=[1, 2, 3], keepdims=True
            )
            Vrr = (
                tf.reduce_mean(tf.square(xr - Mr), axis=[1, 2, 3], keepdims=True)
                + self.eps
            )
            Vri = tf.reduce_mean((xr - Mr) * (xi - Mi), axis=[1, 2, 3], keepdims=True)
            Vii = (
                tf.reduce_mean(tf.square(xi - Mi), axis=[1, 2, 3], keepdims=True)
                + self.eps
            )

            if self.track_running_stats:
                new_RMr = tf.reduce_mean(
                    Mr, axis=[0, 1, 2]
                )  # Adjust axis based on actual input shape if needed
                new_RMi = tf.reduce_mean(Mi, axis=[0, 1, 2])
                self.RMr.assign(
                    self.RMr * self.momentum + new_RMr * (1 - self.momentum)
                )
                self.RMi.assign(
                    self.RMi * self.momentum + new_RMi * (1 - self.momentum)
                )
                new_RVrr = tf.reduce_mean(
                    Vrr, axis=[0, 1, 2]
                )  # Adjust axis based on actual input shape if needed
                new_RVri = tf.reduce_mean(Vri, axis=[0, 1, 2])
                new_RVii = tf.reduce_mean(Vii, axis=[0, 1, 2])
                self.RVrr.assign(
                    self.RVrr * self.momentum + new_RVrr * (1 - self.momentum)
                )
                self.RVri.assign(
                    self.RVri * self.momentum + new_RVri * (1 - self.momentum)
                )
                self.RVii.assign(
                    self.RVii * self.momentum + new_RVii * (1 - self.momentum)
                )
                self.num_batches_tracked.assign_add(1)
        else:
            Mr, Mi, Vrr, Vri, Vii = self.RMr, self.RMi, self.RVrr, self.RVri, self.RVii

        # Centralize data using either batch or running statistics
        xr, xi = xr - Mr, xi - Mi

        # Compute the square root of the matrix [[Vrr, Vri], [Vri, Vii]] + I*eps
        tau = Vrr + Vii
        delta = Vrr * Vii - Vri * Vri
        s = tf.sqrt(delta)
        t = tf.sqrt(tau + 2 * s)
        rst = 1 / (s * t)
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst

        if self.affine:
            Zrr = self.Wrr * Urr + self.Wri * Uri
            Zri = self.Wrr * Uri + self.Wri * Uii
            Zir = self.Wri * Urr + self.Wii * Uri
            Zii = self.Wri * Uri + self.Wii * Uii
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = Zrr * xr + Zri * xi
        yi = Zir * xr + Zii * xi

        if self.affine:
            yr += self.Br
            yi += self.Bi

        outputs = tf.stack([yr, yi], axis=-1)
        return outputs

    def get_config(self):
        config = super(ComplexBatchNorm, self).get_config()
        config.update(
            {
                "eps": self.eps,
                "momentum": self.momentum,
                "affine": self.affine,
                "track_running_stats": self.track_running_stats,
                "feature_axis": self.feature_axis,
            }
        )
        return config


# "Complex AveragePooling2D"


class complex_AveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding="valid", **kwargs):
        super(complex_AveragePooling2D, self).__init__(**kwargs)
        # Define real and imaginary average pooling layers
        self.real_pool = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size, strides=strides, padding=padding
        )
        self.imag_pool = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size, strides=strides, padding=padding
        )

    def call(self, x):

        real, imag = x[..., 0], x[..., 1]

        # Apply average pooling to real and imaginary parts separately
        pooled_real = self.real_pool(real)
        pooled_imag = self.imag_pool(imag)

        out = tf.stack([pooled_real, pooled_imag], axis=-1)

        return out


# "COMPLEX LSTM"


class complex_LSTM(tf.keras.layers.Layer):
    def __init__(self, hidden_size, projection_dim=None, bidirectional=False):
        super(complex_LSTM, self).__init__()

        self.rnn_units = hidden_size // 2
        self.bidirectional = bidirectional

        # Define real and imaginary LSTM layers
        if self.bidirectional:
            self.real_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.rnn_units, return_sequences=True)
            )
            self.imag_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.rnn_units, return_sequences=True)
            )
        else:
            self.real_lstm = tf.keras.layers.LSTM(self.rnn_units, return_sequences=True)
            self.imag_lstm = tf.keras.layers.LSTM(self.rnn_units, return_sequences=True)

        # Define projection layers if projection_dim is provided
        self.projection_dim = projection_dim if projection_dim is not None else None
        if self.projection_dim is not None:
            self.r_trans = tf.keras.layers.Dense(self.projection_dim)
            self.i_trans = tf.keras.layers.Dense(self.projection_dim)

    def call(self, x):
        # Assuming inputs are in the shape [batch, time, features] if batch_first=True
        real, imag = x[..., 0], x[..., 1]

        # Process real and imaginary parts through their respective LSTMs
        r2r_out = self.real_lstm(real)
        r2i_out = self.imag_lstm(real)
        i2r_out = self.real_lstm(imag)
        i2i_out = self.imag_lstm(imag)

        # Combine the outputs
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out

        # Apply projection if needed
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)

        out = tf.stack([real_out, imag_out], axis=-1)

        return out


# "COMPLEX DENSE"


class complex_Dense(tf.keras.layers.Layer):
    """
    tf.keras.layers.Dense(
        units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        **kwargs
    )
    """

    def __init__(
        self,
        units=512,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ):
        super(complex_Dense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def build(self, inputs_shape):
        self.real_Dense = tf.keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.imag_Dense = tf.keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        super(complex_Dense, self).build(inputs_shape)

    def call(self, x):

        real_outputs = self.real_Dense(x[..., 0]) - self.imag_Dense(x[..., 1])
        imag_outputs = self.real_Dense(x[..., 1]) + self.imag_Dense(x[..., 0])

        out = tf.stack([real_outputs, imag_outputs], axis=-1)
        return out


# "COMPLEX CONVOLUTION 2D"


class complex_Conv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        super(complex_Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, inputs_shape):
        self.real_Conv2D = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.imag_Conv2D = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        super(complex_Conv2D, self).build(inputs_shape)

    def call(self, x):

        real_outputs = self.real_Conv2D(x[..., 0]) - self.imag_Conv2D(x[..., 1])
        imag_outputs = self.real_Conv2D(x[..., 1]) + self.imag_Conv2D(x[..., 0])

        out = tf.stack([real_outputs, imag_outputs], axis=-1)
        return out


# "COMPLEX CONV 2D TRANSPOSE"


class complex_Conv2DTranspose(tf.keras.layers.Layer):
    def __init__(
        self,
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        super(complex_Conv2DTranspose, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, inputs_shape):
        self.real_Conv2DTranspose = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.imag_Conv2DTranspose = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        super(complex_Conv2DTranspose, self).build(inputs_shape)

    def call(self, x):

        real_outputs = self.real_Conv2DTranspose(x[..., 0]) - self.imag_Conv2DTranspose(
            x[..., 1]
        )
        imag_outputs = self.real_Conv2DTranspose(x[..., 1]) + self.imag_Conv2DTranspose(
            x[..., 0]
        )

        out = tf.stack([real_outputs, imag_outputs], axis=-1)
        return out


# "COMPLEX POOLING"


class complex_MaxPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(1, 1), padding="same"):
        super(complex_MaxPooling, self).__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def build(self, inputs_shape):
        self.real_maxpooling = tf.keras.layers.MaxPool2D(
            pool_size=self.pool_size, strides=self.strides, padding=self.padding
        )
        self.imag_maxpooling = tf.keras.layers.MaxPool2D(
            pool_size=self.pool_size, strides=self.strides, padding=self.padding
        )
        super(complex_MaxPooling, self).build(inputs_shape)

    def call(self, real_inputs, imag_inputs):
        real_outputs = self.real_maxpooling(real_inputs)
        imag_outputs = self.imag_maxpooling(imag_inputs)
        return real_outputs, imag_outputs


# "COMPLEX naive BatchNormalization"


class complex_NaiveBatchNormalization(tf.keras.layers.Layer):
    """
    tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                        gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
                                        fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
                                        **kwargs)
    """

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        trainable=True,
        virtual_batch_size=None,
        adjustment=None,
        **kwargs
    ):

        super(complex_NaiveBatchNormalization, self).__init__()

        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.renorm = renorm
        self.renorm_clipping = renorm_clipping
        self.renorm_momentum = renorm_momentum
        self.fused = fused
        self.trainable = trainable
        self.virtual_batch_size = virtual_batch_size
        self.adjustment = adjustment

        self.real_batchnormalization = tf.keras.layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            beta_initializer=self.beta_initializer,
            gamma_initializer=self.gamma_initializer,
            moving_mean_initializer=self.moving_mean_initializer,
            moving_variance_initializer=self.moving_variance_initializer,
            beta_regularizer=self.beta_regularizer,
            gamma_regularizer=self.gamma_regularizer,
            beta_constraint=self.beta_constraint,
            gamma_constraint=self.gamma_constraint,
            renorm=self.renorm,
            renorm_clipping=self.renorm_clipping,
            renorm_momentum=self.renorm_momentum,
            fused=self.fused,
            trainable=self.trainable,
            virtual_batch_size=self.virtual_batch_size,
            adjustment=self.adjustment,
        )

        self.imag_batchnormalization = tf.keras.layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            beta_initializer=self.beta_initializer,
            gamma_initializer=self.gamma_initializer,
            moving_mean_initializer=self.moving_mean_initializer,
            moving_variance_initializer=self.moving_variance_initializer,
            beta_regularizer=self.beta_regularizer,
            gamma_regularizer=self.gamma_regularizer,
            beta_constraint=self.beta_constraint,
            gamma_constraint=self.gamma_constraint,
            renorm=self.renorm,
            renorm_clipping=self.renorm_clipping,
            renorm_momentum=self.renorm_momentum,
            fused=self.fused,
            trainable=self.trainable,
            virtual_batch_size=self.virtual_batch_size,
            adjustment=self.adjustment,
        )

    def call(self, x, training=True):

        real_outputs = self.real_batchnormalization(x[..., 0], training=training)
        imag_outputs = self.imag_batchnormalization(x[..., 1], training=training)

        out = tf.stack([real_outputs, imag_outputs], axis=-1)
        return out


# "COMPLEX LeakeyReLU"


class complex_LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.3):
        super(complex_LeakyReLU, self).__init__()
        # Separate LeakyReLU instances for real and imaginary parts
        self.real_leaky_relu = tf.keras.layers.LeakyReLU(alpha=alpha)
        self.imag_leaky_relu = tf.keras.layers.LeakyReLU(alpha=alpha)

    def call(self, x):

        real_outputs = self.real_leaky_relu(x[..., 0])
        imag_outputs = self.imag_leaky_relu(x[..., 1])

        out = tf.stack([real_outputs, imag_outputs], axis=-1)

        return out


class complex_Swish(tf.keras.layers.Layer):
    def __init__(self):
        super(complex_Swish, self).__init__()
        # Separate LeakyReLU instances for real and imaginary parts
        self.real_swish = tf.keras.layers.Activation("swish")
        self.imag_swish = tf.keras.layers.Activation("swish")

    def call(self, x):

        real_outputs = self.real_swish(x[..., 0])
        imag_outputs = self.imag_swish(x[..., 1])

        out = tf.stack([real_outputs, imag_outputs], axis=-1)

        return out
