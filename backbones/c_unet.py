# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:08:46 2024

@author: dimak
"""
import math

import tensorflow as tf

from .clayers import (
    complex_Conv2D,
    complex_Conv2DTranspose,
    complex_Dense,
    complex_LeakyReLU,
    complex_LSTM,
    complex_NaiveBatchNormalization,
    complex_Swish,
)


class DiffusionStepEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim):
        super(DiffusionStepEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def call(self, t):
        t = tf.cast(t, dtype=tf.float32)

        emb = math.log(10000) / (self.embed_dim - 1)
        emb = tf.math.exp(tf.range(self.embed_dim, dtype=tf.float32) * -emb)
        emb = t[:, None] * emb[None, :]

        # Compute real (cosine) and imaginary (sine) parts separately
        real_part = tf.cos(emb)
        imag_part = tf.sin(emb)

        out = tf.stack([real_part, imag_part], axis=-1)

        return out  # Return as separate tensors


class Embedding(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(Embedding, self).__init__()

        self.channels = channels

        self.emb = DiffusionStepEmbedding(self.channels * 2)
        self.cdense_1 = complex_Dense(self.channels * 4)
        self.c_act = complex_Swish()
        self.cdense_2 = complex_Dense(self.channels * 4)

    def call(self, x):

        x = self.emb(x)
        x = self.cdense_1(x)
        x = self.c_act(x)
        x = self.cdense_2(x)

        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, emb, filters, kernel_size, strides):
        super(Encoder, self).__init__()
        self.emb = emb
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.cConv2D = complex_Conv2D(
            self.filters, self.kernel_size, self.strides, padding="same"
        )
        self.cbn = complex_NaiveBatchNormalization()
        # self.cbn = ComplexBatchNorm()
        self.clrelu = complex_LeakyReLU()
        # self.clrelu = tf.keras.layers.PReLU() #PreLU

        # for embeddings
        if self.emb:
            self.cswish = complex_Swish()
            self.cdense = complex_Dense(self.filters)

    def call(self, inputs, temb, training=True):

        out = self.cConv2D(inputs)

        if temb is not None:
            # Add in timestep embedding.
            t_out = self.cdense(self.cswish(temb))
            out = out + t_out[:, None, None, :, :]

        out = self.cbn(out, training)
        out = self.clrelu(out)

        return out


class Decoder(tf.keras.layers.Layer):
    def __init__(self, emb, filters, kernel_size, strides, last_layer=False):
        super(Decoder, self).__init__()
        self.emb = emb
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.cConv2DT = complex_Conv2DTranspose(
            self.filters, self.kernel_size, self.strides, padding="same"
        )
        self.cbn = complex_NaiveBatchNormalization()
        # self.cbn = ComplexBatchNorm()
        self.clrelu = complex_LeakyReLU()
        # self.clrelu = tf.keras.layers.PReLU() #PreLU

        # for embeddings
        if self.emb:
            self.cswish = complex_Swish()
            self.cdense = complex_Dense(self.filters)

        self.last_layer = last_layer

    def call(self, inputs, temb, training=True):

        out = self.cConv2DT(inputs)

        if not self.last_layer:

            if temb is not None:
                # Add in timestep embedding.
                t_out = self.cdense(self.cswish(temb))
                out = out + t_out[:, None, None, :, :]

            out = self.cbn(out, training)
            out = self.clrelu(out)

            return out
        else:

            # calculate mask from DCCRN paper
            mask_real = out[..., 0]
            mask_imag = out[..., 1]

            mask_mags = (mask_real**2 + mask_imag**2) ** 0.5
            real_phase = mask_real / (mask_mags + 1e-8)
            imag_phase = mask_imag / (mask_mags + 1e-8)

            mask_phase = tf.atan2(imag_phase, real_phase)

            mask_mags = tf.math.tanh(mask_mags)

            return mask_mags, mask_phase


class DCUNet(tf.keras.models.Model):
    """
    Deep Complex U-Net class of the model.
    """

    def __init__(self, emb=True, use_lstms=True, hidden_dim=128):
        super().__init__()

        self.emb = emb  # with False it can be turned to predictive
        self.use_lstms = use_lstms
        self.hidden_dim = hidden_dim

        if self.emb:
            self.emb_layer = Embedding(32)

        # downsampling/encoding
        self.downsample0 = Encoder(
            self.emb, filters=32, kernel_size=(7, 5), strides=(2, 2)
        )
        self.downsample1 = Encoder(
            self.emb, filters=32, kernel_size=(7, 5), strides=(2, 1)
        )
        self.downsample2 = Encoder(
            self.emb, filters=64, kernel_size=(7, 5), strides=(2, 2)
        )
        self.downsample3 = Encoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 1)
        )
        self.downsample4 = Encoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 2)
        )
        self.downsample5 = Encoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 1)
        )
        self.downsample6 = Encoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 2)
        )
        self.downsample7 = Encoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 1)
        )

        # if using complex LSTMs it turns to DCCRN
        if self.use_lstms:
            self.mid_lstm1 = complex_LSTM(
                self.hidden_dim, projection_dim=64, bidirectional=False
            )
            self.mid_lstm2 = complex_LSTM(
                self.hidden_dim, projection_dim=64, bidirectional=False
            )

        # upsampling/decoding
        self.upsample0 = Decoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 1)
        )
        self.upsample1 = Decoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 2)
        )
        self.upsample2 = Decoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 1)
        )
        self.upsample3 = Decoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 2)
        )
        self.upsample4 = Decoder(
            self.emb, filters=64, kernel_size=(5, 3), strides=(2, 1)
        )
        self.upsample5 = Decoder(
            self.emb, filters=32, kernel_size=(7, 5), strides=(2, 2)
        )
        self.upsample6 = Decoder(
            self.emb, filters=32, kernel_size=(7, 5), strides=(2, 1)
        )
        self.upsample7 = Decoder(
            self.emb, filters=1, kernel_size=(7, 5), strides=(2, 2), last_layer=True
        )

    def call(self, x, temb=None, training=True):

        real = x[..., 0]
        imag = x[..., 1]

        spec_mags = tf.sqrt(real**2 + imag**2 + 1e-8)
        spec_phase = tf.atan2(imag, real)

        if temb is not None:
            # embedding layer
            emb_out = self.emb_layer(temb)
        else:
            emb_out = None

        # downsampling/encoding
        d0 = self.downsample0(x, emb_out, training)
        d1 = self.downsample1(d0, emb_out, training)
        d2 = self.downsample2(d1, emb_out, training)
        d3 = self.downsample3(d2, emb_out, training)
        d4 = self.downsample4(d3, emb_out, training)
        d5 = self.downsample5(d4, emb_out, training)
        d6 = self.downsample6(d5, emb_out, training)
        d7 = self.downsample7(d6, emb_out, training)

        # mid lstms
        if self.use_lstms:
            # squeeze height dimension (1)
            d7 = tf.squeeze(d7, axis=1)
            l1 = self.mid_lstm1(d7)
            l2 = self.mid_lstm2(l1)
            # expand dim
            d7 = tf.expand_dims(l2, axis=1)

        # upsampling/decoding
        u0 = self.upsample0(d7, emb_out, training)
        # skip-connection
        c0 = tf.concat([u0, d6], axis=3)

        u1 = self.upsample1(c0, emb_out, training)
        c1 = tf.concat([u1, d5], axis=3)

        u2 = self.upsample2(c1, emb_out, training)
        c2 = tf.concat([u2, d4], axis=3)

        u3 = self.upsample3(c2, emb_out, training)
        c3 = tf.concat([u3, d3], axis=3)

        u4 = self.upsample4(c3, emb_out, training)
        c4 = tf.concat([u4, d2], axis=3)

        u5 = self.upsample5(c4, emb_out, training)
        c5 = tf.concat([u5, d1], axis=3)

        u6 = self.upsample6(c5, emb_out, training)
        c6 = tf.concat([u6, d0], axis=3)

        # last
        mask_mags, mask_phase = self.upsample7(c6, emb_out, training)

        # apply mask for mag and phase
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real_part = est_mags * tf.math.cos(est_phase)
        imag_part = est_mags * tf.math.sin(est_phase)

        out = tf.stack([real_part, imag_part], axis=-1)

        return out
