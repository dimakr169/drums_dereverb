import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models

# Sinusoidal Embedding Functions
@tf.keras.utils.register_keras_serializable()
def sinusoidal_embedding(timesteps, embedding_dim):
    frequencies = tf.math.exp(
        tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), embedding_dim // 2)
    )
    t = tf.cast(timesteps, tf.float32)[:, None]
    ang = 2.0 * math.pi * frequencies
    return tf.concat([tf.sin(ang * t), tf.cos(ang * t)], axis=-1)

@tf.keras.utils.register_keras_serializable()
def get_timestep_embedding(timesteps, embedding_dim):
    half = embedding_dim // 2
    emb = -math.log(10000.0) / (half - 1)
    emb = tf.exp(tf.range(half, dtype=tf.float32) * emb)
    emb = tf.cast(timesteps, tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    if embedding_dim % 2:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb

class TimestepEmbedding(layers.Layer):
    def __init__(self, dim, var=False):
        super().__init__()
        self.dim = dim; self.var = var
    def call(self, t):
        return sinusoidal_embedding(t, self.dim) if self.var else get_timestep_embedding(t, self.dim)

class AttentionBlock(layers.Layer):
    def __init__(self, channels: int, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=channels // num_heads,
            attention_axes=(1,)
        )
        self.proj = layers.Dense(channels)
    def call(self, x):
        h = self.norm(x)
        B, H, W, C = tf.unstack(tf.shape(h))
        seq = tf.reshape(h, [B, H * W, C])
        attn = self.mha(seq, seq)
        attn = tf.reshape(attn, [B, H, W, C])
        return x + self.proj(attn)

class Downsample(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.avg = layers.AveragePooling2D(pool_size=3, strides=1, padding='same')
        self.conv = layers.Conv2D(channels, 3, strides=2, padding='same')
    def call(self, x):
        return self.conv(self.avg(x))

class Upsample(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv = layers.Conv2D(channels, 3, padding='same')
    def call(self, x):
        B, H, W, C = tf.unstack(tf.shape(x))
        x = tf.image.resize(x, [H * 2, W * 2], method='nearest')
        return self.conv(x)

class ResNetBlock(layers.Layer):
    def __init__(self, in_ch, out_ch=None, dropout=0.1, use_norm=True, conv_shortcut=True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch or in_ch
        self.use_norm = use_norm; self.dropout = dropout; self.conv_shortcut = conv_shortcut
        if use_norm:
            self.norm1 = layers.LayerNormalization(epsilon=1e-5)
            self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.act1 = layers.Activation('swish')
        self.conv1 = layers.Conv2D(self.out_ch, 3, padding='same')
        self.scale = layers.Dense(self.out_ch)
        self.shift = layers.Dense(self.out_ch)
        self.act2 = layers.Activation('swish')
        self.drop = layers.Dropout(self.dropout)
        self.conv2 = layers.Conv2D(self.out_ch, 3, padding='same')
        if self.in_ch != self.out_ch:
            self.shortcut = layers.Conv2D(self.out_ch, 1, padding='same') if conv_shortcut else layers.Dense(self.out_ch)
    def call(self, x, temb):
        h = x
        if self.use_norm: h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)
        s = self.scale(tf.nn.swish(temb))[:, None, None, :]
        b = self.shift(tf.nn.swish(temb))[:, None, None, :]
        h = h * (1 + s) + b
        if self.use_norm: h = self.norm2(h)
        h = self.act2(h)
        h = self.drop(h)
        h = self.conv2(h)
        res = x if self.in_ch == self.out_ch else self.shortcut(x)
        return res + 0.1 * h

class UNet(models.Model):
    def __init__(self, config):
        super().__init__()
        C = config.channels
        ch_mult = config.ch_mult
        num_res = len(ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.use_attention = config.use_attention
        self.continuous_emb = config.continuous_emb
        self.create_mask = config.create_mask
        # timestep embedding
        self.temb_layers = [
            TimestepEmbedding(C * 2, self.continuous_emb),
            layers.Dense(C * 4, activation='swish'),
            layers.Dense(C * 4, activation='swish')
        ]
        # input conv
        self.input_conv = layers.Conv2D(C, (9, 1), padding='same')
        # down path blocks
        self.down_res = []
        self.down_samp = []
        in_ch = C
        for i, mult in enumerate(ch_mult):
            out_ch = C * mult
            res_blocks = []
            for _ in range(self.num_res_blocks):
                res_blocks.append(ResNetBlock(in_ch, out_ch, config.dropout, config.use_norm))
                in_ch = out_ch
            self.down_res.append(res_blocks)
            if i < num_res - 1:
                self.down_samp.append(Downsample(out_ch))
        # middle blocks
        self.mid_blocks = []
        for _ in range(2):
            self.mid_blocks.append(ResNetBlock(in_ch, in_ch, config.dropout, config.use_norm))
        if self.use_attention:
            self.mid_blocks.append(AttentionBlock(in_ch))
        # up path blocks
        self.up_samp = []
        self.up_res = []
        for i, mult in enumerate(reversed(ch_mult)):
            out_ch = C * mult
            res_blocks = []
            for _ in range(self.num_res_blocks + 1):
                res_blocks.append(ResNetBlock(in_ch * 2, out_ch, config.dropout, config.use_norm))
                in_ch = out_ch
            self.up_res.append(res_blocks)
            if i < num_res - 1:
                self.up_samp.append(Upsample(in_ch))
        # output convs
        out_c = 2 if config.ri_inp else 1
        self.output_convs = [layers.Conv2D(C, 3, padding='same'), layers.Conv2D(out_c, 3, padding='same')]

    def call(self, inputs):
        x, t = inputs
        # timestep embed
        for layer in self.temb_layers:
            t = layer(t)
        # initial conv
        h = self.input_conv(x)
        # store skips
        skips = []
        # down path
        for i, res_blocks in enumerate(self.down_res):
            for block in res_blocks:
                h = block(h, t)
            skips.append(h)
            if i < len(self.down_samp):
                h = self.down_samp[i](h)
        # middle
        for block in self.mid_blocks:
            if isinstance(block, ResNetBlock):
                h = block(h, t)
            else:
                h = block(h)
        # up path
        for i, res_blocks in enumerate(self.up_res):
            if i < len(self.up_samp):
                h = self.up_samp[i](h)
            skip = skips.pop()
            # resize skip
            skip = tf.image.resize(skip, [tf.shape(h)[1], tf.shape(h)[2]], method='nearest')
            for block in res_blocks:
                h = block(tf.concat([h, skip], axis=-1), t)
        # output
        for layer in self.output_convs:
            h = layer(h)
        return tf.sigmoid(h) if self.create_mask else h
