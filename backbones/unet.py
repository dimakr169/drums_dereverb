import math

import tensorflow as tf
from tensorflow.keras import layers, models

embedding_min_frequency = 1.0
embedding_max_frequency = 1000.0  # 1000 default


@tf.keras.utils.register_keras_serializable()
def sinusoidal_embedding(timesteps, embedding_dim):
    # Works better for continues values
    frequencies = tf.math.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),  # start
            tf.math.log(embedding_max_frequency),  # stop
            embedding_dim // 2,  # num
        )
    )
    t = tf.cast(timesteps, dtype=tf.float32)[:, None]
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, dtype=tf.float32)
    embeddings = tf.concat(
        [tf.math.sin(angular_speeds * t), tf.math.cos(angular_speeds * t)], axis=-1
    )
    return embeddings


@tf.keras.utils.register_keras_serializable()
def get_timestep_embedding(timesteps, embedding_dim):
    # Discrete timesteps

    # From fairseq. Build sinusoidal embeddings. This matches the
    # implementation in tensor2tensor, but differs slightly from the
    # description in Section 3.5 of "Attention Is All You Need".
    # assert len(timesteps.shape) == 1 # and timesteps.dtype == tf.int32

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad.
        # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    # assert emb.shape == [timesteps.shape[0], embedding_dim]
    return emb


class TimestepEmbedding(layers.Layer):
    def __init__(self, dim, var=False):
        super(TimestepEmbedding, self).__init__()
        self.dim = dim
        self.var = var

    def call(self, t):
        if self.var:
            return sinusoidal_embedding(t, self.dim)  # for continues
        else:
            return get_timestep_embedding(t, self.dim)  # original


class Upsample(layers.Layer):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.channels = channels
        self.conv = layers.Conv2DTranspose(
            self.channels, (3, 3), padding="same", strides=2
        )

    def call(self, inputs):
        x = self.conv(inputs)
        return x


class Downsample(layers.Layer):
    def __init__(self, channels, with_conv=True):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        self.channels = channels
        self.conv = layers.Conv2D(self.channels, (3, 3), padding="same", strides=2)
        self.avg_pool = layers.AveragePooling2D(strides=2, padding="same")

    def call(self, inputs):

        if self.with_conv:
            x = self.conv(inputs)
        else:
            x = self.avg_pool(inputs)

        return x


# Kernel initializer to use
@tf.keras.utils.register_keras_serializable()
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


# not used so far
class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, channels, groups=8, **kwargs):
        self.channels = channels
        self.groups = groups
        super().__init__(**kwargs)

        # self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(self.channels, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(self.channels, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(self.channels, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(self.channels, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.channels, tf.float32) ** (-0.5)

        # inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class ResNetBlock(layers.Layer):
    def __init__(
        self, in_ch, out_ch=None, conv_shortcut=False, use_bn=False, dropout=0.0
    ):
        super(ResNetBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut
        self.use_bn = use_bn
        self.dropout = dropout

        if self.out_ch is None:
            self.out_ch = self.in_ch
        self.c_not_out_ch = self.in_ch != self.out_ch

        # GN/BN layers
        if self.use_bn:
            self.group_norm1 = tf.keras.layers.BatchNormalization()
            self.group_norm3 = tf.keras.layers.BatchNormalization()

        # Rest Layers.
        self.non_linear1 = layers.Activation("swish")
        self.conv1 = layers.Conv2D(self.out_ch, (3, 3), padding="same")

        self.non_linear2 = layers.Activation("swish")
        self.dense2 = layers.Dense(self.out_ch)

        self.non_linear3 = layers.Activation("swish")
        self.dropout3 = layers.Dropout(self.dropout)

        self.conv4 = layers.Conv2D(self.out_ch, (3, 3), padding="same")
        self.dense4 = layers.Dense(self.out_ch)

    def call(self, inputs, temb):
        x = inputs

        if self.use_bn:
            x = self.group_norm1(x)
        x = self.non_linear1(x)
        x = self.conv1(x)

        # Add in timestep embedding.
        # x += self.dense2(self.non_linear2(temb))[:, tf.newaxis, tf.newaxis, :]

        # Compute the expected shape for broadcasting
        temb_processed = self.dense2(self.non_linear2(temb))
        batch_size, height, width, _ = x.shape
        temb_processed = tf.reshape(temb_processed, [batch_size, 1, 1, self.out_ch])
        temb_processed = tf.tile(temb_processed, [1, height, width, 1])

        # Add in timestep embedding, ensuring the shape is statically known.
        x += temb_processed

        if self.use_bn:
            x = self.group_norm3(x)
        x = self.non_linear3(x)
        x = self.dropout3(x)

        if self.c_not_out_ch:
            if self.conv_shortcut:
                inputs = self.conv4(inputs)
            else:
                inputs = self.dense4(inputs)

        return inputs + x


@tf.keras.utils.register_keras_serializable()
class UNet(models.Model):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.config = config
        self.num_res_blocks = self.config.num_res_blocks
        self.attn_resolutions = self.config.attn_resolutions
        self.channels = self.config.channels
        self.ch_mult = self.config.ch_mult
        self.dropout = self.config.dropout
        self.resample_with_conv = self.config.resample_with_conv
        self.num_resolutions = len(self.ch_mult)
        self.create_mask = (
            self.config.create_mask
        )  # creating a mask or being generative
        self.continuous_emb = self.config.continuous_emb
        self.ri_inp = (
            self.config.ri_inp
        )  # if input is Magnitude only (1) or Real/Imaginary (2)
        self.use_bn = self.config.use_bn  # if BN layers to be used on Residual blocks
        self.has_attention = any(self.attn_resolutions)

        self.in_embed = [
            TimestepEmbedding(self.channels * 2, self.continuous_emb),
            layers.Dense(self.channels * 4),
            layers.Activation("swish"),
            layers.Dense(self.channels * 4),
        ]

        # Downsampling.
        self.pre_process = layers.Conv2D(self.channels, (3, 3), padding="same")
        self.downsampling = []
        channel_track = self.channels
        for i_level in range(len(self.ch_mult)):
            downsampling_block = []
            # Residual blocks for this resolution.
            for _ in range(self.num_res_blocks):
                downsampling_block.append(
                    ResNetBlock(
                        in_ch=channel_track,
                        out_ch=self.channels * self.ch_mult[i_level],
                        use_bn=self.use_bn,
                        dropout=self.dropout,
                    )
                )
                # add attention
                if self.attn_resolutions[i_level]:
                    downsampling_block.append(
                        AttentionBlock(channels=self.channels * self.ch_mult[i_level])
                    )
            if i_level != self.num_resolutions - 1:
                downsampling_block.append(
                    Downsample(
                        channels=self.channels * self.ch_mult[i_level],
                        with_conv=self.resample_with_conv,
                    )
                )
            channel_track = self.channels * self.ch_mult[i_level]
            self.downsampling.append(downsampling_block)

        # Middle.
        if self.has_attention:
            self.middle = [
                ResNetBlock(
                    in_ch=channel_track, use_bn=self.use_bn, dropout=self.dropout
                ),
                AttentionBlock(channels=channel_track),
                ResNetBlock(
                    in_ch=channel_track, use_bn=self.use_bn, dropout=self.dropout
                ),
            ]
        else:
            self.middle = [
                ResNetBlock(
                    in_ch=channel_track, use_bn=self.use_bn, dropout=self.dropout
                ),
                ResNetBlock(
                    in_ch=channel_track, use_bn=self.use_bn, dropout=self.dropout
                ),
            ]

        # Upsampling.
        self.upsampling = []
        channel_track = self.channels * self.ch_mult[-1] * 2
        for i_level in reversed(range(self.num_resolutions)):
            upsampling_block = []
            # Residual blocks for this resolution.
            for _ in range(self.num_res_blocks + 1):
                upsampling_block.append(
                    ResNetBlock(
                        in_ch=channel_track,
                        out_ch=self.channels * self.ch_mult[i_level],
                        use_bn=self.use_bn,
                        dropout=self.dropout,
                    )
                )
                # add attention
                if self.attn_resolutions[i_level]:
                    upsampling_block.append(
                        AttentionBlock(channels=self.channels * self.ch_mult[i_level])
                    )
            # Upsample.
            if i_level != 0:
                upsampling_block.append(
                    Upsample(
                        channels=self.channels * self.ch_mult[i_level],
                    )
                )
            channel_track = self.channels * self.ch_mult[i_level]
            self.upsampling.append(upsampling_block)

        # End.
        if self.ri_inp:
            out_channels = 2
        else:
            out_channels = 1

        self.end = [
            layers.Conv2D(self.channels, (3, 3), padding="same"),
            layers.Conv2D(out_channels, (3, 3), (1, 1), padding="same"),
        ]

    def call(self, inp):

        x = inp[0]
        temb = inp[1]

        for lay in self.in_embed:
            temb = lay(temb)
        # Downsampling.
        hs = [self.pre_process(x)]
        for i in range(len(self.downsampling)):
            block = self.downsampling[i]
            if self.attn_resolutions[i]:
                total_res_blocks = self.num_res_blocks * 2  # with attention
                for idx_block in range(0, total_res_blocks, 2):
                    h = block[idx_block](hs[-1], temb)
                    h = block[idx_block + 1](h)
                    hs.append(h)
            else:
                total_res_blocks = self.num_res_blocks
                for idx_block in range(total_res_blocks):
                    h = block[idx_block](hs[-1], temb)
                    hs.append(h)
            if len(block) > total_res_blocks:
                for extra_lay in block[total_res_blocks:]:
                    hs.append(extra_lay(hs[-1]))

        # Middle.
        h = hs[-1]
        for n, lay in enumerate(self.middle):
            if self.has_attention and not (n % 2) == 0:
                h = lay(h)
            else:
                h = lay(h, temb)

        # Upsampling.
        for i in range(len(self.upsampling)):
            block = self.upsampling[i]
            if self.attn_resolutions[self.num_resolutions - 1 - i]:
                total_res_blocks = self.num_res_blocks * 2 + 1
                for idx_block in range(0, total_res_blocks, 2):
                    h = block[idx_block](tf.concat([h, hs.pop()], axis=-1), temb)
                    h = block[idx_block + 1](h)
            else:
                total_res_blocks = self.num_res_blocks + 1
                for idx_block in range(total_res_blocks):
                    h = block[idx_block](tf.concat([h, hs.pop()], axis=-1), temb)
            # Upsample.
            if len(block) > total_res_blocks:
                for extra_lay in block[total_res_blocks:]:
                    h = extra_lay(h)

        # End.
        for lay in self.end:
            h = lay(h)

        if self.create_mask:  # whather creates mask or not

            h = tf.keras.activations.sigmoid(h)
            return tf.multiply(x, h)
        else:
            return h