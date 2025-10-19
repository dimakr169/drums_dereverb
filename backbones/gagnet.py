import tensorflow as tf


class GateConv2D(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GateConv2D, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.k_t = kernel_size[0]

        # Define the convolutional layer
        self.conv = tf.keras.layers.Conv2D(
            filters=self.out_channels * 2,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="same" if self.k_t == 1 else "valid",
        )

    def call(self, inputs):
        # Permute the inputs to match TensorFlow's dimension order
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])

        if self.k_t > 1:
            padding = [[0, 0], [self.k_t - 1, 0], [0, 0], [0, 0]]
            inputs = tf.pad(inputs, padding, "CONSTANT")

        x = self.conv(inputs)

        # Split the output and apply sigmoid activation to the gate
        outputs, gate = tf.split(x, num_or_size_splits=2, axis=3)

        # Permute the outputs
        return tf.transpose(outputs * tf.sigmoid(gate), perm=[0, 3, 1, 2])


class NormSwitch(tf.keras.layers.Layer):
    def __init__(self, norm_type, dim_size):
        super(NormSwitch, self).__init__()
        if norm_type == "BN":
            self.norm = tf.keras.layers.BatchNormalization(axis=1)
        elif norm_type == "IN":

            # The axis parameter might need adjustment based on the dimensionality (1D or 2D).
            if dim_size == "1D":
                self.norm = tf.keras.layers.LayerNormalization(
                    axis=-1, center=True, scale=True
                )  # [1, 2]
            elif dim_size == "2D":
                self.norm = tf.keras.layers.LayerNormalization(
                    axis=[2, 3], center=True, scale=True
                )  # [1, 2, 3]
            else:
                raise ValueError(
                    "Unsupported dimension size for Instance Normalization: {}".format(
                        dim_size
                    )
                )
        else:
            raise ValueError("Unsupported normalization type: {}".format(norm_type))

    def call(self, inputs):

        out = self.norm(inputs)

        return out


class UNet_Encoder(tf.keras.Model):
    def __init__(self, cin, k1, c, norm_type):
        super(UNet_Encoder, self).__init__()
        k_beg = (2, 5)
        c_end = 24  # 64

        self.unet_blocks = [
            tf.keras.Sequential(
                [
                    GateConv2D(cin, c, k_beg, (1, 2)),  # 1, 3
                    NormSwitch(norm_type, "2D"),
                    tf.keras.layers.PReLU(),
                ]
            ),
            tf.keras.Sequential(
                [
                    GateConv2D(c, c, k1, (1, 2)),
                    NormSwitch(norm_type, "2D"),
                    tf.keras.layers.PReLU(),
                ]
            ),
            # tf.keras.Sequential([
            #    GateConv2D(c, c, k1, (1, 2)),
            #    NormSwitch(norm_type, "2D"),
            #    tf.keras.layers.PReLU()
            # ]),
            tf.keras.Sequential(
                [
                    GateConv2D(c, c, k1, (1, 2)),
                    NormSwitch(norm_type, "2D"),
                    tf.keras.layers.PReLU(),
                ]
            ),
            # Repeat similar blocks for the remaining layers
            tf.keras.Sequential(
                [
                    GateConv2D(c, c_end, k1, (1, 2)),
                    NormSwitch(norm_type, "2D"),
                    tf.keras.layers.PReLU(),
                ]
            ),
        ]

    def call(self, inputs):
        x = inputs
        for block in self.unet_blocks:
            x = block(x)
        return x


class Conv2dunit(tf.keras.layers.Layer):
    def __init__(self, k, c, norm_type):
        super(Conv2dunit, self).__init__()
        self.conv = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(c, k, (1, 3), padding="valid"),  # check padding
                NormSwitch(norm_type, "2D"),
                tf.keras.layers.PReLU(),
            ]
        )

    def call(self, inputs):

        # Permute the inputs from [N, C, H, W] to [N, H, W, C]
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])

        x = self.conv(inputs)

        # Permute the outputs back to [N, C, H, W]
        return tf.transpose(x, perm=[0, 3, 1, 2])


class Deconv2dunit(tf.keras.layers.Layer):
    def __init__(self, k, c, intra_connect, norm_type):
        super(Deconv2dunit, self).__init__()
        self.intra_connect = intra_connect

        # Conditional layer creation based on intra_connect
        if intra_connect == "add":
            self.deconv = self._create_deconv_layer(c, c, k, norm_type)
        elif intra_connect == "cat":
            self.deconv = self._create_deconv_layer(2 * c, c, k, norm_type)
        else:
            raise ValueError("Unsupported intra_connect: {}".format(intra_connect))

    def _create_deconv_layer(self, input_c, output_c, kernel_size, norm_type):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    output_c, kernel_size, strides=(1, 3), padding="valid"
                ),  # check padding
                NormSwitch(norm_type, "2D"),
                tf.keras.layers.PReLU(),
            ]
        )

    def call(self, inputs, enc_dims):
        # Permute input
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])

        x = self.deconv(inputs)

        # check if pad or cropping is needed in the width dimension
        current_width = x.shape[2]
        target_width = enc_dims

        delta_w = target_width - current_width

        if delta_w > 0:
            # Pad if output is smaller than target
            paddings = [
                [0, 0],
                [0, 0],
                [max(delta_w // 2, 0), max(delta_w - delta_w // 2, 0)],
                [0, 0],
            ]
            x = tf.pad(x, paddings, "CONSTANT")
        elif delta_w < 0:
            # Crop if output is larger than target
            crop_w_start = abs(delta_w) // 2
            x = x[:, :, crop_w_start : crop_w_start + target_width, :]

        # Permute the outputs back to [N, C, H, W]
        return tf.transpose(x, perm=[0, 3, 1, 2])


class Skip_connect(tf.keras.layers.Layer):
    def __init__(self, connect):
        super(Skip_connect, self).__init__()
        self.connect = connect

    def call(self, x_main, x_aux):
        if self.connect == "add":
            return x_main + x_aux
        elif self.connect == "cat":
            return tf.concat([x_main, x_aux], axis=1)
        else:
            raise ValueError("Unsupported connection type: {}".format(self.connect))


class En_unet_module(tf.keras.layers.Layer):
    def __init__(self, cin, cout, k1, k2, intra_connect, norm_type, scale):
        super(En_unet_module, self).__init__()
        self.scale = scale

        # Initial Convolution Block
        self.in_conv = tf.keras.Sequential(
            [
                GateConv2D(cin, cout, k1, (1, 3)),
                NormSwitch(norm_type, "2D"),
                tf.keras.layers.PReLU(),
            ]
        )

        # Encoder Blocks
        self.enco = [Conv2dunit(k2, cout, norm_type) for _ in range(scale)]

        # Decoder Blocks
        self.deco = [
            Deconv2dunit(k2, cout, "add" if i == 0 else intra_connect, norm_type)
            for i in range(scale)
        ]

        # Skip Connection
        self.skip_connect = Skip_connect(intra_connect)

    def call(self, inputs):
        # store encoder's dimension
        x_dims = (
            []
        )  # keep the width dimension for reconstruction and if pad/crop should occur

        x_resi = self.in_conv(inputs)
        x_dims.append(x_resi.shape[-1])

        x = x_resi
        x_list = []
        # Encoding
        for i in range(0, len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)
            if i != len(self.enco) - 1:
                x_dims.append(x.shape[-1])  # no need to append

        # Decoding with Skip Connections
        for i in range(0, len(self.deco)):
            if i == 0:
                x = self.deco[i](x, x_dims[-(i + 1)])
            else:
                x_con = self.skip_connect(x, x_list[-(i + 1)])
                x = self.deco[i](x_con, x_dims[-(i + 1)])

        x_resi = x_resi + x
        return x_resi


class U2Net_Encoder(tf.keras.Model):
    def __init__(self, cin, k1, k2, c, intra_connect, norm_type):
        super(U2Net_Encoder, self).__init__()
        k_beg = (2, 5)
        c_end = 24

        # Meta U-Net blocks
        self.meta_unet_list = [
            En_unet_module(cin, c, k_beg, k2, intra_connect, norm_type, scale=3),
            # En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=3),
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=2),
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=1),
        ]

        # Final Convolution Block
        self.last_conv = tf.keras.Sequential(
            [
                GateConv2D(c, c_end, k1, (1, 2)),
                NormSwitch(norm_type, "2D"),
                tf.keras.layers.PReLU(),
            ]
        )

    def call(self, inputs):
        x = inputs
        for meta_unet in self.meta_unet_list:
            x = meta_unet(x)
        x = self.last_conv(x)
        return x


class SqueezedTCM(tf.keras.layers.Layer):
    def __init__(self, kd1, cd1, d_feat, dilation, is_causal, norm_type):
        super(SqueezedTCM, self).__init__()
        self.in_conv = tf.keras.layers.Conv1D(cd1, 1, use_bias=False)

        if is_causal:
            padding = "causal"
        else:
            padding = "same"

        self.d_conv = tf.keras.Sequential(
            [
                tf.keras.layers.PReLU(),
                NormSwitch(norm_type, "1D"),
                tf.keras.layers.Conv1D(
                    cd1, kd1, dilation_rate=dilation, padding=padding, use_bias=False
                ),
            ]
        )

        self.out_conv = tf.keras.Sequential(
            [
                tf.keras.layers.PReLU(),
                NormSwitch(norm_type, "1D"),
                tf.keras.layers.Conv1D(d_feat, 1, use_bias=False),
            ]
        )

    def call(self, x):
        resi = x
        x = self.in_conv(x)
        x = self.d_conv(x)
        x = self.out_conv(x)
        return x + resi


class SqueezedTCNGroup(tf.keras.layers.Layer):
    def __init__(self, kd1, cd1, d_feat, dilas, is_causal, norm_type):
        super(SqueezedTCNGroup, self).__init__()
        self.tcns = [
            SqueezedTCM(kd1, cd1, d_feat, dilation, is_causal, norm_type)
            for dilation in dilas
        ]

    def call(self, x):
        for tcn in self.tcns:
            x = tcn(x)
        return x


class GlanceBlock(tf.keras.layers.Layer):
    def __init__(
        self, kd1, cd1, d_feat, p, dilas, fft_num, is_causal, acti_type, norm_type
    ):
        super(GlanceBlock, self).__init__()

        self.in_conv_main = tf.keras.layers.Conv1D(
            d_feat, 1
        )  # , input_shape=(None, ci)
        self.in_conv_gate = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(d_feat, 1),  # , input_shape=(None, ci)
                tf.keras.layers.Activation("sigmoid"),
            ]
        )

        self.tcn_g = tf.keras.Sequential(
            [
                SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type)
                for _ in range(p)
            ]
        )

        if acti_type not in ["sigmoid", "tanh", "relu"]:
            raise RuntimeError("Invalid activation type: {}".format(acti_type))
        self.linear_g = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(fft_num // 2 + 1, 1),
                tf.keras.layers.Activation(acti_type),
            ]
        )

    def call(self, feat_x, pre_x):
        b_size, _, freq_num, seq_len = pre_x.shape

        pre_x = tf.reshape(pre_x, (b_size, -1, seq_len))
        inpt = tf.concat([feat_x, pre_x], axis=1)

        inpt = tf.transpose(inpt, perm=[0, 2, 1])
        x = self.in_conv_main(inpt) * self.in_conv_gate(inpt)
        # x = tf.transpose(x, perm=[0, 2, 1])
        x = self.tcn_g(x)
        gain = self.linear_g(x)
        # transpose back
        gain = tf.transpose(gain, perm=[0, 2, 1])

        return gain


class GazeBlock(tf.keras.layers.Layer):
    def __init__(
        self, kd1, cd1, d_feat, p, dilas, fft_num, is_causal, is_squeezed, norm_type
    ):
        super(GazeBlock, self).__init__()
        self.in_conv_main = tf.keras.layers.Conv1D(
            d_feat, 1
        )  # , input_shape=(None, ci)
        self.in_conv_gate = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(d_feat, 1),  # , input_shape=(None, ci)
                tf.keras.layers.Activation("sigmoid"),
            ]
        )
        self.is_squeezed = is_squeezed

        if not is_squeezed:
            self.tcm_r = tf.keras.Sequential(
                [
                    SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type)
                    for _ in range(p)
                ]
            )
            self.tcm_i = tf.keras.Sequential(
                [
                    SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type)
                    for _ in range(p)
                ]
            )
        else:
            self.tcm_ri = tf.keras.Sequential(
                [
                    SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type)
                    for _ in range(p)
                ]
            )

        self.linear_r = tf.keras.layers.Conv1D(fft_num // 2 + 1, 1)
        self.linear_i = tf.keras.layers.Conv1D(fft_num // 2 + 1, 1)

    def call(self, feat_x, pre_x):
        b_size, _, freq_num, seq_len = pre_x.shape

        pre_x = tf.reshape(pre_x, (b_size, -1, seq_len))
        inpt = tf.concat([feat_x, pre_x], axis=1)

        inpt = tf.transpose(inpt, perm=[0, 2, 1])
        x = self.in_conv_main(inpt) * self.in_conv_gate(inpt)

        if not self.is_squeezed:
            x_r = self.tcm_r(x)
            x_i = self.tcm_i(x)
        else:
            x = self.tcm_ri(x)
            x_r, x_i = x, x

        x_r = self.linear_r(x_r)
        x_i = self.linear_i(x_i)
        x_r = tf.transpose(x_r, perm=[0, 2, 1])
        x_i = tf.transpose(x_i, perm=[0, 2, 1])

        return tf.stack([x_r, x_i], axis=1)


class GlanceGazeModule(tf.keras.Model):
    def __init__(
        self,
        kd1,
        cd1,
        d_feat,
        p,
        dilas,
        fft_num,
        is_causal,
        is_squeezed,
        acti_type,
        norm_type,
    ):
        super(GlanceGazeModule, self).__init__()

        self.glance_block = GlanceBlock(
            kd1, cd1, d_feat, p, dilas, fft_num, is_causal, acti_type, norm_type
        )

        self.gaze_block = GazeBlock(
            kd1, cd1, d_feat, p, dilas, fft_num, is_causal, is_squeezed, norm_type
        )

    def call(self, feat_x, pre_x):
        gain_filter = self.glance_block(feat_x, pre_x)
        com_resi = self.gaze_block(feat_x, pre_x)

        # Computing the magnitude and phase of pre_x
        pre_mag = tf.norm(pre_x, axis=1)
        pre_phase = tf.math.atan2(pre_x[:, -1, ...], pre_x[:, 0, ...])

        # Apply gain filter
        filtered_x = pre_mag * gain_filter
        coarse_x = tf.stack(
            [filtered_x * tf.math.cos(pre_phase), filtered_x * tf.math.sin(pre_phase)],
            axis=1,
        )

        # Combine with complex residual
        x = coarse_x + com_resi
        return x


@tf.keras.utils.register_keras_serializable()
class GaGNet(tf.keras.Model):
    def __init__(self, config, fft_num):
        super(GaGNet, self).__init__()
        self.config = config
        self.fft_num = fft_num

        self.cin = self.config.cin
        self.k1 = self.config.k1
        self.k2 = self.config.k2
        self.c = self.config.c
        self.kd1 = self.config.kd1
        self.cd1 = self.config.cd1
        self.d_feat = self.config.d_feat
        self.p = self.config.p
        self.q = self.config.q
        self.dilas = self.config.dilas
        self.is_u2 = self.config.is_u2
        self.is_causal = self.config.is_causal
        self.is_squeezed = self.config.is_squeezed
        self.acti_type = self.config.acti_type
        self.intra_connect = self.config.intra_connect
        self.norm_type = self.config.norm_type

        # Choose between U2Net_Encoder or UNet_Encoder
        if self.is_u2:
            self.en = U2Net_Encoder(
                self.cin, self.k1, self.k2, self.c, self.intra_connect, self.norm_type
            )
        else:
            self.en = UNet_Encoder(self.cin, self.k1, self.c, self.norm_type)

        # Create a list of GlanceGazeModules
        self.gags = [
            GlanceGazeModule(
                self.kd1,
                self.cd1,
                self.d_feat,
                self.p,
                self.dilas,
                self.fft_num,
                self.is_causal,
                self.is_squeezed,
                self.acti_type,
                self.norm_type,
            )
            for _ in range(self.q)
        ]

    def call(self, inpt):
        if len(inpt.shape) == 3:
            inpt = tf.expand_dims(inpt, axis=-1)

        b_size, _, seq_len, _ = inpt.shape
        feat_x = self.en(inpt)

        x = tf.transpose(feat_x, perm=[0, 1, 3, 2])

        x = tf.reshape(x, [b_size, -1, seq_len])
        pre_x = tf.transpose(inpt, perm=[0, 1, 3, 2])

        out_list = []
        for gag in self.gags:
            tmp = gag(x, pre_x)
            pre_x = tmp
            out_list.append(tmp)

        return out_list
