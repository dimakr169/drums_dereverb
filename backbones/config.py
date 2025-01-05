# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:28:43 2024

@author: dimak
"""


class Config:
    """Configuration for backbones."""

    def __init__(self):

        """UNet (Cold Diffusion)"""
        self.num_res_blocks = 2  # Default: 2
        self.attn_resolutions = (
            False,
            False,
            False,
            False,
        )  # whether to apply attention to each ch_mult
        self.channels = 24  # Default: 16
        self.ch_mult = (1, 2, 4, 8)  # Default: (1, 2, 4, 8, 16, 32, 64)
        self.dropout = 0.1  # Default: 0.2
        self.ri_inp = True  # if input is Magnitude only (1) or Real/Imaginary (2)
        self.use_bn = False  # Usage of BN layers in Residual blocks
        self.resample_with_conv = False  # Dowsampling with conv2d
        self.create_mask = False  # wether to create a mask to apply for othe input
        self.continuous_emb = (
            False  # select if time embedding is continuous or discrete
        )

        """GaGNet (Predictive)"""
        self.cin = 2  # number of inputs. 2 for Real and Imaginary
        self.k1 = (2, 3)  # k1: kernel size of 2-D GLU, (2, 3) by default
        self.k2 = (1, 3)  # k2: kernel size of the UNet-block, (1, 3) by default
        self.c = 24  # c: channels of the 2-D Convs, 64 by default
        self.kd1 = (
            3  # kd1: kernel size of the dilated convs in the squeezedTCM, 3 by default
        )
        self.cd1 = (
            24  # cd1: channels of the dilated convs in the squeezedTCM, 64 by default
        )
        self.d_feat = 64  # d_feat: channels in the regular 1-D convs, 256 by default
        self.p = 1  # p: number of SqueezedTCMs within a group, 2 by default
        self.q = 2  # q: number of GGMs, 3 by default
        self.dilas = [1, 2, 5, 9]  # dilas: dilation rates, [1, 2, 5, 9] by default
        self.is_u2 = True  # is_u2: whether U^{2} Encoder is set, True by default
        self.is_causal = True  # is_causal: whether causal setting, True by default
        self.is_squeezed = False  # is_squeezed: whether to squeeze the complex residual modeling path, False by default
        self.acti_type = (
            "sigmoid"  # the activation type in the glance block, "sigmoid" by default
        )
        self.intra_connect = "cat"  # intra_connect: skip-connection type within the UNet-block , "cat" by default
        self.norm_type = "IN"  # norm_type: "IN" by default or 'BN'
