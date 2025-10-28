# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:28:43 2024

@author: dimak
"""


class Config:
    """Configuration for backbones."""

    def __init__(self):

        """Common for all backbones"""
        self.in_chans = 4          # stereo RI: [L_R, L_I, R_R, R_I]
        self.dropout = 0.1  # Default: 0.2
        self.continuous_emb = False  # select if time embedding is continuous or discrete
        self.use_ckpt = True # enable gradient checkpointing inside backbones (VRAM saver)
        self.residual_prediction = False #predicting the delta Δ = x_{t-1} - x_t 
                         #usually optimizes better than predicting the absolute x_{t-1}

        """UNet (Cold Diffusion)"""
        self.num_res_blocks = 2  # Default: 2
        self.use_attention = False  # Apply attention globally (True or False)
        self.channels = 32  # Default: 16
        self.ch_mult = (1, 2, 4, 4)  # Default: (1, 2, 4, 8, 16, 32, 64)
        self.ri_inp = True  # if input is Real/Imaginary (True) or Magnintude (False)
        self.use_norm = True  # Usage of BN or GN layers in Residual blocks
        self.num_groups = 4  # or 8 if out_ch%8==0 else 4
        self.resample_with_conv = True  # Dowsampling with conv2d



        """TransformerDiffuser (Cold Diffusion)"""
        self.embed_dim = 768
        self.num_heads = 8
        self.num_layers = 4
        self.max_freq = 1000.0 # use if continuous_emb = True
        self.use_checkpoint = False # Faster training per step but Higher memory footprint. Set True for big DiTs
        self.patch_f = 19 #27
        self.patch_t = 11 #23
        self.time_stride = 6 # set to 8 for 50% overlap in time if you want fewer artifacts
        self.pos_embed = "sincos_2d" # ignored when use_rope=True
        self.use_rope = False    # set True to enable RotaryEmbedding (1D over tokens)