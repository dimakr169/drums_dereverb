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
        # for SGMSE and CDiffuse baseline in_chans = 8 
        self.dropout = 0.1  # for DiT 0.0 for UNet 0.1
        self.continuous_emb = False  # select if time embedding is continuous (only for SGMSE+) or discrete
        self.use_ckpt = True # enable gradient checkpointing inside backbones (VRAM saver)
        self.residual_prediction = True #predicting the delta Δ = x_{t-1} - x_t 
                         #usually optimizes better than predicting the absolute x_{t-1}
                         # False for DDPM baseline

        """UNet (Cold Diffusion)"""
        self.num_res_blocks = 2  # Default: 2
        self.use_attention = True # Apply attention in middle (True or False) False
        self.channels = 64  # 32
        self.ch_mult = (1, 2, 4, 8)   # (1, 2, 4, 4)  (1, 2, 4, 8)
        self.ri_inp = True  # if input is Real/Imaginary (True) or Magnintude (False)
        self.use_norm = True  # Usage of BN or GN layers in Residual blocks
        self.num_groups = 8  # or 8 if out_ch%8==0 else 4 (4)
        self.resample_with_conv = True  # Dowsampling with conv2d



        """TransformerDiffuser (Cold Diffusion)"""
        self.embed_dim = 768  
        self.num_heads = 8
        self.num_layers = 5
        self.max_freq = 1000.0 # use if continuous_emb = True
        self.use_checkpoint = True # Faster training per step but Higher memory footprint. Set True for big DiTs
        self.patch_f = 19 #
        self.patch_t = 9 #9
        self.time_stride = 4 # set to 8 for 50% overlap in time if you want fewer artifacts. Default 5
        self.pos_embed = "sincos_2d" # "sincos_2d" ignored when use_rope=True
        self.use_rope = True   # set True to enable RotaryEmbedding (1D over tokens)
