# Cold Diffusion for stereo Drums Dereverberation
## Create dataset
- dataset/prepare_audio_stereo.py: Creates pairs of revereberant - anechoic audio stereo wavs, 2 seconds, 44.1KHz.

## Backbones
Both diffusers works on stereo RI spectrograms
- backbones/unet_stereo.py: 54,6M params
- backbones/dit_stereo.py: 57,2M params

## Trainer configs
- train_ri_unet_pt.py: Trains Cold UNet in Δnorm residual and Direct modes.
- train_ri_dit_pt.py: Trains Cold DiT in Δnorm residual mode. It may not run properly in envs with PyTorch <2.9