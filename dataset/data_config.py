class DataConfig:
    """Configuration for dataset construction."""

    def __init__(self):
        # audio config
        self.inp_type = "wav"  # 'wav' or 'flac'
        self.sr = 44100  # sample rate

        self.dur = 2  # duration in seconds
        self.context_dur = 10 # split audio inputs into context segments

        self.lufs = -24.0  # for audio normalizing
        self.threshold = 0.0005  # for energy threshold to avoid silent parts

        # Synthetic RIR parameters
        # following paper: https://arxiv.org/abs/2212.11851
        self.t60_r = [0.4, 1.5]  # Range for reverb time in seconds
        self.room_dim_r = [5, 15, 5, 15, 2, 6] # Range 5 to 15 meters length-width, 2 to 6 for height
        self.min_distance_to_wall = 1.0  # for mic and source positions

        # Augmentation
        self.aug_factor = 3 # apply augmentations for each context
