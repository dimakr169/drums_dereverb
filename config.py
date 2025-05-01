# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:01:03 2024

@author: dimak
"""

from backbones.config import Config as ModelConfig
from backbones.noam_scheduler import NoamScheduler
from dataset.config import Config as DataConfig


class TrainConfig:
    """Configuration for training loop."""

    def __init__(self):
        # choose "fixed"  or  "cosine_restart or "warmup_cosine"
        self.lr_policy     = "warmup_cosine"
        self.learning_rate = 5e-5        # base LR  (ignored when cosine sets its own)
        self.warmup_steps  = 2_000          # optional warm‑up inside cosine
        self.restart_epochs = 30            # cosine restart period
        # warmup_cosine
        self.warmup_initial_lr = 1e-8
        self.warmup_epochs = 5
        self.cosine_floor_factor = 1e-6
        # Adam hyper‑params
        self.ema_decay = 0.999
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps   = 1e-8 # 1e-9

        # global params
        self.diffusions_steps = 16 #10 default
        self.epochs = 100
        self.patience = 10
        self.gen_val_batch = True # generate random val batch after epoch

        # path config
        self.log = "./log"
        self.ckpt = "./ckpt"

        # model name
        self.name = "cold_diff_16steps"

    def lr(self):
        """Generate proper learning rate scheduler."""
        mapper = {"noam": NoamScheduler}
        if self.lr_policy == "fixed":
            return self.learning_rate
        if self.lr_policy in mapper:
            return mapper[self.lr_policy](self.learning_rate, **self.lr_params)
        raise ValueError("invalid lr_policy")


class Config:
    """Integrated configuration."""

    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.train = TrainConfig()

    def dump(self):
        """Dump configurations into serializable dictionary."""
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration."""
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes."""
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
