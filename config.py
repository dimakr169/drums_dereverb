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
        # optimizer
        self.lr_policy = "fixed"
        self.learning_rate = 1e-4
        # self.lr_policy = 'noam'
        # self.learning_rate = 1
        # self.lr_params = {
        #     'warmup_steps': 4000,
        #     'channels': 64
        # }
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.eps = 1e-9

        # global params
        self.diffusions_steps = 10
        self.epochs = 200
        self.patience = 6
        self.gen_val_batch = True # generate random val batch after epoch

        # path config
        self.log = "./log"
        self.ckpt = "./ckpt"

        # model name
        self.name = "cold_diff_10steps"

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
