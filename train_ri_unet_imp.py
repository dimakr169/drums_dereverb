# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:22:58 2024

@author: dimak
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_addons as tfa   
import tqdm
from config import Config

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")   # one line, at the very top

from backbones.losses import NormalizedMutualInformationLoss
from backbones.metrics import SISAR, SISDR, SISIR
from backbones.unet import UNet
from dataset.ad2_dataset import AD2


class ColdRIUNetTrainer:
    """RI UNet trainer"""

    def __init__(self, model, pre_params, train_params, ad2_data, output_dir):
        """Initializer.
        Args:
            model: UNet RI
            pre_params: Preprocessing parameters from yaml
            train_params: Training parameters from yaml
            dataset: AD2 dataset
            output_dir: dir for saving ckpt, logs, samples
        """
        self.model = model
        self.pre_params = pre_params
        self.train_params = train_params
        self.ad2_data = ad2_data
        self.output_dir = output_dir

        self.diffusions_steps = self.train_params.diffusions_steps
        # define ranges depending on diffusion steps
        # OLD (linear)
        # self.alpha_bar = tf.linspace(1, 0, self.diffusions_steps + 1)

        # NEW (cosine)
        t = tf.cast(tf.range(self.diffusions_steps + 1), tf.float32)
        self.alpha_bar = tf.cos(0.5 * np.pi * t / self.diffusions_steps) ** 2

        # initialize AD2 dataset
        self.tr_dataset, self.val_dataset = self.ad2_data.create_datasets()

        # ---------- learning‑rate schedule selector -----------------
        steps_per_epoch = len(self.tr_dataset)

        if self.train_params.lr_policy == "fixed":
            lr_schedule = float(self.train_params.learning_rate)

        elif self.train_params.lr_policy == "cosine_restart":

            first_decay = steps_per_epoch * self.train_params.restart_epochs

            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=float(self.train_params.learning_rate),
                first_decay_steps=max(first_decay, 1),
                t_mul=1.0,
                m_mul=0.5,          # halve LR after each restart
                alpha=1e-6
            )

        elif train_params.lr_policy == 'warmup_cosine':
            total_steps  = steps_per_epoch * self.train_params.epochs
            warmup_steps = int(steps_per_epoch * self.train_params.warmup_epochs)

            # 1) Warm-up schedule
            warmup = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.train_params.warmup_initial_lr,
                decay_steps=warmup_steps,
                end_learning_rate=float(self.train_params.learning_rate),
                power=1.0
            )

            # 2) Cosine decay schedule
            cosine = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=float(self.train_params.learning_rate),
                decay_steps=total_steps - warmup_steps,
                alpha=self.train_params.cosine_floor_factor
            )

            # 3) Inline “Concatenate” substitute
            class _WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, warmup, cosine, warmup_steps):
                    self.warmup = warmup
                    self.cosine = cosine
                    self.warmup_steps = warmup_steps
                def __call__(self, step):
                    return tf.where(
                        step < self.warmup_steps,
                        self.warmup(step),
                        self.cosine(step - self.warmup_steps)
                    )
                def get_config(self):
                    return {
                        'warmup_steps': self.warmup_steps,
                        'initial_learning_rate': self.warmup.initial_learning_rate,
                        'end_learning_rate': getattr(self.warmup, 'end_learning_rate', None),
                        'decay_steps': getattr(self.cosine, 'decay_steps', None),
                        'alpha': getattr(self.cosine, 'alpha', None)
                    }

            lr_schedule = _WarmupCosine(warmup, cosine, warmup_steps)

        else:
            raise ValueError("Unknown lr_policy")        

        # base optimizer
        base_opt = tf.keras.optimizers.legacy.Adam(
            learning_rate=lr_schedule,
            beta_1=self.train_params.beta1,
            beta_2=self.train_params.beta2,
            epsilon=self.train_params.eps
        )

        # EMA optimizer
        self.optimizer = tfa.optimizers.MovingAverage(
            base_opt,
            average_decay=train_params.ema_decay,
        )


        # initialize noise and audio loss
        self.loss = tf.keras.losses.MeanAbsoluteError()

        # initialize custom loss
        self.nmi_loss = NormalizedMutualInformationLoss(bins=512)

        # initialize metrics
        self.noise_loss_train = tf.keras.metrics.Mean(name="noise_loss_train")
        self.noise_loss_val = tf.keras.metrics.Mean(name="noise_loss_val")
        self.nmi_loss_train = tf.keras.metrics.Mean(name="nmi_loss_train")
        self.nmi_loss_val = tf.keras.metrics.Mean(name="nmi_loss_val")
        self.audio_loss_train = tf.keras.metrics.Mean(name="audio_loss_train")
        self.audio_loss_val = tf.keras.metrics.Mean(name="audio_loss_val")

        # custom mectrics
        self.sisdr = SISDR(name="si_sdr_val_loss")
        self.sisir = SISIR(name="si_sir_val_loss")
        self.sisar = SISAR(name="si_sar_val_loss")

        # initialize checkpoint path
        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_path = os.path.join(self.output_dir, "checkpoints")

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

        # initialize Tensorboard loggers
        log_path = os.path.join(self.output_dir, "logs")
        train_log_dir = os.path.join(log_path, "train")
        val_log_dir = os.path.join(log_path, "validation")
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def diffusion(self, reverb_ri, clean_ri, noise_level):
        α = tf.reshape(noise_level, [-1,1,1,1])
        return α*clean_ri + (1-α)*reverb_ri

    @tf.function
    def train_step(self, inp_tar_ri):
        # seperate inputs for tf.keras model class

        # see ad2_dataset.py about dimensions
        reverb_ri = inp_tar_ri[0]
        clean_ri = inp_tar_ri[1]

        # calc batch size
        bsize = tf.shape(reverb_ri)[0]

        # select random timesteps (for every example in batch)
        timesteps = tf.random.uniform(
            [bsize], 1, self.diffusions_steps + 1, dtype=tf.int32
        )

        # apply that to get the corresponding alpha value
        noise_level = tf.cast(tf.gather(self.alpha_bar, timesteps), tf.float32)
        noise_level_next = tf.cast(tf.gather(self.alpha_bar, timesteps - 1), tf.float32)

        # apply cold diffusion
        noised = self.diffusion(reverb_ri, clean_ri, noise_level)
        noised_next = self.diffusion(reverb_ri, clean_ri, noise_level_next)

        # call model
        with tf.GradientTape() as tape:
            # calculate noise
            est_ri = self.model([noised, timesteps], training=True)
            #mixed float precision
            est_ri = tf.cast(est_ri, tf.float32)
            # real/imag noise L1
            er, ei = est_ri[...,0], est_ri[...,1]
            tr, ti = noised_next[...,0], noised_next[...,1]
            noise_loss = (self.loss(er, tr) + self.loss(ei, ti)) * 50 #seperate
            # generate audio from predictions
            est_wav = self.get_signal_from_RI_stft(est_ri)
            tar_wav = self.get_signal_from_RI_stft(
                noised_next
            )  # ground truth = noised_next
            # calculate audio loss
            audio_loss = self.loss(est_wav, tar_wav) * 400  # as a weight
            # calc nmi loss
            nmi_loss = self.nmi_loss(tar_wav, est_wav)
            # get combined
            combined_loss = noise_loss + nmi_loss + audio_loss

        # apply gradients
        gradients = tape.gradient(combined_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # apply metrics
        self.noise_loss_train.update_state(noise_loss)
        self.nmi_loss_train.update_state(nmi_loss)
        self.audio_loss_train.update_state(audio_loss)

    @tf.function
    def val_step(self, inp_tar_ri):
        # seperate inputs for tf.keras model class

        reverb_ri = inp_tar_ri[0]
        clean_ri = inp_tar_ri[1]

        # calc batch size
        bsize = tf.shape(reverb_ri)[0]

        # select random timesteps (for every example in batch)
        timesteps = tf.random.uniform(
            [bsize], 1, self.diffusions_steps + 1, dtype=tf.int32
        )

        # apply that to get the corresponding alpha value
        noise_level = tf.cast(tf.gather(self.alpha_bar, timesteps), tf.float32)
        noise_level_next = tf.cast(tf.gather(self.alpha_bar, timesteps - 1), tf.float32)

        # apply cold diffusion
        noised = self.diffusion(reverb_ri, clean_ri, noise_level)
        noised_next = self.diffusion(reverb_ri, clean_ri, noise_level_next)
        # call model and calculate noise
        est_ri = self.model([noised, timesteps], training=False)
        #mixed float precision
        est_ri = tf.cast(est_ri, tf.float32)
        # real/imag noise L1
        er, ei = est_ri[...,0], est_ri[...,1]
        tr, ti = noised_next[...,0], noised_next[...,1]
        noise_loss = (self.loss(er, tr) + self.loss(ei, ti)) * 50 #seperate
        # generate audio from predictions
        est_wav = self.get_signal_from_RI_stft(est_ri)
        tar_wav = self.get_signal_from_RI_stft(
            noised_next
        )  # ground truth = noised_next
        # calculate audio loss
        audio_loss = self.loss(est_wav, tar_wav) * 400  # as a weight
        # calc nmi loss
        nmi_loss = self.nmi_loss(tar_wav, est_wav)

        # apply metrics
        self.noise_loss_val.update_state(noise_loss)
        self.nmi_loss_val.update_state(nmi_loss)
        self.audio_loss_val.update_state(audio_loss)

        # get si metrics for single step diffusion predictions
        self.sisdr.update_state(tar_wav, est_wav)
        self.sisir.update_state(tar_wav, est_wav)
        self.sisar.update_state(tar_wav, est_wav)

    def get_spec_mag(self, ri_stft):

        polar_spec = tf.complex(ri_stft[..., 0], ri_stft[..., 1])
        # get magnitude
        mag = tf.abs(polar_spec)

        return polar_spec, mag

    def get_signal_from_RI_stft(self, ri_stft):
        
        ri_stft = tf.cast(ri_stft, tf.float32)
        polar_spec = tf.complex(ri_stft[..., 0], ri_stft[..., 1])

        signal = tf.signal.inverse_stft(
            polar_spec,
            frame_length=self.pre_params.win,
            frame_step=self.pre_params.hop,
            fft_length=self.pre_params.fft,
            window_fn=tf.signal.inverse_stft_window_fn(
                self.pre_params.hop, forward_window_fn=tf.signal.hann_window
            ),
        )

        return signal

    @tf.function
    def reverse_diffusion(self, inp_ri, step_stop=0):

        # calc batch size
        bsize = tf.shape(inp_ri)[0]

        base = tf.ones([bsize], dtype=tf.int32)
        # step_stop=0 is full reverse diffusion

        # store all steps
        all_diff_steps = []

        for t in range(self.diffusions_steps, step_stop, -1):
            inp_ri = self.model([inp_ri, base * t], training=False)
            all_diff_steps.append(inp_ri)

        return all_diff_steps

    def generate_random_batch(self, epoch):

        # create sample path for each epoch
        epoch_path = "epoch_" + str(epoch)
        out_path = os.path.join(self.output_dir, "samples", epoch_path)
        os.makedirs(out_path, exist_ok=True)

        # take random batch from validation
        random_batch = next(iter(self.val_dataset))

        # seperate batch to inputs
        reverb_ri = random_batch[0]
        clean_ri = random_batch[1]

        # make predictions. call reverse diffusion
        inp_ri = reverb_ri
        all_preds_ri = self.reverse_diffusion(inp_ri)

        # for every pair in random batch
        for i in range(0, reverb_ri.shape[0]):
            # create path
            val_path = os.path.join(out_path, "val_" + str(i))
            os.makedirs(val_path, exist_ok=True)
            # create wavs for input and target
            inp_wav = self.get_signal_from_RI_stft(reverb_ri[i, :])
            sf.write(os.path.join(val_path, "input.wav"), inp_wav.numpy(), pre_params.sr)
            tar_wav = self.get_signal_from_RI_stft(clean_ri[i, :])
            sf.write(os.path.join(val_path, "target.wav"), tar_wav.numpy(), pre_params.sr)

            # returns list with all step predictions
            for t in range(0, len(all_preds_ri)):
                pred_ri = all_preds_ri[t][i, :]
                pred_wav = self.get_signal_from_RI_stft(pred_ri)
                sf.write(
                    os.path.join(val_path, "diffused_" + str(t) + ".wav"),
                    pred_wav.numpy(),
                    pre_params.sr,
                )

    def train(self):

        train_size = len(self.tr_dataset)
        val_size = len(self.val_dataset)
        print(
            "Dataset with", train_size, "training and", val_size, "validation batches"
        )

        # Manual Early Stopping mechanism
        patience = 0
        curr_loss = 99.99

        for epoch in range(self.train_params.epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # reset metrics
            self.noise_loss_train.reset_states()
            self.noise_loss_val.reset_states()
            self.nmi_loss_train.reset_states()
            self.nmi_loss_val.reset_states()
            self.audio_loss_train.reset_states()
            self.audio_loss_val.reset_states()

            self.sisdr.reset_states()
            self.sisir.reset_states()
            self.sisar.reset_states()

            # Training Loop
            with tqdm.tqdm(total=train_size, desc="Training") as pbar:
                for batch, inp_tar in enumerate(self.tr_dataset):
                    self.train_step(inp_tar)
                    pbar.update(1)

                    if epoch == 0 and batch == 0:  # print once the model summary
                        print(self.model.summary())

                    if batch % 300 == 0:
                        print(f"Batch {batch}")
                        print(f"Noise MAE Loss {self.noise_loss_train.result():.4f}")
                        print(f"NMI Loss {self.nmi_loss_train.result():.4f}")
                        print(f"Audio MAE Loss {self.audio_loss_train.result():.4f}")

                        # Writing  metrics and losses to TensorBoard
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar(
                                "Noise MAE Loss",
                                self.noise_loss_train.result(),
                                step=epoch * train_size + batch,
                            )
                            tf.summary.scalar(
                                "NMI Loss",
                                self.nmi_loss_train.result(),
                                step=epoch * train_size + batch,
                            )
                            tf.summary.scalar(
                                "Audio MAE Loss",
                                self.audio_loss_train.result(),
                                step=epoch * train_size + batch,
                            )

            print("----")
            combined_train_loss = np.round(
                self.noise_loss_train.result()
                + self.nmi_loss_train.result()
                + self.audio_loss_train.result(),
                decimals=4,
            )
            print(f"Overal Combined Training Loss {combined_train_loss:.4f}")
            print("----")

            # ---------- swap *in* EMA weights ---------- 
            raw_vars = [v.read_value() for v in self.model.trainable_variables]
            self.optimizer.assign_average_vars(self.model.trainable_variables)

            # Validation Loop
            with tqdm.tqdm(total=len(self.val_dataset), desc="Validation") as pbar:
                for _batch, inp_tar in enumerate(self.val_dataset):
                    self.val_step(inp_tar)
                    pbar.update(1)

            # Writing  metrics and losses to TensorBoard
            with self.val_summary_writer.as_default():
                tf.summary.scalar(
                    "Noise MAE Loss", self.noise_loss_val.result(), step=epoch
                )
                tf.summary.scalar("NMI Loss", self.nmi_loss_val.result(), step=epoch)
                tf.summary.scalar(
                    "Audio MAE Loss", self.audio_loss_val.result(), step=epoch
                )

                tf.summary.scalar("SISDR Loss", self.sisdr.result(), step=epoch)
                tf.summary.scalar("SISIR Loss", self.sisir.result(), step=epoch)
                tf.summary.scalar("SISAR Loss", self.sisar.result(), step=epoch)

            print("----")
            val_loss = np.round(
                self.noise_loss_val.result()
                + self.nmi_loss_val.result()
                + self.audio_loss_val.result(),
                decimals=4,
            )
            print(f"Total Noise MAE Loss {self.noise_loss_val.result():.4f}")
            print(f"Total NMI Loss {self.nmi_loss_val.result():.4f}")
            print(f"Total Audio MAE Loss {self.audio_loss_val.result():.4f}")
            print(f"Overal Combined Validation Loss {val_loss:.4f}")
            print("----")
            print(f"SISDR Loss {self.sisdr.result():.4f}")
            print(f"SISIR Loss {self.sisir.result():.4f}")
            print(f"SISAR Loss {self.sisar.result():.4f}")

            if curr_loss > val_loss:
                # save checkpoint and reset early stopping mechanism
                print("Checkpoint saved.")
                patience = 0
                self.ckpt_manager.save()
                curr_loss = val_loss
                best_loss = val_loss
                if self.train_params.gen_val_batch:  # whether generate random batch
                    self.generate_random_batch(epoch)

            else:
                print("No validation loss improvement.")
                patience += 1

            print(f"Time taken for this epoch: {time.time() - start_time:.2f} secs\n")
            print("*******************************")

            if patience > self.train_params.patience:
                print("Terminating the training.")
                print("Best val loss stopped to", best_loss)             
                break

            # restore raw weights
            for var, raw in zip(self.model.trainable_variables, raw_vars):
                var.assign(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default='data/out_combined')
    parser.add_argument("--model-name", default="CDiff_RI_combined_IMP2")
    parser.add_argument("--gpu", default=2, type=int)  # set GPU
    args = parser.parse_args()

    # 1) List all GPUs
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPUs found, running on CPU")
    else:
        # 2) Pick one GPU to use
        chosen = gpus[args.gpu]
        tf.config.set_visible_devices(chosen, "GPU")

        # 3) Enable memory growth on it
        tf.config.experimental.set_memory_growth(chosen, True)

        print("Using GPU:", chosen)

    # Load config parameters
    params = Config()
    pre_params = params.data
    train_params = params.train
    model_params = params.model

    # Setting name will save logs/samples/checkpoints in its own folder in /model path
    output_dir = f"saved_models/{args.model_name}"

    # Create tf.datasets
    ad2_data = AD2(pre_params, data_dir=args.data_dir)

    # Initialize RI UNet
    ri_unet = UNet(model_params)

    # Initialize Trainer
    trainer = ColdRIUNetTrainer(ri_unet, pre_params, train_params, ad2_data, output_dir)

    # Start training
    trainer.train()

#CDiff_RI_gmd 27 checkpoints ~1.52 Noise Loss

#Diff_RI_combined 17 checkpoints
#Best val loss stopped to 3.1997
#Diff_RI_combined_pre_5e-5
#Best val loss stopped to 3.8514

#IMP Epoch 70 val loss 1.8634 

#IMP2 Epoch 74 val loss 1.3539


