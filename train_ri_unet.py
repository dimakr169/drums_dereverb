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
import tqdm
from config import Config

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
        self.alpha_bar = tf.linspace(1, 0, self.diffusions_steps + 1)

        # initialize AD2 dataset
        self.tr_dataset, self.val_dataset = self.ad2_data.create_datasets()

        # initiliaze optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(self.train_params.learning_rate),
            beta_1=self.train_params.beta1,
            beta_2=self.train_params.beta2,
            epsilon=1e-9,
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

    def diffusion(self, reverb_mags, clean_mags, alpha_bar):
        def diffusion_step(x):
            return self._diffusion(x[0], x[1], x[2])

        return tf.map_fn(
            fn=diffusion_step,
            elems=[reverb_mags, clean_mags, alpha_bar],
            fn_output_signature=(tf.float32),
        )

    def _diffusion(self, reverb_mag, clean_mag, timestep):

        diffed_mag = timestep * clean_mag + (1 - tf.sqrt(timestep)) * reverb_mag
        return diffed_mag

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
            est_com, _ = self.get_spec_mag(est_ri)
            noised_next_com, _ = self.get_spec_mag(noised_next)
            # calc noise loss
            noise_loss = self.loss(est_com, noised_next_com) * 100  # as a weight
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
        est_com, _ = self.get_spec_mag(est_ri)
        noised_next_com, _ = self.get_spec_mag(noised_next)
        # calc noise loss
        noise_loss = self.loss(est_com, noised_next_com) * 100  # as a weight
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
                    44100,
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
                if self.train_params.gen_val_batch:  # whether generate random batch
                    self.generate_random_batch(epoch)

            else:
                print("No validation loss improvement.")
                patience += 1

            print(f"Time taken for this epoch: {time.time() - start_time:.2f} secs\n")
            print("*******************************")

            if patience > self.train_params.patience:
                print("Terminating the training.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default='data/out_gmd')
    parser.add_argument("--model-name", default="CDiff_RI_gmd")
    parser.add_argument("--gpu", default=0)  # set GPU
    args = parser.parse_args()

    # Activate CUDA if GPU id is given
    physical_devices = tf.config.list_physical_devices("GPU")
    print("Physical_devices:", physical_devices)

    if len(physical_devices) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

#CDiff_RI_gmd 16 epochs ~1.59 Noise Loss