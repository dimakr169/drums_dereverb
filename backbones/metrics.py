# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:57:15 2024

@author: dimak
"""

import tensorflow as tf


class SISDR(tf.keras.metrics.Metric):
    def __init__(self, name="si_sdr", **kwargs):
        super(SISDR, self).__init__(name=name, **kwargs)

        # is estimated per batch and is averaged across batches
        self.sisdr_tracker = tf.keras.metrics.Mean(name="sisdr_tracker")

    def update_state(self, reference, estimation, sample_weight=None):
        epsilon = 1e-7

        reference_energy = tf.reduce_sum(reference * reference, axis=-1)
        scale = tf.reduce_sum(reference * estimation, axis=-1) / (
            reference_energy + epsilon
        )

        projection = scale[..., tf.newaxis] * reference
        noise = estimation - projection

        numerator = tf.reduce_sum(projection * projection, axis=-1)
        denominator = tf.reduce_sum(noise * noise, axis=-1)

        divided_value = tf.clip_by_value(
            numerator / (denominator + epsilon),
            clip_value_min=epsilon,
            clip_value_max=tf.float32.max,
        )
        si_sdr = 10 * tf.math.log(divided_value) / tf.math.log(10.0)

        # update the average SISDR estimate
        self.sisdr_tracker.update_state(si_sdr)

    def result(self):
        return self.sisdr_tracker.result()

    def reset_states(self):
        self.sisdr_tracker.reset_state()


class SISIR(tf.keras.metrics.Metric):
    def __init__(self, name="si_sir", **kwargs):
        super(SISIR, self).__init__(name=name, **kwargs)

        # is estimated per batch and is averaged across batches
        self.sisir_tracker = tf.keras.metrics.Mean(name="sisir_tracker")

    def update_state(self, reference, estimated, sample_weight=None):
        epsilon = 1e-7
        ref_power = tf.reduce_sum(tf.square(reference))
        est_power = tf.reduce_sum(tf.square(estimated))

        scaling = tf.sqrt(ref_power / (est_power + epsilon))
        scaled_estimated = scaling * estimated
        interference = reference - scaled_estimated
        interference_power = tf.reduce_sum(tf.square(interference))
        signal_power = ref_power - interference_power

        divided_value = tf.clip_by_value(
            (signal_power + epsilon) / (interference_power + epsilon),
            clip_value_min=epsilon,
            clip_value_max=tf.float32.max,
        )
        si_sir = 10 * tf.math.log(divided_value) / tf.math.log(10.0)

        # update the average SISIR estimate
        self.sisir_tracker.update_state(si_sir)

    def result(self):
        return self.sisir_tracker.result()

    def reset_states(self):
        self.sisir_tracker.reset_state()


class SISAR(tf.keras.metrics.Metric):
    def __init__(self, name="si_sar", **kwargs):
        super(SISAR, self).__init__(name=name, **kwargs)

        # is estimated per batch and is averaged across batches
        self.sisar_tracker = tf.keras.metrics.Mean(name="sisar_tracker")

    def update_state(self, reference, estimated, sample_weight=None):
        epsilon = 1e-7
        ref_power = tf.reduce_sum(tf.square(reference))
        est_power = tf.reduce_sum(tf.square(estimated))

        scaling = tf.sqrt(ref_power / (est_power + epsilon))
        scaled_estimated = scaling * estimated
        artifacts = scaled_estimated - reference
        artifacts_power = tf.reduce_sum(tf.square(artifacts))
        signal_power = ref_power

        divided_value = tf.clip_by_value(
            signal_power / (artifacts_power + epsilon),
            clip_value_min=epsilon,
            clip_value_max=tf.float32.max,
        )
        si_sar = 10 * tf.math.log(divided_value) / tf.math.log(10.0)
        # update the average SISAR estimate

        self.sisar_tracker.update_state(si_sar)

    def result(self):
        return self.sisar_tracker.result()

    def reset_states(self):
        self.sisar_tracker.reset_state()
