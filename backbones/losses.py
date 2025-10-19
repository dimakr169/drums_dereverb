# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:57:04 2024

@author: dimak
"""

import tensorflow as tf
import tensorflow_probability as tfp


class NormalizedEntropyLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        bins=256,
        window_length=1024,
        hop_length=512,
        use_dB_scale=False,
        name="normalized_entropy_loss",
    ):
        super().__init__(name=name)
        self.bins = bins
        self.window_length = window_length
        self.hop_length = hop_length
        self.use_dB_scale = use_dB_scale

    def _to_dB_scale(self, signal):
        # Convert signal to decibels, adding a small constant to avoid log(0)
        return (
            20
            * tf.math.log(tf.maximum(signal, 1e-10))
            / tf.math.log(tf.constant(10, dtype=tf.float32))
        )

    def _compute_entropy(self, signal):
        # Normalize the signal
        signal_norm = (signal - tf.reduce_mean(signal)) / tf.math.reduce_std(signal)
        # Compute histograms
        hist = tf.histogram_fixed_width(
            signal_norm, [-1, 1], nbins=self.bins, dtype=tf.int32
        )
        # Convert histograms to probability distributions
        prob = hist / tf.reduce_sum(hist)
        # Calculate entropy
        entropy = -tf.reduce_sum(tf.where(prob > 0, prob * tf.math.log(prob), 0))
        return tf.cast(entropy, tf.float32)

    def call(self, y_true, y_pred):
        if self.use_dB_scale:
            y_true = self._to_dB_scale(y_true)
            y_pred = self._to_dB_scale(y_pred)

        # Apply windowing with overlap
        windows_true = tf.signal.frame(
            y_true, self.window_length, self.hop_length, pad_end=True
        )
        windows_pred = tf.signal.frame(
            y_pred, self.window_length, self.hop_length, pad_end=True
        )

        # Calculate entropy for each window and average
        entropy_true = tf.map_fn(self._compute_entropy, windows_true, dtype=tf.float32)
        entropy_pred = tf.map_fn(self._compute_entropy, windows_pred, dtype=tf.float32)

        # Return the average absolute difference in entropy across all windows
        return tf.reduce_mean(tf.abs(entropy_true - entropy_pred))


class NormalizedMutualInformationLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        bins=256,
        window_length=1024,
        hop_length=512,
        use_dB_scale=False,
        name="normalized_mutual_information",
    ):
        super().__init__(name=name)
        self.bins = bins
        self.window_length = window_length
        self.hop_length = hop_length
        self.use_dB_scale = use_dB_scale

    def _to_dB_scale(self, signal):
        # Convert signal to decibels, adding a small constant to avoid log(0)
        return (
            20
            * tf.math.log(tf.maximum(signal, 1e-10))
            / tf.math.log(tf.constant(10, dtype=tf.float32))
        )

    def _discretize_signal(self, signal):
        # Discretize signal into bins
        bin_edges = tf.linspace(
            tf.reduce_min(signal), tf.reduce_max(signal), self.bins + 1
        )
        bin_indices = tfp.stats.find_bins(signal, bin_edges)
        return tf.cast(bin_indices, tf.int32)

    """
    def _calculate_entropy(self, labels):
        # Calculate the entropy of labels
        labels = tf.reshape(labels, [-1])
        _, _, counts = tf.unique_with_counts(labels)
        probabilities = counts / tf.reduce_sum(counts)
        entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-10))
        return tf.cast(entropy, tf.float32)
    """

    def _calculate_entropy(self, labels):
        # Compute the histogram of the discretized signal across the specified bins.
        hist = tf.histogram_fixed_width(
            labels, [0, self.bins - 1], nbins=self.bins, dtype=tf.int32
        )
        # Convert histogram to probability distribution
        probabilities = hist / tf.reduce_sum(hist)
        # Calculate entropy
        entropy = -tf.reduce_sum(
            tf.where(
                probabilities > 0, probabilities * tf.math.log(probabilities + 1e-10), 0
            )
        )
        return tf.cast(entropy, tf.float32)

    def _create_contingency_matrix(self, labels_true, labels_pred):
        # Create a contingency matrix
        indices = labels_true * self.bins + labels_pred
        counts = tf.math.bincount(indices, minlength=self.bins * self.bins)
        contingency_matrix = tf.reshape(counts, (self.bins, self.bins))
        return contingency_matrix

    def _calculate_mutual_information(self, contingency):
        # Calculate mutual information from the contingency matrix
        contingency = tf.cast(contingency, tf.float32)
        total = tf.reduce_sum(contingency)
        P_ij = contingency / total
        P_i = tf.reduce_sum(P_ij, axis=1, keepdims=True)
        P_j = tf.reduce_sum(P_ij, axis=0, keepdims=True)
        MI = tf.reduce_sum(P_ij * tf.math.log(P_ij / (P_i * P_j + 1e-10) + 1e-10))
        return MI

    def call(self, y_true, y_pred):
        if self.use_dB_scale:
            y_true = self._to_dB_scale(y_true)
            y_pred = self._to_dB_scale(y_pred)

        # Apply windowing with overlap
        windows_true = tf.signal.frame(
            y_true, self.window_length, self.hop_length, pad_end=True
        )
        windows_pred = tf.signal.frame(
            y_pred, self.window_length, self.hop_length, pad_end=True
        )

        def process_window(args):
            window_true, window_pred = args
            labels_true = self._discretize_signal(window_true)
            labels_pred = self._discretize_signal(window_pred)
            contingency_matrix = self._create_contingency_matrix(
                labels_true, labels_pred
            )
            mi = self._calculate_mutual_information(contingency_matrix)
            entropy_true = self._calculate_entropy(labels_true)
            entropy_pred = self._calculate_entropy(labels_pred)
            nmi = 2 * mi / (entropy_true + entropy_pred + 1e-10)
            return nmi

        # Calculate MI for each window and average
        mutual_info = tf.map_fn(
            process_window, (windows_true, windows_pred), dtype=tf.float32
        )
        # TODO see why with tf.vectorized_map is failing

        # NMI is a score to maximize, we subtract from 1.
        return 1 - tf.reduce_mean(mutual_info)
