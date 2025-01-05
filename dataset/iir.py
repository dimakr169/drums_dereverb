# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:04:57 2022

@author: dimos.makris
"""
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.stats import truncnorm


def get_random_eq_values(prob, rate):

    # create dict to store ground-truth parameters
    eq_gt = {
        "Low_Shelf": {},
        "Body": {},
        "Nasal": {},
        "Presence1": {},
        "Presence2": {},
        "High_Shelf": {},
    }

    filter_acts = [
        get_filter_act(prob),
        get_filter_act(prob),
        get_filter_act(prob),  # 75% chance for activated filters
        get_filter_act(prob),
        get_filter_act(prob),
        get_filter_act(prob),
    ]

    # filter_pos_gains = [get_filter_act(65),get_filter_act(65), #65% chance for having positive gain
    #               get_filter_act(65),get_filter_act(65)]

    """Low Shelf"""
    # put conditions to avoid close values to be selected between consecutive filters
    freq_0 = random.randrange(50, 120, 1)
    gain_normal = get_truncated_normal(mean=0.0, sd=3.0, low=-6.0, upp=6.0)
    if filter_acts[0]:
        gain_0 = round(gain_normal.rvs(), 1)
        # if p[0]:
        #    gain_1 = abs(gain_1) #make it positive
    else:
        gain_0 = 0.0
    q_0 = 0.707  # fixed
    # store them to load it to PEQ
    eq_gt["Low_Shelf"] = [gain_0, q_0, freq_0]

    """Band 1"""
    # put conditions to avoid close values to be selected between consecutive filters
    freq10_diff = 120 - freq_0

    if freq10_diff < 60:
        freq_1 = random.randrange(50 + 120 - freq10_diff, 400, 1)
    else:
        freq_1 = random.randrange(120, 400, 1)

    gain_normal = get_truncated_normal(mean=0.0, sd=5.0, low=-10.0, upp=10.0)
    if filter_acts[1]:
        gain_1 = round(gain_normal.rvs(), 1)
        # if p[0]:
        #    gain_1 = abs(gain_1) #make it positive
    else:
        gain_1 = 0.0
    q_normal = get_truncated_normal(mean=1.5, sd=2.5, low=0.5, upp=3.5)  # v1
    q_1 = round(q_normal.rvs(), 1)
    # store them to load it to PEQ
    eq_gt["Body"] = [gain_1, q_1, freq_1]

    """Band 2"""
    # put conditions to avoid close values to be selected between consecutive filters
    freq21_diff = 400 - freq_1

    if freq21_diff < 150:
        freq_2 = random.randrange(400 + 150 - freq21_diff, 1000, 1)
    else:
        freq_2 = random.randrange(400, 1000, 1)
    gain_normal = get_truncated_normal(mean=0.0, sd=5.0, low=-10.0, upp=10.0)
    if filter_acts[2]:
        gain_2 = round(gain_normal.rvs(), 1)
        # if p[1]:
        #    gain_2 = abs(gain_2) #make it positive
    else:
        gain_2 = 0.0
    # q_normal = get_truncated_normal(mean=1.5, sd=2.5, low=0.1, upp=3.5) #v1
    q_normal = get_truncated_normal(mean=1.5, sd=2.5, low=0.5, upp=3.5)  # v1
    q_2 = round(q_normal.rvs(), 1)
    eq_gt["Nasal"] = [gain_2, q_2, freq_2]

    # put conditions to avoid close values to be selected between consecutive filters
    freq32_diff = 1000 - freq_2

    """Band 3"""
    if freq32_diff < 300:
        freq_3 = random.randrange(1000 + 300 - freq32_diff, 2200, 1)
    else:
        freq_3 = random.randrange(1000, 2200, 1)
    gain_normal = get_truncated_normal(mean=0.0, sd=5.0, low=-10.0, upp=10.0)
    if filter_acts[3]:
        gain_3 = round(gain_normal.rvs(), 1)
        # if p[2]:
        #   gain_3 = abs(gain_3) #make it positive
    else:
        gain_3 = 0.0
    # q_normal = get_truncated_normal(mean=1.5, sd=2.5, low=0.1, upp=3.5) #v1
    q_normal = get_truncated_normal(mean=1.5, sd=2.5, low=0.5, upp=3.5)  # v1
    q_3 = round(q_normal.rvs(), 1)
    eq_gt["Presence1"] = [gain_3, q_3, freq_3]

    # put conditions to avoid close values to be selected between consecutive filters
    freq43_diff = 2200 - freq_3

    """Band 4"""
    if freq43_diff < 600:
        freq_4 = random.randrange(600 + 2200 - freq43_diff, 5000, 1)
    else:
        freq_4 = random.randrange(2200, 5000, 1)
    gain_normal = get_truncated_normal(mean=0.0, sd=5.0, low=-10.0, upp=10.0)
    if filter_acts[4]:
        gain_4 = round(gain_normal.rvs(), 1)
        # if p[3]:
        #   gain_4 = abs(gain_4) #make it positive
    else:
        gain_4 = 0.0
    # q_normal = get_truncated_normal(mean=0.707, sd=1.0, low=0.1, upp=2.0) #v1
    q_normal = get_truncated_normal(mean=1.5, sd=2.5, low=0.5, upp=3.5)  # v1
    q_4 = round(q_normal.rvs(), 1)
    eq_gt["Presence2"] = [gain_4, q_4, freq_4]

    freq54_diff = 5000 - freq_4

    """High Shelf"""
    if freq54_diff < 1500:
        freq_5 = random.randrange(1500 + 5000 - freq54_diff, 10000, 1)
    else:
        freq_5 = random.randrange(5000, 10000, 1)
    gain_normal = get_truncated_normal(mean=0.0, sd=4.0, low=-6.0, upp=6.0)
    if filter_acts[5]:
        gain_5 = round(gain_normal.rvs(), 1)
        # if p[3]:
        #   gain_4 = abs(gain_4) #make it positive
    else:
        gain_5 = 0.0
    # q_normal = get_truncated_normal(mean=0.707, sd=1.0, low=0.1, upp=2.0) #v1
    q_normal = get_truncated_normal(mean=0.707, sd=1.0, low=0.1, upp=1.5)  # v1
    q_5 = round(q_normal.rvs(), 1)
    eq_gt["High_Shelf"] = [gain_5, q_5, freq_5]

    # Set PEQ (6 bands)
    peq = PEQ(
        eq_gt["Low_Shelf"],
        eq_gt["Body"],
        eq_gt["Nasal"],
        eq_gt["Presence1"],
        eq_gt["Presence2"],
        eq_gt["High_Shelf"],
        rate,
    )

    return peq, eq_gt


def get_filter_act(perc):

    r = random.randrange(0, 100, 1)

    if r <= perc:
        is_ok = True
    else:
        is_ok = False

    return is_ok


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class PEQ(object):

    """6-band parametric EQ.
    Hi pass -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf"""

    # [40-160]  [160-600] [600-1200] [1200-2500] [2500-5000] [5000-10000]

    def __init__(self, eq0, eq1, eq2, eq3, eq4, eq5, rate):

        self.g0 = eq0[0]
        self.q0 = eq0[1]  # fixed 0.707
        self.fc0 = eq0[2]
        self.g1 = eq1[0]
        self.q1 = eq1[1]
        self.fc1 = eq1[2]
        self.g2 = eq2[0]
        self.q2 = eq2[1]
        self.fc2 = eq2[2]
        self.g3 = eq3[0]
        self.q3 = eq3[1]
        self.fc3 = eq3[2]
        self.g4 = eq4[0]
        self.q4 = eq4[1]
        self.fc4 = eq4[2]
        self.g5 = eq5[0]
        self.q5 = eq5[1]
        self.fc5 = eq5[2]
        self.rate = rate

        self.low_shelf = IIRfilter(
            self.g0, self.q0, self.fc0, self.rate, "low_shelf"
        )  # Body Low Cut
        self.band1 = IIRfilter(self.g1, self.q1, self.fc1, self.rate, "peaking")  # Body
        self.band2 = IIRfilter(self.g2, self.q2, self.fc2, self.rate, "peaking")  # Mud
        self.band3 = IIRfilter(
            self.g3, self.q3, self.fc3, self.rate, "peaking"
        )  # Presence1
        self.band4 = IIRfilter(
            self.g4, self.q4, self.fc4, self.rate, "peaking"
        )  # Presence2
        self.high_shelf = IIRfilter(
            self.g5, self.q5, self.fc5, self.rate, "high_shelf"
        )  # Air

    def freqz(self, b, a, n_fft: int = 512):

        # B = torch.fft.rfft(b, n_fft)
        B = np.fft.rfft(b, n_fft)
        # A = torch.fft.rfft(a, n_fft)
        A = np.fft.rfft(a, n_fft)

        H = B / A

        return H

    def freq_domain_filter(self, x, H, n_fft):

        # X = torch.fft.rfft(x, n_fft)
        X = np.fft.rfft(x, n_fft)

        Y = X * H

        # y = torch.fft.irfft(Y, n_fft)
        y = np.fft.irfft(Y, n_fft)

        return y

    def approx_iir_filter_cascade(self, b_lst, a_lst, x):
        """Apply a cascade of IIR filters.
        Args:
            b (list[Tensor]): List of tensors of shape (3)
            a (list[Tensor]): List of tensors of (3)
            x (torch.Tensor): 1d Tensor.
        """

        if len(b_lst) != len(a_lst):
            raise RuntimeError(
                f"Must have same number of coefficients. Got b: {len(b_lst)} and a: {len(a_lst)}."
            )

        # round up to nearest power of 2 for FFT
        n_fft = 2 ** math.ceil(math.log2(x.shape[-1] + x.shape[-1] - 1))
        # n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
        # n_fft = n_fft.int()

        # this could be done in parallel
        # b = torch.stack(b_s, dim=0).type_as(x)
        b = np.stack(b_lst)
        # a = torch.stack(a_s, dim=0).type_as(x)
        a = np.stack(a_lst)

        H = self.freqz(b, a, n_fft=n_fft)
        # H = torch.prod(H, dim=0).view(-1)
        H = np.prod(H, axis=0)

        # apply filter
        y = self.freq_domain_filter(x, H, n_fft)

        # crop
        y = y[: x.shape[-1]]

        return y

    def apply_eq(self, x):

        """Args:
        x: 1d signal."""

        # Generate coefficients and store them all to lists
        b0, a0 = self.low_shelf.generate_coefficients()
        b1, a1 = self.band1.generate_coefficients()
        b2, a2 = self.band2.generate_coefficients()
        b3, a3 = self.band3.generate_coefficients()
        b4, a4 = self.band4.generate_coefficients()
        b5, a5 = self.high_shelf.generate_coefficients()

        b_lst = [b0, b1, b2, b3, b4, b5]
        a_lst = [a0, a1, a2, a3, a4, a5]

        # apply filters
        y = self.approx_iir_filter_cascade(b_lst, a_lst, x)

        return y

    def plot_eq_response(self, save_path):

        # Generate coefficients and store them all to lists
        b0, a0 = self.low_shelf.generate_coefficients()
        b1, a1 = self.band1.generate_coefficients()
        b2, a2 = self.band2.generate_coefficients()
        b3, a3 = self.band3.generate_coefficients()
        b4, a4 = self.band4.generate_coefficients()
        b5, a5 = self.high_shelf.generate_coefficients()

        self.w_low_shelf, self.h_low_shelf = scipy.signal.freqz(b=b0, a=a0, worN=15000)
        self.w_band1, self.h_band1 = scipy.signal.freqz(b=b1, a=a1, worN=15000)
        self.w_band2, self.h_band2 = scipy.signal.freqz(b=b2, a=a2, worN=15000)
        self.w_band3, self.h_band3 = scipy.signal.freqz(b=b3, a=a3, worN=15000)
        self.w_band4, self.h_band4 = scipy.signal.freqz(b=b4, a=a4, worN=15000)
        self.w_high_shelf, self.h_high_shelf = scipy.signal.freqz(
            b=b5, a=a5, worN=15000
        )

        self.h_low_shelf_db = 20 * np.log10(abs(self.h_low_shelf))
        self.h_band1_db = 20 * np.log10(abs(self.h_band1))
        self.h_band2_db = 20 * np.log10(abs(self.h_band2))
        self.h_band3_db = 20 * np.log10(abs(self.h_band3))
        self.h_band4_db = 20 * np.log10(abs(self.h_band4))
        self.h_high_shelf_db = 20 * np.log10(abs(self.h_high_shelf))
        self.total_response = (
            self.h_low_shelf_db
            + self.h_band1_db
            + self.h_band2_db
            + self.h_band3_db
            + self.h_band4_db
            + self.h_high_shelf_db
        )

        # fig = plt.figure(200, figsize=[14.0, 5.0], facecolor="black")
        freq_axis = self.w_low_shelf / np.pi * self.rate / 2.0
        # x = np.arange(len(freq_axis))
        ax1 = plt.axes()
        ax1.set_title("Smart EQ", color="C0")
        ax1.scatter(self.fc0, self.g0, color="orange")
        ax1.fill_between(
            freq_axis, self.h_low_shelf_db, color="orange", alpha=0.3, label="lowcut"
        )
        ax1.scatter(self.fc1, self.g1, color="purple")
        ax1.fill_between(
            freq_axis, self.h_band1_db, color="purple", alpha=0.3, label="body"
        )
        ax1.scatter(self.fc2, self.g2, color="y")
        ax1.fill_between(freq_axis, self.h_band2_db, color="y", alpha=0.3, label="mud")
        ax1.scatter(self.fc3, self.g3, color="g")
        ax1.fill_between(
            freq_axis, self.h_band3_db, color="g", alpha=0.3, label="presence1"
        )
        ax1.scatter(self.fc4, self.g4, color="cyan")
        ax1.fill_between(
            freq_axis, self.h_band4_db, color="cyan", alpha=0.3, label="presence2"
        )
        ax1.scatter(self.fc5, self.g5, color="b")
        ax1.fill_between(
            freq_axis, self.h_high_shelf_db, color="b", alpha=0.3, label="air"
        )
        ax1.semilogx(freq_axis, self.total_response, "--w")
        ax1.set_facecolor("xkcd:black")
        ax1.set_xlim(20, 20000)
        ax1.set_ylim(-14.0, 14.0)
        ax1.tick_params(axis="x", colors="grey")
        ax1.tick_params(axis="y", colors="grey")
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        ax1.set_xticklabels([1, 10, 100, "1k", "10k", "20k"])
        ax1.legend()
        plt.savefig(save_path, dpi=600)
        plt.close()


class IIRfilter(object):
    """IIR Filter object to pre-filtering

    This class allows for the generation of various IIR filters
        in order to apply different frequency weighting to audio data
        before measuring the loudness.
    Parameters
    ----------
    G : float
        Gain of the filter in dB.
    Q : float
        Q of the filter.
    fc : float
        Center frequency of the shelf in Hz.
    rate : float
        Sampling rate in Hz.
    filter_type: str
        Shape of the filter.
    """

    def __init__(self, G, Q, fc, rate, filter_type, passband_gain=1.0):
        self.G = G
        self.Q = Q
        self.fc = fc
        self.rate = rate
        self.filter_type = filter_type
        self.passband_gain = passband_gain

    def generate_coefficients(self):
        """Generates biquad filter coefficients using instance filter parameters.
        This method is called whenever an IIRFilter is instantiated and then sets
        the coefficients for the filter instance.
        Design of the 'standard' filter types are based upon the equations
        presented by RBJ in the "Cookbook formulae for audio equalizer biquad
        filter coefficients" which can be found at the link below.
        http://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
        Additional filter designs are also available. Brecht DeMan found that
        the coefficients generated by the RBJ filters do not directly match
        the coefficients provided in the ITU specification. For full compliance
        use the 'DeMan' filters below when constructing filters. Details on his
        work can be found at the GitHub repository below.
        https://github.com/BrechtDeMan/loudness.py
        Returns
        -------
        b : ndarray
            Numerator filter coefficients stored as [b0, b1, b2]
        a : ndarray
            Denominator filter coefficients stored as [a0, a1, a2]
        """
        A = 10 ** (self.G / 40.0)
        w0 = 2.0 * np.pi * (self.fc / self.rate)
        alpha = np.sin(w0) / (2.0 * self.Q)

        if self.filter_type == "high_shelf":
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        elif self.filter_type == "low_shelf":
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        elif self.filter_type == "high_pass":

            b0 = (1 + np.cos(w0)) / 2
            b1 = -(1 + np.cos(w0))
            b2 = (1 + np.cos(w0)) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
        elif self.filter_type == "low_pass":
            b0 = (1 - np.cos(w0)) / 2
            b1 = 1 - np.cos(w0)
            b2 = (1 - np.cos(w0)) / 2
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
        elif self.filter_type == "peaking":
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
        elif self.filter_type == "notch":
            b0 = 1
            b1 = -2 * np.cos(w0)
            b2 = 1
            a0 = 1 + alpha
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha
        elif self.filter_type == "high_shelf_DeMan":
            K = np.tan(np.pi * self.fc / self.rate)
            Vh = np.power(10.0, self.G / 20.0)
            Vb = np.power(Vh, 0.499666774155)
            a0_ = 1.0 + K / self.Q + K * K
            b0 = (Vh + Vb * K / self.Q + K * K) / a0_
            b1 = 2.0 * (K * K - Vh) / a0_
            b2 = (Vh - Vb * K / self.Q + K * K) / a0_
            a0 = 1.0
            a1 = 2.0 * (K * K - 1.0) / a0_
            a2 = (1.0 - K / self.Q + K * K) / a0_
        elif self.filter_type == "high_pass_DeMan":
            K = np.tan(np.pi * self.fc / self.rate)
            a0 = 1.0
            a1 = 2.0 * (K * K - 1.0) / (1.0 + K / self.Q + K * K)
            a2 = (1.0 - K / self.Q + K * K) / (1.0 + K / self.Q + K * K)
            b0 = 1.0
            b1 = -2.0
            b2 = 1.0
        else:
            raise ValueError("Invalid filter type", self.filter_type)

        return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0

    def apply_filter(self, data):
        """Apply the IIR filter to an input signal.
        Params
        -------
        data : ndarrary
            Input audio data.
        Returns
        -------
        filtered_signal : ndarray
            Filtered input audio.
        """
        b, a = self.generate_coefficients()

        return self.passband_gain * scipy.signal.lfilter(b, a, data)
