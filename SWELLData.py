import os
import numpy as np
import pandas as pd

from scipy import signal
from scipy.ndimage import uniform_filter1d

import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

frame_length = 256
num_frames = 10


def get_window():
    windowSize = 26
    window = np.kaiser(windowSize, 15)
    window = window / window.sum()
    return window


def butter_bandpass(data, cutoff=[0.5, 100], order=5, fs=256):
    sos = signal.butter(order, cutoff, btype='band', analog=False, output='sos', fs=fs)
    x = signal.sosfiltfilt(sos, data)
    return x


def butter_lowpass(data, cutoff=5, order=5, fs=256):
    sos = signal.butter(order, cutoff, btype='low', analog=False, output='sos', fs=fs)
    x = signal.sosfiltfilt(sos, data)
    return x


def butter_highpass(data, cutoff=5, order=5, fs=256):
    sos = signal.butter(order, cutoff, btype='high', analog=False, output='sos', fs=fs)
    x = signal.sosfiltfilt(sos, data)
    return x


def ecg_mwa(sig, window_size):
    # Shamelessly copied from Neurokit
    window_size = int(window_size)
    mwa = uniform_filter1d(sig, window_size, origin=(window_size - 1) // 2)
    head_size = min(window_size - 1, len(sig))
    mwa[:head_size] = np.cumsum(sig[:head_size]) / np.linspace(1, head_size, head_size)
    return mwa


def read_part_ecg(id, cond, ):
    ecg = pd.read_csv(f'E:/SWELL/ecg_pp{id}_c{cond}.csv', header=None).to_numpy().flatten()  # Converted .S00 to csv data
    ecg = ecg / 1000  # Scaling data
    # Downsample to 256Hz
    ecg = signal.resample(ecg, int((len(ecg) / 2048) * 256))
    # Bandpass filter
    ecg = butter_bandpass(ecg, [8, 20], 2, 256)
    ecg_ffilt = ecg  # Storing frequency filtered ECG signal

    # Experimenting with further filtering. Not used in the final implementation
    ecg = np.abs(ecg)
    ecg = ecg_mwa(ecg, round(0.1222 * 256))
    # ecg = np.convolve(ecg, np.ones(round(0.1222 * 2048)) / round(0.1222 * 2048), mode='same')

    return ecg, ecg_ffilt


def read_part_eda(id, cond):
    eda = pd.read_csv(f'E:/SWELL/eda_pp{id}_c{cond}.csv', header=None).to_numpy().flatten()
    eda_rt = butter_lowpass(eda, 1, 3, 2048)
    eda = butter_lowpass(eda, 4, 4, 2048)
    eda = signal.resample(eda, int((len(eda) / 2048) * 256))
    eda = np.convolve(eda, np.ones(26) / 26, mode='same')

    eda_t = butter_lowpass(eda_rt, 0.05, 3, 2048)
    eda_r = eda_rt - eda_t
    eda_t = signal.resample(eda_t, int((len(eda_t) / 2048) * 256))
    eda_r = signal.resample(eda_r, int((len(eda_r) / 2048) * 256))
    return eda, eda_r, eda_t


# No EDA for participant 23. Participant 11, no c2

subs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
# bloks = [[1, 2], [1, 3]]
# Tried separating the stress conditions. Doesn't seem to change much.
# So, the final implementation is with all stress conditions together.
bloks = [[1, 2, 3], [1, 2, 3]]
bloks_id = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]

for sub, b_id in zip(subs, bloks_id):

    ecgs = []
    ecgs_ffilt = []
    ys = []

    conds = bloks[b_id]
    for c in conds:
        if (sub == 11) and (c == 2):
            continue

        sub_ecg, sub_ecg_ffilt = read_part_ecg(sub, c)

        ecgs.append(sub_ecg)
        ecgs_ffilt.append(sub_ecg_ffilt)

    ecgs = np.array(ecgs)
    ecgs_ffilt = np.array(ecgs_ffilt)

    np.save(f'D:/StressSignals/P{sub}_ecg.npy', ecgs)
    np.save(f'D:/StressSignals/P{sub}_ecg_ffilt.npy', ecgs_ffilt)

