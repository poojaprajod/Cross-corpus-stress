import os
import pickle
import numpy as np

from scipy import signal
from scipy.ndimage import uniform_filter1d

import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class read_data_one_subject:

    def __init__(self, subject):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        with open(f'E:/WESAD/{subject}/{subject}.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        return wrist_data

    def get_chest_data(self):
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data


def butter_bandpass(data, cutoff=[0.5, 100], order=5, fs=256):
    sos = signal.butter(order, cutoff, btype='band', analog=False, output='sos', fs=fs)
    x = signal.sosfiltfilt(sos, data)
    return x


def butter_highpass(data, cutoff=0.67, order=5, fs=256):
    sos = signal.butter(order, cutoff, btype='high', analog=False, output='sos', fs=fs)
    x = signal.sosfiltfilt(sos, data)
    return x


def butter_lowpass(data, cutoff=5, order=5, fs=256):
    sos = signal.butter(order, cutoff, btype='low', analog=False, output='sos', fs=fs)
    x = signal.sosfiltfilt(sos, data)
    return x


def notch(data, cutoff=60, order=30, fs=256):
    b, a = signal.iirnotch(cutoff, order, fs)
    x = signal.lfilter(b, a, data)
    return x


def get_window():
    windowSize = 40
    window = np.kaiser(windowSize, 2)
    window = window / window.sum()
    return window


def ecg_mwa(sig, window_size):
    # Shamelessly copied from Neurokit
    window_size = int(window_size)
    mwa = uniform_filter1d(sig, window_size, origin=(window_size - 1) // 2)
    head_size = min(window_size - 1, len(sig))
    mwa[:head_size] = np.cumsum(sig[:head_size]) / np.linspace(1, head_size, head_size)
    return mwa


def extract_series_by_index(chest_data_dict, idx):
    # ECG data
    ecg_data = chest_data_dict["ECG"][idx].flatten()
    # Downsample to 256Hz
    ecg_data = signal.resample(ecg_data, int((len(ecg_data) / 700) * 256))
    # Bandpass filter
    ecg_data = butter_bandpass(ecg_data, [8, 20], 2, 256)
    ecg_ffilt = ecg_data  # Storing frequency filtered ECG signal

    # Experimenting with further filtering. Not used in the final implementation
    ecg_data = np.abs(ecg_data)
    ecg_data = ecg_mwa(ecg_data, round(0.1222 * 256))
    # ecg_data = np.convolve(ecg_data, np.ones(round(0.1222 * 700)) / round(0.1222 * 700), mode='same')

    return ecg_data, ecg_ffilt


def get_all_ecg_eda_data():
    all_ecg_data = {}
    all_eda_data = {}
    subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    for i in subs:
        subject = 'S' + str(i)
        print("Reading data", subject)
        obj_data = read_data_one_subject(subject)
        labels = obj_data.get_labels()

        chest_data_dict = obj_data.get_chest_data()

        # Do for each subject
        baseline = np.asarray([idx for idx, val in enumerate(labels) if val == 1])
        # print("Baseline:", chest_data_dict['ECG'][baseline].shape)
        # print(baseline.shape)

        stress = np.asarray([idx for idx, val in enumerate(labels) if val == 2])
        # print(stress.shape)

        amusement = np.asarray([idx for idx, val in enumerate(labels) if val == 3])
        # print(amusement.shape)

        baseline_ecg_data, baseline_ecg_ffilt = extract_series_by_index(chest_data_dict, baseline)
        stress_ecg_data, stress_ecg_ffilt = extract_series_by_index(chest_data_dict, stress)
        amusement_ecg_data, amusement_ecg_ffilt = extract_series_by_index(chest_data_dict, amusement)

        full_ecg_data = np.array([baseline_ecg_data, stress_ecg_data, amusement_ecg_data])
        full_ecg_ffilt = np.array([baseline_ecg_ffilt, stress_ecg_ffilt, amusement_ecg_ffilt])

        np.save(f'D:/StressSignals/{subject}_ecg.npy', full_ecg_data)
        np.save(f'D:/StressSignals/{subject}_ecg_ffilt.npy', full_ecg_ffilt)


get_all_ecg_eda_data()
