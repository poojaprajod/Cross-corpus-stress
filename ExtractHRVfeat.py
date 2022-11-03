import numpy as np
import neurokit2 as nk
import pandas as pd
import peakutils
from scipy.stats import zscore
from scipy import signal
from scipy.ndimage import uniform_filter1d
import gc

num_frames = 60
frame_length_ecg = 256
frame_length_bvp = 64


def prepare_data(subs):
    X_ecg = []
    X_bvp = []
    y_ecg = []
    y_bvp = []
    for i in subs:
        subject = 'S' + str(i)
        sub_ecg_data = np.load(f'D:/StressSignals/{subject}_ecg_ffilt.npy', allow_pickle=True)
        sub_bvp_data = np.load(f'D:/StressSignals/{subject}_bvp.npy', allow_pickle=True)

        for label in [0, 1, 2]:  # baseline, stress, amusement
            ecg_data_sub_label = sub_ecg_data[label]
            bvp_data_sub_label = sub_bvp_data[label]
            if label == 2:
                label = 0

            for idx in range(0, len(ecg_data_sub_label) - frame_length_ecg * num_frames + 1,
                             int(frame_length_ecg * 1.0)):  # overlap of 1s
                X_ecg.append(ecg_data_sub_label[idx: idx + frame_length_ecg * num_frames].flatten())
                y_ecg.append(label)

            for idx in range(0, len(bvp_data_sub_label) - frame_length_bvp * num_frames + 1,
                             int(frame_length_bvp * 1.0)):  # overlap of 1s
                X_bvp.append(bvp_data_sub_label[idx: idx + frame_length_bvp * num_frames].flatten())
                y_bvp.append(label)

    X_ecg = np.array(X_ecg)
    X_bvp = np.array(X_bvp)

    y_ecg = np.array(y_ecg)
    y_bvp = np.array(y_bvp)

    return X_ecg, X_bvp, y_ecg, y_bvp


def get_rri(peaks=None, sampling_rate=1000):
    # Shamelessly copied from nk
    rri = np.diff(peaks) / sampling_rate * 1000
    return rri


def get_hrv_nl(rri):
    # Shamelessly copied from nk
    out = {}
    rri_n = rri[:-1]
    rri_plus = rri[1:]
    x1 = (rri_n - rri_plus) / np.sqrt(2)  # Eq.7
    x2 = (rri_n + rri_plus) / np.sqrt(2)
    sd1 = np.std(x1, ddof=1)
    sd2 = np.std(x2, ddof=1)

    out["SD1"] = sd1
    out["SD2"] = sd2

    out["SD1SD2"] = sd1 / sd2

    out["S"] = np.pi * out["SD1"] * out["SD2"]

    out = pd.DataFrame.from_dict(out, orient="index")
    out = out.fillna(-1)

    return out


def extract_hrv(sig, sr=1000, typ='ecg', clip_max=3.0):
    hrv_feats = []
    peaks = []
    if typ == 'bvp':
        ppg_clean = nk.ppg_clean(sig, method='elgendi', sampling_rate=sr)
        info = nk.ppg_findpeaks(ppg_clean, sampling_rate=sr)
        peaks = info["PPG_Peaks"]
    elif typ == 'ecg':
        sig = zscore(sig)
        info = nk.ecg_findpeaks(sig, method='elgendi2010', sampling_rate=sr)
        peaks = info["ECG_R_Peaks"]
    else:
        # sig = sig*sig
        # sig = np.clip(sig, -1.0, 1.0)
        # sorted_sig_idx = np.argsort(sig)
        # threshold = np.median(sig[sorted_sig_idx[-60:]])
        # sig = butter_bandpass(sig, [0.5, 8], 3, sr)
        # sig = calc_mwa(sig, 3)
        q90 = np.percentile(sig, 90)
        q75 = np.percentile(sig, 75)
        threshold = (q90 + q75)/2
        # sig = np.clip(sig, -q90, q90)
        # sig = zscore(sig)
        # sig = np.clip(sig, -3.0, 3.0)
        peaks = peakutils.indexes(sig, threshold, int(np.rint(0.3333 * sr)), True)

    sig_duration_s = len(sig) / sr
    sig_duration_min = sig_duration_s / 60
    hrv_feats.append(round(len(peaks) / sig_duration_min))

    hrv_time = nk.hrv_time(peaks, sampling_rate=sr)
    hrv_time = hrv_time.fillna(-1)
    hrv_time = hrv_time[["HRV_MeanNN", "HRV_SDNN", "HRV_CVNN", "HRV_MedianNN", "HRV_MadNN", "HRV_RMSSD", "HRV_SDSD",
                         "HRV_IQRNN", "HRV_pNN50", "HRV_pNN20", "HRV_TINN", "HRV_HTI"]].to_numpy().flatten().tolist()

    hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sr, psd_method="lomb")
    hrv_freq = hrv_freq.fillna(-1)
    hrv_freq = hrv_freq[["HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn", "HRV_HFn"]].to_numpy().flatten().tolist()

    rri = get_rri(peaks, sr)
    hrv_nl = get_hrv_nl(rri).to_numpy().flatten().tolist()  # SD1, SD2. SD1SD2, S, SampEn
    hrv_feats = hrv_feats + hrv_time + hrv_freq + hrv_nl
    return hrv_feats


def prepare_hrv(sigs, sr=1000, typ='ecg'):
    hrv_feat_lists = []
    i = 0
    for sig in sigs:
        # print(i)
        hrv_feats = extract_hrv(sig, sr, typ)
        hrv_feat_lists.append(hrv_feats)
        i = i + 1

    return hrv_feat_lists


def butter_bandpass(data, cutoff=[0.5, 100], order=5, fs=256):
    sos = signal.butter(order, cutoff, btype='band', analog=False, output='sos', fs=fs)
    x = signal.sosfiltfilt(sos, data)
    return x


def calc_mwa(sig, window_size):
    # Shamelessly copied from Neurokit
    window_size = int(window_size)
    mwa = uniform_filter1d(sig, window_size, origin=(window_size - 1) // 2)
    head_size = min(window_size - 1, len(sig))
    mwa[:head_size] = np.cumsum(sig[:head_size]) / np.linspace(1, head_size, head_size)
    return mwa


subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
for s in subs:
    subject = 'S' + str(s)
    test_subs = [s]

    X_ecg_test, X_bvp_test, y_ecg_test, y_bvp_test = prepare_data(test_subs)
    # X_bvp_hrv_test = prepare_hrv(X_bvp_test, sr=frame_length_bvp, typ='bvp')
    X_ecg_hrv_test = prepare_hrv(X_ecg_test, sr=frame_length_ecg, typ='ecg')

    # X_bvp_hrv_test = np.array(X_bvp_hrv_test)
    X_ecg_hrv_test = np.array(X_ecg_hrv_test)

    # np.save(f'D:/StressSignals/{subject}_bvp_hrv.npy', X_bvp_hrv_test)
    np.save(f'D:/StressSignals/{subject}_ecg_hrv.npy', X_ecg_hrv_test)

    np.save(f'D:/StressSignals/{subject}_ecg_hrv_y.npy', y_ecg_test)
    # np.save(f'D:/StressSignals/{subject}_bvp_hrv_y.npy', y_bvp_test)

    del X_ecg_test, X_bvp_test, y_ecg_test, y_bvp_test

    gc.collect()


def prepare_swell_data(id):
    X_ecg = []
    ys = []

    subject = 'P' + str(id)
    sub_ecg_data = np.load(f'D:/StressSignals/{subject}_ecg_ffilt.npy', allow_pickle=True)
    for label in [0, 1, 2]:  # baseline, stress, stress
        ecg_data_sub_label = sub_ecg_data[label]
        if label == 2:
            label = 1

        for idx in range(0, len(ecg_data_sub_label) - frame_length_ecg * num_frames + 1,
                         int(frame_length_ecg * 1.0)):
            ecg = ecg_data_sub_label[idx: idx + frame_length_ecg * num_frames].flatten()
            X_ecg.append(ecg)
            ys.append(label)

    X_ecg = np.array(X_ecg)
    ys = np.array(ys)
    return X_ecg, ys


# Missing data 11, 23
# Something wrong with 8
subs = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
for s in subs:
    subject = 'P' + str(s)

    X_ecg_test, y_ecg_test = prepare_swell_data(s)

    X_ecg_hrv_test = prepare_hrv(X_ecg_test, sr=frame_length_ecg, typ='ecg')

    X_ecg_hrv_test = np.array(X_ecg_hrv_test)

    np.save(f'D:/StressSignals/{subject}_ecg_hrv.npy', X_ecg_hrv_test)

    np.save(f'D:/StressSignals/{subject}_ecg_hrv_y.npy', y_ecg_test)

    del X_ecg_test, y_ecg_test

    gc.collect()


