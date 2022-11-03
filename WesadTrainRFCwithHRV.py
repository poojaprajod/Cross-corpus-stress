import numpy as np
import gc
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report


def prepare_hrv_data(sub_list):
    X_hrv = None
    ys = None
    for sub in sub_list:
        subject = 'S' + str(sub)
        sub_hrv = np.load(f'D:/StressSignals/{subject}_ecg_hrv.npy', allow_pickle=True)
        sub_hrv = sub_hrv[:, :22]
        sub_hrv = sub_hrv[::10].copy()
        sub_hrv = (sub_hrv - np.nanmin(sub_hrv[6:36], axis=0))/(np.nanmax(sub_hrv[6:36], axis=0) - np.nanmin(sub_hrv[6:36], axis=0))
        sub_y = np.load(f'D:/StressSignals/{subject}_ecg_hrv_y.npy', allow_pickle=True)
        sub_y = sub_y[::10].copy()
        if X_hrv is None:
            X_hrv = sub_hrv
            ys = sub_y
        else:
            X_hrv = np.vstack((X_hrv, sub_hrv))
            ys = np.append(ys, sub_y)

    return X_hrv, ys


subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
for s in subs:
    test_subs = [s]
    train_subs = list(set(subs) - set(test_subs))

    X_hrv_train, ys_train = prepare_hrv_data(train_subs)
    X_hrv_test, ys_test = prepare_hrv_data(test_subs)

    trained = False
    if not trained:
        rfc = RandomForestClassifier(random_state=42, min_samples_split=20, n_estimators=100, class_weight='balanced')
        rfc.fit(X_hrv_train, ys_train)
        joblib.dump(rfc, f"rfc_model_ecg_hrv_val_sub{s}")

    rfc = joblib.load(f"rfc_model_ecg_hrv_val_sub{s}")
    pred = rfc.predict(X_hrv_test)

    print(f'subject {s}')
    print(classification_report(ys_test, pred))

    del X_hrv_train, ys_train, X_hrv_test, ys_test
    del rfc, pred

    gc.collect()


def prepare_swell_hrv():
    parts = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
    X_hrv = None
    ys = None
    for p in parts:
        subject = 'P' + str(p)
        sub_hrv = np.load(f'D:/StressSignals/{subject}_ecg_hrv.npy', allow_pickle=True)
        sub_hrv = sub_hrv[:, :22]
        sub_hrv = sub_hrv[::10].copy()
        sub_hrv = (sub_hrv - np.nanmin(sub_hrv[6:36], axis=0))/(np.nanmax(sub_hrv[6:36], axis=0) - np.nanmin(sub_hrv[6:36], axis=0))
        sub_y = np.load(f'D:/StressSignals/{subject}_ecg_hrv_y.npy', allow_pickle=True)
        sub_y = sub_y[::10].copy()
        if X_hrv is None:
            X_hrv = sub_hrv
            ys = sub_y
        else:
            X_hrv = np.vstack((X_hrv, sub_hrv))
            ys = np.append(ys, sub_y)

    return X_hrv, ys


X_swell_hrv, ys_swell = prepare_swell_hrv()
for s in subs:
    loaded_rfc = joblib.load(f"rfc_model_ecg_hrv_val_sub{s}")

    pred = loaded_rfc.predict(X_swell_hrv)

    print(f'subject {s}')
    print(classification_report(ys_swell, pred))

