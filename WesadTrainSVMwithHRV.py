import numpy as np
import gc
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance


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

    trained = True
    if not trained:
        svc = SVC(random_state=42, kernel='rbf', class_weight='balanced', max_iter=100000)
        svc.fit(X_hrv_train, ys_train)
        joblib.dump(svc, f"svc_model_ecg_hrv_val_sub{s}")

    svc = joblib.load(f"svc_model_ecg_hrv_val_sub{s}")
    pred = svc.predict(X_hrv_test)

    print(f'subject {s}')
    print(classification_report(ys_test, pred))
    # perm_importance = permutation_importance(svc, X_hrv_test, ys_test, n_repeats=10, random_state=42)
    # print(perm_importance.importances_mean)

    del X_hrv_train, ys_train, X_hrv_test, ys_test
    del svc, pred

    gc.collect()

##### Trying SVC with probabilitites for another project #####

# test_subs = [3, 10, 8]
# train_subs = list(set(subs) - set(test_subs))
#
# X_hrv_train, ys_train = prepare_hrv_data(train_subs)
# X_hrv_test, ys_test = prepare_hrv_data(test_subs)
# svc = SVC(random_state=42, kernel='rbf', class_weight='balanced', max_iter=100000, probability=True)
# svc.fit(X_hrv_train, ys_train)
# joblib.dump(svc, f"svc_model_ecg_hrv")
#
# svc = joblib.load(f"svc_model_ecg_hrv")
# pred = svc.predict_proba(X_hrv_test)
# pred = np.argmax(pred, axis=1)
# print(classification_report(ys_test, pred))


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
    loaded_svc = joblib.load(f"svc_model_ecg_hrv_val_sub{s}")

    pred = loaded_svc.predict(X_swell_hrv)

    print(f'subject {s}')
    print(classification_report(ys_swell, pred))

##### Trying SVC with probabilitites for another project #####

# loaded_svc = joblib.load(f"svc_model_ecg_hrv")
# pred = loaded_svc.predict_proba(X_swell_hrv)
# pred = np.argmax(pred, axis=1)
# print(classification_report(ys_swell, pred))


