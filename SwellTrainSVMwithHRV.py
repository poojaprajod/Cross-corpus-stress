import numpy as np
import gc
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.inspection import permutation_importance


def prepare_hrv_data(sub_list):
    X_hrv = None
    ys = None
    for sub in sub_list:
        subject = 'P' + str(sub)
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


def prepare_wesad_hrv():
    subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    X_hrv = None
    ys = None
    for p in subs:
        subject = 'S' + str(p)
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


f1_scores = []
accuracies = []
parts = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
for s in parts:
    test_subs = [s]
    train_subs = list(set(parts) - set(test_subs))

    X_hrv_train, ys_train = prepare_hrv_data(train_subs)
    X_hrv_test, ys_test = prepare_hrv_data(test_subs)

    trained = True
    if not trained:
        svc = SVC(random_state=42, kernel='rbf', class_weight='balanced', max_iter=100000)
        svc.fit(X_hrv_train, ys_train)
        joblib.dump(svc, f"svc_model_ecg_hrv_val_part{s}")

    svc = joblib.load(f"svc_model_ecg_hrv_val_part{s}")
    pred = svc.predict(X_hrv_test)

    print(f'subject {s}')
    print(classification_report(ys_test, pred))
    # perm_importance = permutation_importance(svc, X_hrv_test, ys_test, n_repeats=10, random_state=42)
    # print(perm_importance.importances_mean)
    sub_f1 = f1_score(ys_test, pred, average='macro')
    sub_acc = accuracy_score(ys_test, pred)
    f1_scores.append(sub_f1)
    accuracies.append(sub_acc)

    del X_hrv_train, ys_train, X_hrv_test, ys_test
    del svc, pred

    gc.collect()

print(np.mean(accuracies))
print(np.mean(f1_scores))

X_wesad_hrv, ys_wesad = prepare_wesad_hrv()

f1_scores = []
accuracies = []
for s in parts:
    loaded_svc = joblib.load(f"svc_model_ecg_hrv_val_part{s}")

    pred = loaded_svc.predict(X_wesad_hrv)

    print(f'subject {s}')
    print(classification_report(ys_wesad, pred))
    sub_f1 = f1_score(ys_wesad, pred, average='macro')
    sub_acc = accuracy_score(ys_wesad, pred)
    f1_scores.append(sub_f1)
    accuracies.append(sub_acc)

print(np.mean(accuracies))
print(np.mean(f1_scores))

