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
        sub_hrv = np.load(f'D:/StressSignals/{sub}_ecg_hrv.npy', allow_pickle=True)
        sub_hrv = sub_hrv[:, :22]
        sub_hrv = sub_hrv[::10].copy()
        sub_hrv = (sub_hrv - np.nanmin(sub_hrv[6:36], axis=0))/(np.nanmax(sub_hrv[6:36], axis=0) - np.nanmin(sub_hrv[6:36], axis=0))
        sub_y = np.load(f'D:/StressSignals/{sub}_ecg_hrv_y.npy', allow_pickle=True)
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

wesad_f1_scores = []
wesad_accuracies = []

swell_f1_scores = []
swell_accuracies = []

parts = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P9', 'P10', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19',
         'P20', 'P21', 'P22', 'P24', 'P25',
         'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
for s in parts:
    test_subs = [s]
    train_subs = list(set(parts) - set(test_subs))

    X_hrv_train, ys_train = prepare_hrv_data(train_subs)
    X_hrv_test, ys_test = prepare_hrv_data(test_subs)

    trained = True
    if not trained:
        svc = SVC(random_state=42, kernel='rbf', class_weight='balanced', max_iter=100000)
        svc.fit(X_hrv_train, ys_train)
        joblib.dump(svc, f"svc_model_ecg_hrv_joint_val_{s}")

    svc = joblib.load(f"svc_model_ecg_hrv_joint_val_{s}")
    pred = svc.predict(X_hrv_test)

    print(f'subject {s}')
    print(classification_report(ys_test, pred))
    # perm_importance = permutation_importance(svc, X_hrv_test, ys_test, n_repeats=10, random_state=42)
    # print(perm_importance.importances_mean)
    sub_f1 = f1_score(ys_test, pred, average='macro')
    sub_acc = accuracy_score(ys_test, pred)
    f1_scores.append(sub_f1)
    accuracies.append(sub_acc)

    if 'S' in s:
        wesad_f1_scores.append(sub_f1)
        wesad_accuracies.append(sub_acc)
    else:
        swell_f1_scores.append(sub_f1)
        swell_accuracies.append(sub_acc)

    del X_hrv_train, ys_train, X_hrv_test, ys_test
    del svc, pred

    gc.collect()

print(np.mean(accuracies))
print(np.mean(f1_scores))

print(np.mean(wesad_accuracies))
print(np.mean(wesad_f1_scores))

print(np.mean(swell_accuracies))
print(np.mean(swell_f1_scores))


