from tensorflow.keras.models import model_from_json
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.stats import zscore
from tensorflow.keras import backend as K
import tensorflow as tf

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

num_frames = 10
frame_length = 256
channels = 1  # Just ECG for now

json_file = open('model_ecg_deepecg_swell.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


def prepare_data(parts):
    X_ecg = []
    y = []
    for i in parts:
        subject = 'S' + str(i)
        sub_ecg_data = np.load(f'D:/StressSignals/{subject}_ecg_ffilt.npy', allow_pickle=True)
        sub_min = np.min(sub_ecg_data[0][256 * 60: 256 * 360].flatten())
        sub_max = np.max(sub_ecg_data[0][256 * 60: 256 * 360].flatten())
        for label in [0, 1, 2]:  # baseline, stress, stress
            ecg_data_sub_label = sub_ecg_data[label]
            if label == 2:
                label = 0
            # for idx in range(0, len(ecg_data_sub_label) - num_frames + 1, 1):  # 1=1 frame, num_frames=no overlap
            for idx in range(0, len(ecg_data_sub_label) - frame_length * num_frames + 1,
                             int(frame_length * num_frames)):
                ecg = ecg_data_sub_label[idx: idx + frame_length * num_frames].flatten()
                ecg = (ecg - sub_min) / (sub_max - sub_min)
                ecg = zscore(ecg)
                X_ecg.append(ecg)
                y.append(label)

    X_ecg = np.array(X_ecg)
    y = np.array(y)
    X_ecg = X_ecg.reshape((X_ecg.shape[0], int(num_frames * frame_length), 1))

    return X_ecg, y


subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
X_ecg_eda, true = prepare_data(subs)  # ECG data. Keeping the naming error as it was found much later
true = true.astype(int)

f1_scores = []
accuracies = []
parts = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
for s in parts:
    loaded_model.load_weights(f'best_model_ecg_val{s}_znorm_deepecg_swell.h5')
    preds = loaded_model.predict(X_ecg_eda)
    state = np.argmax(preds, axis=1)
    # state = np.round(preds)
    print(f'subject {s}')
    print(classification_report(true, state))
    sub_f1 = f1_score(true, state, average='macro')
    sub_acc = accuracy_score(true, state)
    f1_scores.append(sub_f1)
    accuracies.append(sub_acc)

print(np.mean(accuracies))
print(np.mean(f1_scores))
