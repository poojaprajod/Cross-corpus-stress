import numpy as np
import gc
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, SGD
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import model_from_json

from tensorflow.keras import backend as K
import tensorflow as tf

import random
import os

from tfdeterminism import patch

patch()

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

batch_size = 128
epochs = 200


def prepare_hrv_data(sub_list):
    X_hrv = None
    ys = None
    for sub in sub_list:
        sub_hrv = np.load(f'D:/StressSignals/{sub}_ecg_hrv.npy', allow_pickle=True)
        sub_hrv = sub_hrv[:, :22]
        sub_hrv = sub_hrv[::10].copy()
        sub_hrv = (sub_hrv - np.nanmin(sub_hrv[6:36], axis=0)) / (
                    np.nanmax(sub_hrv[6:36], axis=0) - np.nanmin(sub_hrv[6:36], axis=0))
        sub_y = np.load(f'D:/StressSignals/{sub}_ecg_hrv_y.npy', allow_pickle=True)
        sub_y = sub_y[::10].copy()
        if X_hrv is None:
            X_hrv = sub_hrv
            ys = sub_y
        else:
            X_hrv = np.vstack((X_hrv, sub_hrv))
            ys = np.append(ys, sub_y)

    return X_hrv, ys


def simple_NN():
    model = Sequential()
    model.add(Input(shape=(22,)))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


model_full = simple_NN()
model_full.save_weights('init_nn_model_weights.h5')

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
        model_full.load_weights(f'init_nn_model_weights.h5')

        weights = dict(zip(np.unique(ys_train), compute_class_weight('balanced', classes=np.unique(ys_train), y=ys_train)))

        model_full.compile(loss='binary_crossentropy', optimizer=Adadelta(learning_rate=1.0), metrics=['accuracy'])

        callbacks = [  # EarlyStopping(monitor='val_acc', patience=10, min_delta=0.0001, mode='max'),
            ModelCheckpoint(filepath=f'nn_model_ecg_hrv_joint_val_{s}.h5', monitor='val_loss', save_best_only=True)
        ]
        model_full.fit(X_hrv_train, ys_train, validation_data=(X_hrv_test, ys_test), epochs=epochs,
                       callbacks=callbacks, class_weight=weights,
                       batch_size=batch_size, shuffle=True)
        # max_queue_size=1, workers=1, use_multiprocessing=False, shuffle=False)
        # preds = model_full.predict([X_ecg_test, X_eda_test])

        model_json = model_full.to_json()
        with open("model_ecg_nn.json", "w") as json_file:
            json_file.write(model_json)

    model_full.load_weights(f'nn_model_ecg_hrv_joint_val_{s}.h5')
    preds = model_full.predict(X_hrv_test)
    state = np.round(preds)
    true = np.round(ys_test)
    print(f'subject {s}')
    print(classification_report(true, state))

    sub_f1 = f1_score(true, state, average='macro')
    sub_acc = accuracy_score(true, state)
    f1_scores.append(sub_f1)
    accuracies.append(sub_acc)

    if 'S' in s:
        wesad_f1_scores.append(sub_f1)
        wesad_accuracies.append(sub_acc)
    else:
        swell_f1_scores.append(sub_f1)
        swell_accuracies.append(sub_acc)

    del X_hrv_train, ys_train, X_hrv_test, ys_test, preds

    gc.collect()

print(np.mean(accuracies))
print(np.mean(f1_scores))

print(np.mean(wesad_accuracies))
print(np.mean(wesad_f1_scores))

print(np.mean(swell_accuracies))
print(np.mean(swell_f1_scores))
