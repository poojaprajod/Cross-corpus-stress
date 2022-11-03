import numpy as np
import gc
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, SGD
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, f1_score, accuracy_score
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
parts = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]

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
            ModelCheckpoint(filepath=f'nn_model_ecg_hrv_val_part{s}.h5', monitor='val_loss', save_best_only=True)
        ]
        model_full.fit(X_hrv_train, ys_train, validation_data=(X_hrv_test, ys_test), epochs=epochs,
                       callbacks=callbacks, class_weight=weights,
                       batch_size=batch_size, shuffle=True)
        # max_queue_size=1, workers=1, use_multiprocessing=False, shuffle=False)

        model_json = model_full.to_json()
        with open("model_ecg_nn.json", "w") as json_file:
            json_file.write(model_json)

    model_full.load_weights(f'nn_model_ecg_hrv_val_part{s}.h5')
    preds = model_full.predict(X_hrv_test)
    state = np.round(preds)
    true = np.round(ys_test)
    print(f'subject {s}')
    print(classification_report(true, state))
    sub_f1 = f1_score(true, state, average='macro')
    sub_acc = accuracy_score(true, state)
    f1_scores.append(sub_f1)
    accuracies.append(sub_acc)

    del X_hrv_train, ys_train, X_hrv_test, ys_test

    gc.collect()

print(np.mean(accuracies))
print(np.mean(f1_scores))


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


X_wesad_hrv, ys_wesad = prepare_wesad_hrv()

json_file = open('model_ecg_nn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

f1_scores = []
accuracies = []
for s in parts:
    loaded_model.load_weights(f'nn_model_ecg_hrv_val_part{s}.h5')
    preds = loaded_model.predict(X_wesad_hrv)
    state = np.round(preds)

    print(classification_report(ys_wesad.astype(int), state))
    sub_f1 = f1_score(ys_wesad.astype(int), state, average='macro')
    sub_acc = accuracy_score(ys_wesad.astype(int), state)
    f1_scores.append(sub_f1)
    accuracies.append(sub_acc)

print(np.mean(accuracies))
print(np.mean(f1_scores))

