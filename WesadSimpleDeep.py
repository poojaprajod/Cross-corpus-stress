import os
import pickle
import numpy as np
import math
# import neurokit as nk
# import seaborn as sns
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Activation, LSTM
from tensorflow.keras.layers import Add, Input, GlobalMaxPool1D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import classification_report
from scipy import signal
from scipy.stats import zscore

import gc

from tensorflow.keras import backend as K
import tensorflow as tf

import random

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

num_frames = 10
frame_length = 256
channels = 1  # Just ECG for now

batch_size = 128
epochs = 200


def Deep_Simple_ECG_model(num_classes=3):
    model = Sequential()
    model.add(Input(shape=(int(num_frames * frame_length), channels)))
    # model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=32, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=32, activation='relu'))
    model.add(MaxPooling1D(pool_size=8, strides=2))

    model.add(Conv1D(filters=64, kernel_size=16, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=16, activation='relu'))
    model.add(MaxPooling1D(pool_size=8, strides=2))

    model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))

    model.add(GlobalMaxPool1D())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def full_model(num_classes=3):
    model_ecg = Deep_Simple_ECG_model(num_classes)
    return model_ecg


def prepare_data(subs):
    X_ecg = []
    # X_eda = []
    y = []
    for i in subs:
        subject = 'S' + str(i)
        sub_ecg_data = np.load(f'D:/StressSignals/{subject}_ecg_ffilt.npy', allow_pickle=True)
        sub_min = np.min(sub_ecg_data[0][256 * 60: 256 * 360].flatten())
        sub_max = np.max(sub_ecg_data[0][256 * 60: 256 * 360].flatten())
        for label in [0, 1, 2]:  # baseline, stress, amusement
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

    y = to_categorical(y)

    return X_ecg, y


def step_decay(epoch, lr):
    lrate = lr
    if ((epoch + 1) % 25) == 0:
        lrate = lr * 0.5
    return lrate


# train_subs = [2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17]
# test_subs = [4, 11, 16]
model_full = full_model(num_classes=2)
model_full.save_weights('init_deep_simple_ecg_model_weights.h5')

subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
for s in subs:
    test_subs = [s]
    train_subs = list(set(subs) - set(test_subs))

    X_eda_train, y_train = prepare_data(train_subs)  # ECG data. Keeping the naming error as it was found much later
    X_eda_test, y_test = prepare_data(test_subs)

    trained = True
    if not trained:
        model_full.load_weights(f'init_deep_simple_ecg_model_weights.h5')

        weights = dict(zip(np.unique(y_train), compute_class_weight('balanced', classes=np.unique(y_train), y=np.argmax(y_train, axis=-1))))

        model_full.compile(loss='categorical_crossentropy', optimizer=Adadelta(learning_rate=1.0), metrics=['accuracy'])

        callbacks = [  # EarlyStopping(monitor='val_acc', patience=10, min_delta=0.0001, mode='max'),
            ModelCheckpoint(filepath=f'best_model_ecg_val{s}_znorm_deep_simple_ecg.h5',
                            monitor='val_loss', save_best_only=True)  # , LearningRateScheduler(step_decay)
        ]
        model_full.fit(X_eda_train, y_train, validation_data=(X_eda_test, y_test), epochs=epochs,
                       callbacks=callbacks, class_weight=weights,
                       batch_size=batch_size, shuffle=True)
        # max_queue_size=1, workers=1, use_multiprocessing=False, shuffle=False)

        model_json = model_full.to_json()
        with open("model_ecg_deep_simple_ecg.json", "w") as json_file:
            json_file.write(model_json)

    model_full.load_weights(f'best_model_ecg_val{s}_znorm_deep_simple_ecg.h5')
    preds = model_full.predict(X_eda_test)
    state = np.argmax(preds, axis=1)
    true = np.argmax(y_test, axis=-1)
    # state = np.round(preds)
    # true = np.round(y_test)
    print(f'subject {s}')
    print(classification_report(true, state))

    del X_eda_train, y_train
    del X_eda_test, y_test

    gc.collect()