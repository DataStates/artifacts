
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
...
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
from keras.optimizers import adam_v2
import json
import sys
import ray
import pandas as pd
import tensorflow as tf
import random
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
from tensorflow.keras import layers
from deephyper.nas.metrics import r2
from deephyper.benchmark.nas.linearReg.load_data import load_data
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
def load_preproc_nt3_data_from_file(train_path, test_path, num_classes):
    global X_train
    global Y_train
    global X_test
    global Y_test
    print('Loading data...')
    print("trian path: ")
    print(train_path)
    start_time = time.time()
    df_train = (pd.read_csv(train_path, header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path, header=None).values).astype('float32')
    end_time = time.time()
    print('time to read: ', end='')
    print(end_time - start_time)
    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    start_time = time.time()
    seqlen = df_train.shape[1]

    df_y_train = df_train[:, 0].astype('int')
    df_y_test = df_test[:, 0].astype('int')

    # only training set has noise
    Y_train = to_categorical(df_y_train, num_classes)
    Y_test = to_categorical(df_y_test, num_classes)

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    # this reshaping is critical for the Conv1D to work
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    end_time = time.time()
    print('time to proc: ', end='')
    print(end_time - start_time)
    return (X_train, Y_train), (X_test, Y_test)



rand_seed  = 2
if rand_seed is not None:
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)

train_path='/Users/admin/experiments/datahere/nt_train2.csv'
test_path='/Users/admin/experiments/datahere/nt_test2.csv'
(X_train, y_train), (X_test, y_test) = load_preproc_nt3_data_from_file(train_path, test_path, 2) 

dire = 'nt3_s32_p64_1epoch_expbaseline_1gpu_out/'
new_model = tf.keras.models.load_model(dire+'3-25-0-1-3-0-1-7.h5', custom_objects={'r2': r2})
print(new_model.summary())
print(new_model.metrics.__str__)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=2, min_delta=0.002)


new_model_2=tf.keras.models.clone_model(new_model)
new_model_3=tf.keras.models.clone_model(new_model)
    
new_model_2.build((None, X_train.shape[1]))
new_model_2.compile(optimizer=adam_v2.Adam(0.001),  loss='categorical_crossentropy', metrics='acc')
history = new_model_2.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=100, callbacks=[es])

new_model_3.build((None, X_train.shape[1]))
new_model_3.compile(optimizer=adam_v2.Adam(0.001),  loss='categorical_crossentropy', metrics='acc')
new_model_3.set_weights(new_model.get_weights())
history2 = new_model_3.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=100, callbacks=[es])

plt.plot(history.history['loss'], label='random_weights')
plt.plot(history2.history['loss'], label='transferred_weights')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
