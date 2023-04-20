import numpy as np
import pandas as pd
import tensorflow as tf
import gzip
import h5py

from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.utils import to_categorical

import os
import os.path

X_train = None
Y_train = None
X_test = None
Y_test = None


HERE = os.path.dirname(os.path.abspath(__file__))


def attn_load_data_h5(split="train", h5f_path=None):

    #h5f_path = os.path.join("/lus/grand/projects/VeloC/mmadhya1", "training_attn.h5")
    h5f = h5py.File(h5f_path, "r")

    if split == "train":
        X, y = h5f["X_train"][:], h5f["Y_train"][:]
    elif split == "valid":
        X, y = h5f["X_val"][:], h5f["Y_val"][:]
    elif split == "test":
        X, y = h5f["X_test"][:], h5f["Y_test"][:]

    h5f.close()

    # y = np.argmax(y, axis=1)

    return X, y


def attn_load_data_test():

    X_train, y_train = attn_load_data_h5("train")
    X_valid, y_valid = attn_load_data_h5("valid")
    X_train = np.concatenate([X_train, X_valid], axis=0)
    y_train = np.concatenate([y_train, y_valid], axis=0)
    X_test, y_test = attn_load_data_h5("test")

    return (X_train, y_train), (X_test, y_test)


def attn_load_data(train_path):

    X_train, y_train = attn_load_data_h5("train", h5f_path=train_path)
    X_valid, y_valid = attn_load_data_h5("valid", h5f_path=train_path)

    return (X_train, y_train), (X_valid, y_valid)



def load_data_npz_gz(test=False):

    head_dir = '/lus/grand/projects/VeloC/datastates/'
    if test:
        fname = os.path.join(head_dir, "testing_combo.npy.gz")
    else:
        fname = os.path.join(head_dir, "training_combo.npy.gz")

    with gzip.GzipFile(fname, "rb") as f:
        data = np.load(f, allow_pickle=True).item()

    X, y = data["X"], data["y"]

    return X, y

def combo_load_data_test():

    X_train, y_train = load_data_npz_gz()
    X_test, y_test = load_data_npz_gz(test=True)

    return (X_train, y_train), (X_test, y_test)




def combo_load_data(train_path):

    X, y = load_data_npz_gz()

    for Xi in X:
        assert Xi.shape[0] == y.shape[0]

    # Train/Validation split
    rs = np.random.RandomState(42)
    valid_size = 0.2
    indexes = np.arange(0,y.shape[0])
    rs.shuffle(indexes)
    curr = int((1-valid_size)*y.shape[0])
    indexes_train, indexes_valid = indexes[:curr], indexes[curr:]
    X_train, X_valid = [], []
    for Xi in X:
        X_train.append(Xi[indexes_train])
        X_valid.append(Xi[indexes_valid])
    y_train, y_valid = y[indexes_train], y[indexes_valid]

    print("Train")
    print("Input")
    for Xi in X_train:
        print(np.shape(Xi))

    print("Output")
    print(np.shape(y_train))

    print("Valid")
    print("Input")
    for Xi in X_valid:
        print(np.shape(Xi))

    print("Output")
    print(np.shape(y_train))

    return (X_train, y_train), (X_valid, y_valid)







def load_uno_data_fake():
    global X_train
    global Y_train
    global X_test
    global Y_test

    x_train_shape = [(9588, 1), (9588, 942), (9588, 5270), (9588, 2048)]
    y_train_shape = (9588,)
    x_test_shape = [(2397, 1), (2397, 942), (2397, 5270), (2397, 2048)]
    y_test_shape = (2397,)
    X_train = [np.zeros(i) for i in x_train_shape]
    Y_train = np.zeros(y_train_shape)

    X_test = [np.zeros(i) for i in x_test_shape]
    Y_test = np.zeros(y_test_shape)
    return (X_train, Y_train), (X_test, Y_test)


def load_uno_data_from_file(dataset_path):
    # NOTE: This assumess that the data is already downloaded.
    # To download, pls follow instructions on https://github.com/ECP-CANDLE/Benchmarks/blob/master/Pilot1/Uno.
    global X_train
    global Y_train
    global X_test
    global Y_test
    pickle_datapath = os.path.join(dataset_path, "uno_data.p")
    print(f"load data from {pickle_datapath}")
    data2 = pickle.load(open(pickle_datapath, "rb"))
    (X_train, Y_train) = data2[0]
    (X_test, Y_test) = data2[1]
    return (X_train, Y_train), (X_test, Y_test)


def load_preproc_nt3_data_from_file(train_path, test_path, num_classes):
    # NOTE: This assumess that the data is already downloaded but not yet preprocessed.
    # To download, pls follow instructions on https://github.com/ECP-CANDLE/Benchmarks/blob/master/Pilot1/NT3/nt3_baseline_keras2.py.
    # specifically, look at candle.get_file
    global X_train
    global Y_train
    global X_test
    global Y_test

    """
    X_train = np.load('X_train.npy', 'r')
    Y_train = np.load('Y_train.npy', 'r')
    X_test = np.load('X_test.npy', 'r')
    Y_test = np.load('Y_test.npy', 'r')
    """
    print("Loading data...")
    print("train path: ", train_path)
    if X_train is None:
        df_train = (pd.read_csv(train_path, header=None).values).astype("float32")
        df_test = (pd.read_csv(test_path, header=None).values).astype("float32")

        print("df_train shape:", df_train.shape)
        print("df_test shape:", df_test.shape)

        seqlen = df_train.shape[1]

        df_y_train = df_train[:, 0].astype("int")
        df_y_test = df_test[:, 0].astype("int")

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

        X_train = mat[: X_train.shape[0], :]
        X_test = mat[X_train.shape[0] :, :]

        # this reshaping is critical for the Conv1D to work
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
    return (X_train, Y_train), (X_test, Y_test)
