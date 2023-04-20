global X_train
global y_train
global X_test
global y_test


def read_preproc_nt3_data_from_file(train_path, test_path, num_classes):
    print("Loading data...")
    print("train path: ", train_path)
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
    # return (X_train, Y_train), (X_test, Y_test)
