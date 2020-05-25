from keras import Model
import tensorflow as tf

from keras.optimizers import (
    Adam, 
    RMSprop, 
    Nadam
)

from keras.layers import (
    Input,
    Dense,
    Dropout,
    SpatialDropout1D,
    Embedding,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    AveragePooling1D,
    MaxPooling1D,
    LSTM,
    GRU,
    Conv1D,
    SeparableConv1D,
    Concatenate
)

from keras import regularizers

from keras.initializers import Constant

import numpy as np

def MLPClassifier(
        input_shape,
        num_classes,
        layers = 1, 
        units = 128, 
        dropout_rate = 0.1,
        batch_size = 128,
        regularization = 1e-5,
        dtype = np.float32,
        optimizer = Adam(),
        verbose = False
    ):

    sequence_input = Input(
        shape = input_shape,
        dtype = dtype
    )

    x = None

    for _ in range(layers):
        x = Dense(
            units = units if isinstance(units,int) else units[_], 
            activation = 'relu',
            kernel_regularizer = regularizers.l2(regularization),
            bias_regularizer = regularizers.l2(regularization), 
        ) (sequence_input if _ == 0 else x)

        x = Dropout(
            rate = dropout_rate
        ) (x)

    preds = Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid',
        kernel_regularizer = regularizers.l2(regularization),
        bias_regularizer = regularizers.l2(regularization),
    ) (sequence_input if layers == 0 or layers == None else x)

    model = Model(sequence_input, preds)

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = ['acc']
    )

    if verbose:
        print(model.summary())

    return(model)

def GRUClassifier(
        input_shape,
        num_classes,
        num_features,
        embedding_dim = 256,
        layers = 1,
        units = 64, 
        dropout_rate = 0.1,
        regularization = 1e-5, 
        embedding_trainable = False,
        dtype = np.int32,
        optimizer = Adam(),
        metrics = ['acc'],
        verbose = False
    ):

    sequence_input = Input(
        shape = input_shape,
        dtype = dtype
    )

    embedded_sequences = Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0],
        trainable = embedding_trainable,
        embeddings_initializer = "uniform"
    ) (sequence_input)

    x = Dropout(
        rate = dropout_rate
    ) (embedded_sequences)

    for _ in range(layers - 1):
        x = GRU(
            units = units,
            activation = "tanh",
            recurrent_activation = "sigmoid",
            bias_initializer = 'zeros',
            kernel_regularizer = regularizers.l2(regularization), 
            recurrent_regularizer = regularizers.l2(regularization),
            bias_regularizer = regularizers.l2(regularization),
            dropout = dropout_rate,
            recurrent_dropout = dropout_rate,
            return_sequences = True,
        ) (x)

        x = MaxPooling1D() (x)

        x = Dropout(
            rate = dropout_rate
        ) (x)


    x = GlobalMaxPooling1D() (x)

    x = Dropout(
        rate = dropout_rate
    ) (x)

    preds = Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid',
        kernel_regularizer = regularizers.l2(regularization),
        bias_regularizer = regularizers.l2(regularization),
    ) (x)

    model = Model(sequence_input, preds)

    if verbose:
        print(model.summary())

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = metrics
    )

    return(model)

def LSTMClassifier(
        input_shape,
        num_classes,
        num_features,
        embedding_dim = 256,
        layers = 1,
        units = 64, 
        dropout_rate = 0.1,
        regularization = 1e-5, 
        embedding_trainable = False,
        dtype = np.int32,
        optimizer = Adam(),
        metrics = ['acc'],
        verbose = False
    ):

    sequence_input = Input(
        shape = input_shape,
        dtype = dtype
    )

    embedded_sequences = Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0],
        trainable = embedding_trainable,
        embeddings_initializer = "uniform"
    ) (sequence_input)

    x = Dropout(
        rate = dropout_rate
    ) (embedded_sequences)

    for _ in range(layers - 1):
        x = LSTM(
            units = units,
            activation = "tanh",
            recurrent_activation = "sigmoid",
            bias_initializer = 'zeros',
            kernel_regularizer = regularizers.l2(regularization), 
            recurrent_regularizer = regularizers.l2(regularization),
            bias_regularizer = regularizers.l2(regularization),
            dropout = dropout_rate,
            recurrent_dropout = dropout_rate,
            return_sequences = True,
        ) (x)

        x = MaxPooling1D() (x)

        x = Dropout(
            rate = dropout_rate
        ) (x)


    x = GlobalMaxPooling1D() (x)

    x = Dropout(
        rate = dropout_rate
    ) (x)

    preds = Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid',
        kernel_regularizer = regularizers.l2(regularization),
        bias_regularizer = regularizers.l2(regularization),
    ) (x)

    model = Model(sequence_input, preds)

    if verbose:
        print(model.summary())

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = metrics
    )

    return(model)

def SC1DClassifier(
        input_shape,
        num_classes,
        num_features,
        embedding_dim = 256,
        layers = 1,
        filters = 64, 
        dropout_rate = 0.1,
        kernel_size = 3, 
        regularization = 1e-5, 
        embedding_trainable = False,
        dtype = np.int32,
        optimizer = Adam(),
        metrics = ['acc'],
        verbose = False
    ):

    sequence_input = Input(
        shape = input_shape,
        dtype = dtype
    )
    embedded_sequences = Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0],
        trainable = embedding_trainable,
        embeddings_initializer = "uniform"
    ) (sequence_input)

    x = Dropout(
        rate = dropout_rate
    ) (embedded_sequences)

    for _ in range(layers - 1):
        x = SeparableConv1D(
            filters = filters,
            padding = 'valid',
            kernel_size = kernel_size,
            activation = 'relu',
            bias_initializer = 'random_uniform',
            kernel_regularizer = regularizers.l2(regularization),
            bias_regularizer = regularizers.l2(regularization)
        ) (x)

        x = MaxPooling1D() (x)
        
        x = Dropout(
            rate = dropout_rate
        ) (x)

    x = SeparableConv1D(
        filters = filters,
        padding = 'valid',
        kernel_size = kernel_size,
        activation = 'relu',
        bias_initializer = 'random_uniform',
        kernel_regularizer = regularizers.l2(regularization),
        bias_regularizer = regularizers.l2(regularization)
    ) (x)

    x = GlobalMaxPooling1D() (x)

    x = Dropout(
        rate = dropout_rate
    ) (x)

    preds = Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid',
        kernel_regularizer = regularizers.l2(regularization),
        bias_regularizer = regularizers.l2(regularization),
    ) (x)

    model = Model(sequence_input, preds)

    if verbose:
        print(model.summary())

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = metrics
    )

    return(model)

def C1DSingleClassifier(
        input_shape,
        num_classes,
        num_features,
        embedding_dim = 256,
        layers = 1,
        filters = 64, 
        dropout_rate = 0.1,
        kernel_size = 3, 
        regularization = 1e-5, 
        embedding_trainable = False,
        dtype = np.int32,
        optimizer = Adam(),
        metrics = ['acc'],
        verbose = False
    ):

    sequence_input = Input(
        shape = input_shape,
        dtype = dtype
    )
    embedded_sequences = Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0],
        trainable = embedding_trainable,
        embeddings_initializer = "uniform"
    ) (sequence_input)

    x = Dropout(
        rate = dropout_rate
    ) (embedded_sequences)

    for _ in range(layers - 1):
        x = Conv1D(
            filters = filters,
            padding = 'valid',
            kernel_size = kernel_size,
            activation = 'relu',
            bias_initializer = 'random_uniform',
            kernel_regularizer = regularizers.l2(regularization),
            bias_regularizer = regularizers.l2(regularization)
        ) (x)

        x = MaxPooling1D() (x)
        
        x = Dropout(
            rate = dropout_rate
        ) (x)

    x = Conv1D(
        filters = filters,
        padding = 'valid',
        kernel_size = kernel_size,
        activation = 'relu',
        bias_initializer = 'random_uniform',
        kernel_regularizer = regularizers.l2(regularization),
        bias_regularizer = regularizers.l2(regularization)
    ) (x)

    x = GlobalMaxPooling1D() (x)

    x = Dropout(
        rate = dropout_rate
    ) (x)

    preds = Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid',
        kernel_regularizer = regularizers.l2(regularization),
        bias_regularizer = regularizers.l2(regularization),
    ) (x)

    model = Model(sequence_input, preds)

    if verbose:
        print(model.summary())

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = metrics
    )

    return(model)

def C1DMultiClassifier(
        input_shape,
        num_classes,
        num_features,
        embedding_dim = 256,
        fkl = [(256,1,1),(128,3,1),(64,5,1),(32,7,1),(16,9,1)],
        dropout_rate = 0.1,
        regularization = 1e-5, 
        embedding_trainable = False,
        dtype = np.int16,
        optimizer = Adam(),
        metrics = ['acc'],
        verbose = False
    ):

    sequence_input = Input(
        shape = input_shape,
        dtype = dtype
    )

    embedded_sequences = Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0],
        trainable = embedding_trainable,
        embeddings_initializer = "uniform"
    ) (sequence_input)

    h = []
    for f,k,l in fkl:
        x = Dropout(
                rate = dropout_rate
        ) (embedded_sequences)

        if k == 1:
            x = GlobalMaxPooling1D() (x)
            x = Dense(
                units = f, 
                activation = 'relu',
                kernel_regularizer = regularizers.l2(regularization),
                bias_regularizer = regularizers.l2(regularization),
            ) (x)
            h.append(x)
            continue
        
        x = Conv1D(
            filters = f,
            padding = 'valid',
            kernel_size = k,
            activation = 'relu',
            bias_initializer = 'random_uniform',
            kernel_regularizer = regularizers.l2(regularization),
            bias_regularizer = regularizers.l2(regularization)
        ) (x)

        for _ in range(l-1):
            x = Dropout(
                rate = dropout_rate
            ) (x)

            x = MaxPooling1D() (x)

            x = Conv1D(
                filters = f,
                padding = 'valid',
                kernel_size = k,
                activation = 'relu',
                bias_initializer = 'random_uniform',
                kernel_regularizer = regularizers.l2(regularization),
                bias_regularizer = regularizers.l2(regularization)
            ) (x)
        
        h.append(GlobalMaxPooling1D() (x))

    x = Concatenate()(h)

    x = Dropout(
        rate = dropout_rate
    ) (x)

    preds = Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid',
        kernel_regularizer = regularizers.l2(regularization),
        bias_regularizer = regularizers.l2(regularization),
    ) (x)

    model = Model(sequence_input, preds)

    if verbose:
        print(model.summary())

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = metrics
    )

    return(model)