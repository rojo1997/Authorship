# Tipo de modelo
from keras import Model
import tensorflow as tf

print (tf.version)

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

    for _ in range(layers - 1):
        x = Dense(
            units = units, 
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
    ) (x if layers > 1 else sequence_input)

    model = Model(sequence_input, preds)

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = ['acc']
    )

    if verbose != False:
        print(model.summary())

    return(model)

def GRUClassifier(layers, embedding_dim, units, dropout_rate, regularize, input_shape, num_classes, num_features):
    model = Sequential()

    model.add(Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0],
        embeddings_regularizer = regularizers.l2(regularize)
    ))

    model.add(Dropout(
        rate = dropout_rate
    ))

    for _ in range(layers):
        model.add(GRU(
            units = units, 
            return_sequences = True,
            bias_initializer = 'random_uniform',
            kernel_regularizer = regularizers.l2(regularize), 
            recurrent_regularizer = regularizers.l2(regularize),
            bias_regularizer = regularizers.l2(regularize)
        ))
        model.add(Dropout(
            rate = dropout_rate
        ))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid'
    ))

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['acc']
    )

    return(model)

def LSTMClassifier(
    embedding_dim, 
    dropout_rate, 
    input_shape, 
    num_classes, 
    num_features, 
    filters = 64, 
    kernel_size = 3,
    pool_size = 2):

    model = Sequential()

    model.add(Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0]
    ))

    model.add(Dropout(
        rate = dropout_rate
    ))

    for _ in range(layers):
        model.add(LSTM(
            embedding_dim,
            dropout = dropout_rate,
            recurrent_dropout = dropout_rate
        ))
        model.add(Dropout(
            rate = dropout_rate
        ))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid'
    ))

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['acc']
    )

    return(model)

def SC1DClassifier(layers, embedding_dim, filters, kernel_size, regularize, dropout_rate, input_shape, num_classes, num_features):
    model = Sequential()

    model.add(Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0],
        embeddings_regularizer = regularizers.l2(regularize)
    ))

    for _ in range(layers - 1):
        model.add(Dropout(
            rate = dropout_rate
        ))
        model.add(SeparableConv1D(
            filters = filters,
            padding = 'same',
            kernel_size = kernel_size,
            activation = 'relu',
            bias_initializer = 'random_uniform',
            depthwise_initializer = 'random_uniform',
            depthwise_regularizer = regularizers.l2(regularize),
            bias_regularizer = regularizers.l2(regularize)
        ))
        model.add(MaxPooling1D(2))

    model.add(SeparableConv1D(
        filters = filters,
        padding = 'same',
        kernel_size = kernel_size,
        activation = 'relu',
        bias_initializer = 'random_uniform',
        depthwise_initializer = 'random_uniform',
        depthwise_regularizer = regularizers.l2(0.0001),
        bias_regularizer = regularizers.l2(0.0001)
    ))
    
    model.add(GlobalAveragePooling1D())

    model.add(Dropout(
        rate = dropout_rate
    ))

    model.add(Dense(
        units = num_classes if num_classes > 2 else 1, 
        activation = 'softmax' if num_classes > 2 else 'sigmoid'
    ))

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['acc']
    )
    return(model)

def Conv1D_SingleKernel(
        input_shape,
        num_classes,
        num_features = 20000,
        embedding_dim = 256,
        layers = 1,
        filters = 64, 
        dropout_rate = 0.1,
        kernel_size = 3, 
        regularization = 1e-5, 
        trainable = False,
        dtype = np.int32,
        optimizer = Adam(),
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
        trainable = trainable,
        embeddings_initializer = "uniform"
    ) (sequence_input)

    x = Dropout(
        rate = dropout_rate
    ) (embedded_sequences)

    for _ in range(layers - 1):
        x = Conv1D(
            filters = filters,
            padding = 'same',
            kernel_size = kernel_size,
            activation = 'relu',
            bias_initializer = 'random_uniform',
            kernel_regularizer = regularizers.l2(regularization),
            bias_regularizer = regularizers.l2(regularization)
        ) (x)

        x = MaxPooling1D() (x)

    x = Conv1D(
        filters = filters,
        padding = 'same',
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

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = ['acc']
    )

    return(model)

def Conv1D_MultiKernel(
        input_shape,
        num_classes,
        num_features = 30000,
        embedding_dim = 512,
        fkl = [(256,1,1),(128,3,1),(64,5,1),(32,7,1),(16,9,1)],
        dropout_rate = 0.1,
        regularization = 1e-5, 
        trainable = False,
        dtype = np.int16,
        optimizer = Adam(),
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
        trainable = trainable,
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
            padding = 'same',
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
                padding = 'same',
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

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = ['acc']
    )

    return(model)

if __name__ == "__main__":
    model = Conv1D_MultiKernel(
        input_shape = (30,),
        num_classes = 3,
        fkl = [(256,1,2),(256,3,2),(128,5,3)]
    )
    model = MLPClassifier(
        input_shape = (30,),
        num_classes = 3
    )
    print(model.summary())

    from scipy import sparse
    from keras.utils import to_categorical

    x = np.random.rand(500,30)
    x = sparse.csr_matrix(x)
    y = np.random.randint(low = 0, high = 3, size = 500)
    #y = to_categorical(y, num_classes = 3)

    model.fit(
        x = x,
        y = y,
        batch_size = 128,
        epochs = 5
    )