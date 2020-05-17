# Tipo de modelo
from tensorflow.keras.models import Sequential
#from tensorflow.keras import Model
import tensorflow as tf

# Capas en MLP
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, SpatialDropout1D
from tensorflow.keras import Input

# Capas en LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling1D
from tensorflow.keras.layers import SeparableConv1D

from tensorflow.keras import regularizers

from tensorflow.keras.layers import GRU

import tensorflow_hub as hub

def MLPClassifier(layers, units, dropout_rate, input_shape, num_classes, sparse = True):
    model = Sequential()
    if sparse:
        model.add(Input(
            batch_size = 256,
            shape = input_shape, 
            sparse = True
        ))
    
    model.add(Dense(
        units = units, 
        activation = 'relu' 
    ))

    model.add(Dropout(
        rate = dropout_rate
    ))

    for _ in range(layers-1):
        model.add(Dense(
            units = units, 
            activation = 'relu'
        ))
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

def GRUClassifier(layers, embedding_dim, dropout_rate, input_shape, num_classes, num_features):
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
        model.add(GRU(embedding_dim, return_sequences = True))
        model.add(Dropout(
            rate = dropout_rate
        ))

    model.add(GlobalAveragePooling1D())
    #model.add(GRU(128))

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
        filters = filters * 2,
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

if __name__ == "__main__":
    model = SC1DClassifier(
        layers = 2,
        embedding_dim = 100,
        filters = 64,
        dropout_rate = 0.3,
        input_shape = (200,),
        num_classes = 60,
        num_features = 1000
    )
    print(model.summary())