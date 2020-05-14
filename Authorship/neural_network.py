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
from tensorflow.keras.layers import GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import SeparableConv1D

from tensorflow.keras.layers import GRU

import tensorflow_hub as hub

def MLPClassifier(layers, units, dropout_rate, input_shape, num_classes, sparse = True):
    model = Sequential()
    if sparse:
        model.add(Input(
            batch_size = 1024,
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

def GRUClassifier(embedding_dim, dropout_rate, input_shape, num_classes, num_features):
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

    model.add(SpatialDropout1D(
        rate = dropout_rate
    ))

    model.add(LSTM(
        200,
        dropout = dropout_rate,
        recurrent_dropout = dropout_rate
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

if __name__ == "__main__":
    model = LSTMClassifier(
        embedding_dim = 100,
        dropout_rate = 0.3,
        input_shape = (200,),
        num_classes = 60,
        num_features = 1000
    )
    print(model.summary())