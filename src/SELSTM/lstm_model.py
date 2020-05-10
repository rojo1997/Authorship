from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import GRU

def lstm_model(
    layers = 2,
    embedding_dim = 64,
    dropout_rate = 0.2,
    input_shape = (100,),
    num_classes = 60,
    num_features = 20000
    ):
    
    model = models.Sequential()

    model.add(Embedding(
        input_dim = num_features,
        output_dim = embedding_dim,
        input_length = input_shape[0]
    ))

    """for _ in range(layers-1):
        model.add(LSTM(
            embedding_dim,
            return_sequences = True,
            input_shape = [None, 1]
        ))"""

    model.add(GRU(embedding_dim, return_sequences = True))
    model.add(GRU(embedding_dim))
    
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['acc']
    )
    print(model.summary())
    return model

if __name__ == "__main__":
    model = lstm_model(
        layers = 2,
        embedding_dim = 100,
        dropout_rate = 0.2,
        input_shape = (300,),
        num_classes = 60,
        num_features = 20000
    )
    print(model.summary())