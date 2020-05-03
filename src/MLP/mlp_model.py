# Tipo de modelo
from tensorflow.keras.models import Sequential
#from tensorflow.keras import Model
import tensorflow as tf

# Capas que se van a usar en MLP
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Input

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    model = Sequential()
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

if __name__ == "__main__":
    model = mlp_model(
        layers = 3,
        units = 32,
        dropout_rate = 0.2,
        input_shape = (20000,),
        num_classes = 5
    )
    print(model.summary())