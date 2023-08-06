import keras
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from data import create_resolution

def create_model():
    model = keras.models.Sequential([
        Conv2D       (16, (3, 3), input_shape=create_resolution() + (3, ), activation='relu', padding='same'),                             
        MaxPooling2D (2, 2),
        Conv2D       (32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D (2, 2),
        Conv2D       (64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D (2, 2),
        Conv2D       (128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D (2, 2),
        Conv2D       (256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D (2, 2),
        Flatten      (),
        Dense        (512, activation='relu'),
        Dropout      (0.5),
        Dense        (1, activation='sigmoid')])
    
    print(model.summary())
    return model
