import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
NUM_CLASSES = 5

def create_compiled_model():
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    model = create_compiled_model()
    model.summary()
    # Salvează modelul neantrenat, cerința Etapa 4
    model.save('../../models/untrained_model.h5')
