import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import pandas as pd
from .model import create_compiled_model

DATA_DIR = '../../date_cad'
MODEL_PATH = '../../models/trained_model.h5'
HISTORY_PATH = '../../results/training_history.csv'
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32
EPOCHS = 50

CATEGORIES = [
    "Categoria_1_Baze_Suporturi",
    "Categoria_2_Carcase_Fluidica",
    "Categoria_3_Flanse_Capace",
    "Categoria_4_Arbori_Tije",
    "Categoria_5_Conducte_Curbe"
]

def train_model():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        classes=CATEGORIES,
        seed=42
    )

    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=CATEGORIES,
        seed=42
    )
    
    model = create_compiled_model()

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5,
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_PATH, index=False)

if __name__ == '__main__':
    # Crează folderele necesare
    os.makedirs('../../results', exist_ok=True)
    os.makedirs('../../models', exist_ok=True)
    
    # Adaugă '..\\' în DATA_DIR dacă rulați din src/neural_network
    DATA_DIR = '../../date_cad'
    
    # Rulăm funcția principală
    train_model()
