import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configurare globală ---
DATA_DIR = 'date_cad'
MODEL_PATH = 'models/trained_cad_classifier.h5'
HISTORY_PATH = 'results/training_history.csv'
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
BATCH_SIZE = 32
EPOCHS = 50

# Clasele exacte
CATEGORIES = [
    "CATEGORIA_1_Surub",
    "CATEGORIA_2_Bolt_De_Ghidaj",
    "CATEGORIA_3_Piulita",
    "CATEGORIA_4_Saiba",
]
NUM_CLASSES = len(CATEGORIES)

# --- Data generators ---
def create_data_generators():
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

    print(f"Mostre antrenare: {train_generator.samples}, validare: {validation_generator.samples}")
    return train_generator, validation_generator

def create_eval_generator():
    eval_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    eval_generator = eval_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=CATEGORIES,
        seed=42,
        shuffle=False
    )

    return eval_generator

# --- Model CNN ---
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Grafic Loss ---
def plot_loss(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')

    plt.title('Grafic de Convergență (Loss Curve)')
    plt.xlabel('Epoci')
    plt.ylabel('Eroare (Loss)')
    plt.legend()
    plt.grid(True)

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/loss_curve.png')
    plt.show()

def plot_metrics_evolution(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')

    plt.title('Evoluție Metrici (Accuracy & Loss)')
    plt.xlabel('Epoci')
    plt.ylabel('Valoare')
    plt.legend()
    plt.grid(True)

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/metrics_evolution.png')
    plt.show()

def plot_confusion_matrix(model, eval_generator):
    steps = math.ceil(eval_generator.samples / BATCH_SIZE)
    preds = model.predict(eval_generator, steps=steps)
    y_pred = np.argmax(preds, axis=1)
    y_true = eval_generator.classes

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Validation)')
    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, CATEGORIES, rotation=45, ha='right')
    plt.yticks(tick_marks, CATEGORIES)

    thresh = conf_mat.max() / 2.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(
                j, i, format(conf_mat[i, j], 'd'),
                ha='center', va='center',
                color='white' if conf_mat[i, j] > thresh else 'black'
            )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    os.makedirs('docs', exist_ok=True)
    plt.savefig('docs/confusion_matrix_optimized.png')
    plt.show()

# --- Train și salvare ---
def train_and_save_model(model, train_gen, val_gen):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        callbacks=[early_stopping]
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model salvat: {MODEL_PATH}")

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    pd.DataFrame(history.history).to_csv(HISTORY_PATH, index=False)
    print(f"Istoric salvat: {HISTORY_PATH}")

    plot_loss(history)
    plot_metrics_evolution(history)

    eval_gen = create_eval_generator()
    plot_confusion_matrix(model, eval_gen)

# --- Main ---
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    if not os.path.exists(DATA_DIR) or not any(os.listdir(DATA_DIR)):
        print(f"EROARE: '{DATA_DIR}' nu există sau este gol")
    else:
        train_gen, val_gen = create_data_generators()
        input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        model = create_cnn_model(input_shape, NUM_CLASSES)
        train_and_save_model(model, train_gen, val_gen)
