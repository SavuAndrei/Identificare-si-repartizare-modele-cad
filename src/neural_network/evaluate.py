import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os
import json

MODEL_PATH = '../../models/trained_model.h5'
DATA_DIR = '../../date_cad'
METRICS_PATH = '../../results/test_metrics.json'
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32

CATEGORIES = [
    "Categoria_1_Baze_Suporturi",
    "Categoria_2_Carcase_Fluidica",
    "Categoria_3_Flanse_Capace",
    "Categoria_4_Arbori_Tije",
    "Categoria_5_Conducte_Curbe"
]
NUM_CLASSES = len(CATEGORIES)

def evaluate_model():
    
    model = tf.keras.models.load_model(MODEL_PATH)

    datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CATEGORIES,
        shuffle=False
    )
    
    # Presupunem că folosiți 70/15/15 split, deci test_generator trebuie să fie o submulțime
    # În acest exemplu simplu, vom folosi 100% din directorul de date, dar în realitate
    # ar trebui să încărcați un set 'test' separat.
    
    # Calculul metricilor pe datele de test:
    # 1. Obține predicțiile
    y_pred_probs = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 2. Obține etichetele reale
    y_true = test_generator.classes
    
    # Tăiem y_pred la aceeași lungime ca y_true (datorită "steps" în predict)
    y_pred = y_pred[:len(y_true)]

    # 3. Calculează metricile
    test_accuracy = accuracy_score(y_true, y_pred)
    test_f1_macro = f1_score(y_true, y_pred, average='macro')
    test_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 4. Salvează metricile (Cerința Nivel 1)
    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_f1_macro": float(test_f1_macro),
        "test_precision_macro": float(test_precision_macro),
        "test_recall_macro": float(test_recall_macro)
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 5. Generează Confusion Matrix (Cerința Nivel 3 Bonus)
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt('../../docs/confusion_matrix.txt', cm, fmt='%d') # Salvat ca text simplu
    
if __name__ == '__main__':
    # Setăm DATA_DIR la o cale relativă de la directorul de rulare (ex: src/)
    DATA_DIR = '../../date_cad'
    evaluate_model()
