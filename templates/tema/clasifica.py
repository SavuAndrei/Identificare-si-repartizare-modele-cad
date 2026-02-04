import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# --- Configurare ---
MODEL_PATH = 'models/trained_cad_classifier.h5'
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

CATEGORIES = [
    "Categoria_1_Baze_Suporturi",
    "Categoria_2_Carcase_Fluidica",
    "Categoria_3_Flanse_Capace",
    "Categoria_4_Arbori_Tije",
    "Categoria_5_Conducte_Curbe"
]

DESCRIERII_COMPLETE = [
    "Categoria 1: Baze, Suporturi și Lagăre Despicate",
    "Categoria 2: Carcase și Corpuri de Fluidică",
    "Categoria 3: Flanșe, Capace și Elemente de Cuplare",
    "Categoria 4: Arbori, Tije și Piese Liniare de Control",
    "Categoria 5: Componente de Conductă și Curbe"
]

# Încărcăm modelul o singură dată, la pornirea serverului
if os.path.exists(MODEL_PATH):
    print("Se încarcă modelul AI...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    print(f"ATENȚIE: Modelul nu a fost găsit la {MODEL_PATH}")

def predict(image_path):
    """
    Funcția apelată de app.py pentru a clasifica o imagine urcată de utilizator.
    """
    if model is None:
        return "Eroare: Modelul AI nu este încărcat."

    try:
        # 1. Preprocesare
        img = load_img(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # 2. Predicție
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]

        # 3. Formatare rezultat pentru interfața web
        nume_piesa = DESCRIERII_COMPLETE[predicted_class_index]
        return f"{nume_piesa} (Siguranță: {confidence*100:.2f}%)"

    except Exception as e:
        return f"Eroare la procesare: {str(e)}"