import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

MODEL_PATH = '../../models/trained_model.h5'
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
    "Baze, Suporturi și Lagăre Despicate",
    "Carcase și Corpuri de Fluidică",
    "Flanșe, Capace și Elemente de Cuplare",
    "Arbori, Tije și Piese Liniare de Control",
    "Componente de Conductă și Curbe"
]

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        return None

def preprocess_image(uploaded_file):
    img = load_img(uploaded_file, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def main():
    st.title("SIA: Clasificarea Modelelor CAD (Etapa 5 Demo)")

    model = load_trained_model()

    if model is None:
        st.error("Modelul antrenat nu a fost găsit. Rulați `train.py`.")
        return

    uploaded_file = st.file_uploader("Încărcați o imagine CAD pentru clasificare", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Model CAD încărcat', use_column_width=True)
        
        # Starea PREPROCESS
        input_data = preprocess_image(uploaded_file)
        
        # Starea RN_INFERENCE
        predictions = model.predict(input_data)
        
        # Starea THRESHOLD_CHECK & ALERT (Vizualizare)
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        
        st.markdown("---")
        st.subheader("Rezultat Clasificare (Inferență Reală)")
        
        st.metric(
            label="Categoria Prezisa", 
            value=f"{CATEGORIES[predicted_index]}", 
            delta=f"{DESCRIERII_COMPLETE[predicted_index]}"
        )
        st.progress(confidence)
        st.write(f"Confidență: **{confidence*100:.2f}%**")

        st.markdown("#### Toate Probabilitățile:")
        
        for i, (cat, desc) in enumerate(zip(CATEGORIES, DESCRIERII_COMPLETE)):
            st.text(f"- {cat} ({desc}): {predictions[0][i]*100:.2f}%")

if __name__ == '__main__':
    main()
