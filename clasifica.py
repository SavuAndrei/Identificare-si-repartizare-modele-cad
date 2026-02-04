import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import sys

# =========================================
# Configurare - variabile globale
# =========================================
cale_model_cnn = 'models/trained_cad_classifier.h5'  # modelul salvat
dim_img = (32, 32)  # latime x inaltime

# Clasele pieselor
clase = [
    "CATEGORIA_1_Surub",
    "CATEGORIA_2_Bolt_De_Ghidaj",
    "CATEGORIA_3_Piulita",
    "CATEGORIA_4_Saiba",
]

descrieri = [
    "Categoria 1: Surub",
    "Categoria 2: Bolt De Ghidaj",
    "Categoria 3: Piulita",
    "Categoria 4: Saiba",
]

# =========================================
# Functia de clasificare imagine noua
# =========================================
def clasifica_piesa(model, cale_img):
    """
    Clasifica o piesa CAD folosind modelul CNN.
    Returneaza dictionar cu predictia si procentele pe fiecare clasa.
    """
    # verificam daca fisierul exista
    if not os.path.exists(cale_img):
        print(f"EROARE: fisierul nu exista -> {cale_img}")
        return

    try:
        # incarcam imaginea
        img_input = load_img(cale_img, target_size=dim_img)

        # transformam in array si normalizam
        img_array = img_to_array(img_input) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # adaugam batch dimension

        # facem predictia
        pred_cad = model.predict(img_array)[0]
        idx_pred = np.argmax(pred_cad)
        conf = pred_cad[idx_pred]

        # construim rezultatul principal
        rezultat = {
            "Predicție": descrieri[idx_pred],
            "Confidență": f"{conf*100:.2f}%"
        }

        # adaugam procent pe fiecare categorie
        for clasa, p in zip(clase, pred_cad):
            rezultat[clasa] = f"{p*100:.2f}%"

        print(rezultat)
        return rezultat

    except Exception as e:
        print(f"EROARE: ceva nu a mers la clasificare -> {e}")
        return

# =========================================
# Functia main
# =========================================
def main(cale_img):
    # verificam daca modelul exista
    if not os.path.exists(cale_model_cnn):
        print(f"EROARE: modelul nu exista -> {cale_model_cnn}")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(cale_model_cnn)
    except Exception as e:
        print(f"EROARE la incarcarea modelului -> {e}")
        sys.exit(1)

    # aici clasifica piesa
    return clasifica_piesa(model, cale_img)


# =========================================
# Executie directa
# =========================================
if __name__ == "__main__":
    # exemplu de test - poti schimba cu alt fisier
    cale_test = "imagini_noi/piesa_test.png"

    # TODO: poate adaug un loop pentru mai multe imagini
    main(cale_test)

    # note pentru mine:
    # - verific sa nu fie crash la fisiere lipsa
    # - poate mai tarziu fac output intr-un CSV
