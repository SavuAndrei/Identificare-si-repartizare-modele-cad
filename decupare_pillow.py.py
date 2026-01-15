import cv2
import os
import numpy as np

def decupeaza_contururi_multiple(cale_intrare, cale_iesire):
    """
    Detectează și decupează fiecare contur mare (piesă) individual.
    """
    if not os.path.exists(cale_iesire):
        os.makedirs(cale_iesire)

    for nume_fisier in os.listdir(cale_intrare):
        if nume_fisier.lower().endswith(('.png', '.jpg', '.jpeg')):
            cale_completa_intrare = os.path.join(cale_intrare, nume_fisier)
            
            try:
                img = cv2.imread(cale_completa_intrare)
                if img is None:
                    print(f"Eroare: Nu s-a putut citi {nume_fisier}")
                    continue

                gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Binarizare: Izolează zonele negre (desenul)
                _, thresh = cv2.threshold(gri, 220, 255, cv2.THRESH_BINARY_INV)

                # Găsirea contururilor
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    print(f"Avertisment: Nu s-au găsit contururi în {nume_fisier}.")
                    continue

                numar_piesa = 1
                # Filtrare contururi: Luăm doar contururile cu o arie suficient de mare
                # (Ajustează 5000 în funcție de mărimea pozelor tale)
                contururi_filtrate = [c for c in contours if cv2.contourArea(c) > 5000]

                if not contururi_filtrate:
                    print(f"Avertisment: Niciun contur suficient de mare găsit în {nume_fisier}.")
                    continue
                
                # Decuparea și salvarea fiecărui contur mare
                for contur in contururi_filtrate:
                    x, y, w, h = cv2.boundingRect(contur)
                    
                    # Adăugare padding
                    padding = 15
                    y_start = max(0, y - padding)
                    y_end = min(img.shape[0], y + h + padding)
                    x_start = max(0, x - padding)
                    x_end = min(img.shape[1], x + w + padding)

                    imagine_decupata = img[y_start:y_end, x_start:x_end]
                    
                    nume_iesire = f"decupat_{os.path.splitext(nume_fisier)[0]}_piesa{numar_piesa}.png"
                    cale_completa_iesire = os.path.join(cale_iesire, nume_iesire)
                    cv2.imwrite(cale_completa_iesire, imagine_decupata)
                    
                    print(f"Salvată piesa {numar_piesa} din {nume_fisier}")
                    numar_piesa += 1

            except Exception as e:
                print(f"Eroare la prelucrarea fișierului {nume_fisier}: {e}")

# --- Configurație ---
# !!! Modifică aceste căi folosind prefixul 'r' !!!
FOLDER_INTRARE = r'C:\Users\Flavius\Desktop\proiect Rn\poze_rn'
FOLDER_IESIRE = r'C:\Users\Flavius\Desktop\proiect Rn\poze_decupate_multiple'
# ---------------------

decupeaza_contururi_multiple(FOLDER_INTRARE, FOLDER_IESIRE)