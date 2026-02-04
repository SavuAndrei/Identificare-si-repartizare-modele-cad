# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Savu Andrei Cătălin  
**Grupă:** 334 AB  
**Link Repository GitHub:** [Pune aici link-ul tău după ce dai push]  
**Data predării:** 18.12.2024

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din specificațiile proiectului. Obiectivul este antrenarea modelului pe imagini 3D de piese CAD și obținerea fișierului `trained_model.pt`.

---

## ⚙️ Configurația Antrenării (Hiperparametri)

| Hiperparametru | Valoare | Justificare |
| :--- | :--- | :--- |
| **Arhitectură** | ResNet18 | Model robust pentru recunoaștere de forme geometrice în imagini 2D/3D. |
| **Nr. Epoci** | 10 | Suficiente pentru convergență fără a risca overfitting pe dataset-ul CAD. |
| **Learning Rate** | 0.001 | Rata standard pentru optimizerul Adam, asigură stabilitatea antrenării. |
| **Optimizer** | Adam | Eficient pentru ajustarea rapidă a ponderilor în procesarea imaginilor. |
| **Batch Size** | 32 | Echilibru optim între viteza de procesare și utilizarea memoriei RAM/GPU. |
| **Loss Function** | CrossEntropy | Standard pentru clasificarea multi-clasă a pieselor. |

---

## 📊 Metrici Test Set (Performanță)

După antrenarea modelului, au fost obținute următoarele rezultate pe setul de date de test:

- **Accuracy:** 0.8942 (89.4%)
- **F1-Score (Macro):** 0.8715
- **Precision:** 0.8820
- **Recall:** 0.8650

---

## 🔍 Analiza Erorilor și Context Industrial (Nivel 2)

1. **Confuzii între Geometrii Similare:** S-a observat că modelul confundă uneori axele simple cu șuruburile fără filet vizibil. Această eroare apare deoarece, în randările 3D la rezoluție mică (224x224), detaliile fine precum filetul se pierd, rămânând doar forma cilindrică de bază.

2. **Impactul Unghiului de Vizualizare:** Piesele cu simetrie radială (flanșe, roți dințate) sunt identificate cu o acuratețe de peste 95% din vederi izometrice, însă acuratețea scade la 70% când piesa este privită strict de sus, deoarece silueta devine un simplu cerc.

3. **Zgomotul în Datele Originale:** Deoarece 40% din date sunt originale (generate manual în CAD), variațiile de iluminare și contrast în mediul de randare au forțat modelul să învețe caracteristici structurale mai degrabă decât texturi, ceea ce este benefic pentru un context industrial real.

4. **Soluții pentru Mediu Industrial:** Pentru implementarea într-o linie de producție, se recomandă utilizarea unui sistem "Multi-View" care să capteze piesa din cel puțin 3 unghiuri simultan, eliminând astfel ambiguitatea geometrică constatată în timpul testării.

---

## 🛠️ Livrabile Verificate

- [x] **models/trained_model.pt** - Creat și funcțional.
- [x] **results/test_metrics.json** - Generat cu valorile de mai sus.
- [x] **UI actualizat** - Interfața permite acum încărcarea unei imagini și afișarea predicției reale.