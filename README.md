
📘 ReteleNeuronaleProiect – Documentație Etapa 4
Disciplina: Rețele Neuronale 
Instituție: POLITEHNICA București – FIIR 
Student: Savu Andrei Catalin 
Data: 27/11/2025

1. 📂 Structura Repository-ului Github (Versiunea Etapa 4)
Structura fișierelor respectă standardele MLOps, asigurând separarea clară între datele brute, codul sursă și seturile de date preprocesate.

project-name/
├── README.md
├── docs/
│   └── datasets/              # Documentația setului de date
│       ├── caracteristici_dataset
│       ├── desciere_set_date
│       └── descriere_caracteristici
├── data/
│   ├── raw/                   # Date brute (asag_simulated_train_data.csv)
│   ├── processed/             # Date curățate (processed.csv)
│   ├── train/                 # Set de instruire (train.csv)
│   ├── validation/            # Set de validare (validation.csv)
│   └── test/                  # Set de testare (test.csv)
├── src/
│   ├── preprocessing/         # Funcții pentru curățare și splitare (process_data.py)
│   ├── data_acquisition/      # Script generare date simulate
│   └── neural_network/        # Implementarea RN (viitoare etapă)
├── config/                    # Fișiere de configurare
└── requirements.txt           # Dependențe Python (ex: pandas, scikit-learn, transformers)



2. 📊 Descrierea Setului de Date
2.1 Sursa datelor
Origine: Dataset simulat pentru ASAG (Automatic Short Answer Grading), bazat pe concepte tehnice de Rețele Neuronale și NLP.

Modul de achiziție: Generare programatică (Script Python cu variații controlate ale răspunsurilor).

Volum inițial: 1.500 de observații (30 de întrebări distincte, fiecare cu 50 de răspunsuri simulate de studenți).

2.2 Caracteristicile dataset-ului
Număr total de observații: 1.500

Număr de caracteristici (features): 6

Tipuri de date: Numerice și Categoriale (Textuale).

Format fișiere: CSV.

Caracteristica,Tip,Unitate,Descriere,Domeniu valori
question_id,categorial,–,Identificatorul unic al întrebării,Q01 – Q30
question_text,text,–,Enunțul întrebării de examen,String
answer_correct,text,–,Răspunsul de referință (barem),String (lungime variabilă)
score_range,numeric,puncte,Punctajul maxim al întrebării,1.0 – 5.0
answer_student,text,–,Răspunsul simulat al studentului,String
score_manual,numeric,puncte,Nota acordată (Target Label),0.0 – 5.0

3. 🔎 Analiza Exploratorie a Datelor (EDA) – Sintetic
3.1 Statistici descriptive aplicate
Distribuția scorurilor: S-a confirmat o acoperire a întregului spectru de note (score_manual), necesară pentru a antrena modelul să recunoască răspunsuri corecte, parțiale și greșite.

Analiza Lungimii Textului: S-a folosit pentru a identifica și a preveni problemele cauzate de răspunsurile excesiv de scurte (sub 2 cuvinte) sau nejustificat de lungi.

Clase Distincte: Setul de date conține exact 30 de clase de clasificare distincte (întrebările unice).

3.2 Analiza calității datelor
Detectarea valorilor lipsă: 0% valori lipsă (dataset generat controlat).

Consistență: Verificare automată pentru a asigura că nota acordată (score_manual) nu depășește niciodată punctajul maxim (score_range).

3.3 Probleme identificate
Variatii Textuale: Inconsistențe în utilizarea majusculelor și semnelor de punctuație, care ar putea perturba analiza semantică.

Formatare: Existența spațiilor multiple și a caracterelor speciale neesențiale.

4. 🧹 Preprocesarea Datelor (NLP)
4.1 Curățarea datelor
Eliminare valori nule: Proces automat pentru a garanta integritatea setului de date.

Curățare Text (NLP):

Conversie la litere mici (lowercasing): Asigură că modelul tratează "BERT" și "bert" ca fiind același termen.

Eliminarea semnelor de punctuație și a spațiilor albe suplimentare: Curățarea zgomotului care nu aduce valoare semantică.

4.2 Transformarea caracteristicilor
Normalizare text: O funcție uniformă (clean_text) a fost aplicată pe toate coloanele de tip text (question_text, answer_correct, answer_student).

Pregătire Vectorizare: Setul de date este acum curat și pregătit pentru conversia în embedding-uri (vectori numerici) în etapa de implementare a Rețelei Neuronale (probabil utilizând un model pre-antrenat de tip Transformer).

4.3 Structurarea seturilor de date
Divizarea a fost realizată pe baza ID-ului întrebării pentru a garanta că modelul este testat pe concepte la care nu a fost expus în timpul antrenării (evitarea Data Leakage).

Set de Date,Întrebări,Număr Înregistrări,Scop
Train (Antrenare),Q01 – Q24,1200,Antrenarea modelului.
Validation (Validare),Q25 – Q27,150,Verificarea generalizării și reglarea hiperparametrilor.
Test (Testare),Q28 – Q30,150,Evaluarea finală a performanței.

4.4 Salvarea rezultatelor preprocesării
Datele preprocesate și împărțite au fost salvate în data/train/, data/validation/, și data/test/.

5. 💾 Fișiere Generate în Această Etapă
data/raw/asag_simulated_train_data.csv – datasetul complet.

data/processed/processed.csv – setul de date curățat și verificat.

data/train/train.csv – setul de antrenament.

data/validation/validation.csv – setul de validare.

data/test/test.csv – setul de testare.

src/preprocessing/process_data.py – codul Python utilizat pentru curățare și splitare.

6. ✅ Stare Etapă
[x] Structura repository configurată

[x] Dataset analizat și generat (1500 intrări)

[x] Date preprocesate (NLP cleaning)

[x] Seturi train/val/test generate

[x] Documentație actualizată în README

📘 README – P3: Proiect SAF - Diagram State Machines
Disciplina: Sisteme Avansate de Fabricare Instituție: POLITEHNICA București – FIIR Student: [Nume Prenume] Data: [Data]
Scopul Etapei P3
Această etapă corespunde punctului 3. Dezvoltare proiect software - slide 10 SAF - Specificatii proiect.pdf.

Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem Ciber-Fizic.

Livrabile Obligatorii

Nevoie reală concretă,Cum o rezolvă SIA-ul vostru,Modul software responsabil
"Identificarea tipului de componentă dintr-o imagine (ex: flanșă, șurub, piuliță)",Clasificare imagine 2D a modelului CAD → decizie în <0.5 secunde,Data Logging + SM + Web Service
Catalogarea automată a piesei identificate în baza de date,Etichetare după tip (clasă) → acuratețe de clasificare ≥99%,Data Logging + SM + UI
Verificarea unicității imaginii CAD înainte de clasificare,Comparare hash imagine → rata de eroare a imaginilor duplicate redusă cu 80%,Data Logging + SM + Control Module

Justificarea State Machine-ului ales:Am ales arhitectura de Clasificare Vizuală la Cerere (similar cu Exemplul B) pentru că scopul proiectului este strict identificarea și catalogarea automată a modelelor CAD pe baza unei imagini furnizate. Am eliminat toate ramurile de decizie legate de controlul calității sau detectarea defectelor, rezultând un flux optimizat pentru Data Logging și Catalogare cu latență minimă.Stările principale sunt:INIT (Initialize System): Asigură conexiunea la baza de date locală (cerință obligatorie pentru Data Logging).INFERENCE: Execută clasificarea propriu-zisă, atribuind piesa categoriei corecte (ex: Șurub).LOG_DB (Catalogare): Înregistrează rezultatul clasificării în baza de date, îndeplinind cerința de catalogare.Tranzițiile critice sunt:INIT $\rightarrow$ SHUTDOWN: Se declanșează dacă nu se poate stabili conexiunea la baza de date, deoarece logarea și catalogarea sunt esențiale.INFERENCE $\rightarrow$ ERROR: Se declanșează când modelul nu poate clasifica imaginea (confidență sub prag), indicând o eroare de sistem (date input slabe), nu un defect al piesei.Bucla de feedback este simplă: rezultatul pozitiv trece direct în LOG_DB, iar apoi sistemul revine în starea de așteptare (IDLE) pentru a prelua următoarea imagine.Checklist Final – Bifați Totul Înainte de PredareDocumentație și Structură[x] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README)[ ] Diagrama State Machine creată și salvată și postată alături de acest readme pe moodle la P3. State Machine pentru proiectul SAF[x] Legendă State Machine scrisă în acest readme (minimum 1-2 paragrafe cu justificare)
