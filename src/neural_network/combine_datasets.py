import pandas as pd
import os

RAW_DATA_PATH = '../../data/raw'
GENERATED_DATA_PATH = '../../data/generated'
PROCESSED_DATA_PATH = '../../data/processed'

# Presupunem ca datele vechi sunt un fisier CSV principal
OLD_DATA_FILE = 'dataset_principal_etapa3.csv'
# Presupunem ca datele noi generate sunt intr-un alt fisier CSV
NEW_DATA_FILE = 'dataset_nou_etapa4.csv'
# Fisierul final combinat
COMBINED_DATA_FILE = 'combined_full_dataset.csv'


def combine_datasets():
    
    # Citeste datele vechi
    old_data_path = os.path.join(RAW_DATA_PATH, OLD_DATA_FILE)
    if os.path.exists(old_data_path):
        df_old = pd.read_csv(old_data_path)
    else:
        df_old = pd.DataFrame()

    # Citeste datele noi
    new_data_path = os.path.join(GENERATED_DATA_PATH, NEW_DATA_FILE)
    if os.path.exists(new_data_path):
        df_new = pd.read_csv(new_data_path)
    else:
        df_new = pd.DataFrame()

    if df_old.empty and df_new.empty:
        return

    # Combina cele doua DataFrame-uri
    df_combined = pd.concat([df_old, df_new], ignore_index=True)

    # Salveaza dataset-ul combinat in folderul processed (sau raw, depinde de fluxul dorit)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    combined_path = os.path.join(PROCESSED_DATA_PATH, COMBINED_DATA_FILE)
    df_combined.to_csv(combined_path, index=False)

if __name__ == '__main__':
    combine_datasets()
