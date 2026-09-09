"""

"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from import_dataset import open_hopv15_dataset

def encode_mat(input_mat):
    unique_vals = np.unique(input_mat)
    print(unique_vals)
    for index, key in enumerate(unique_vals):
        input_mat[input_mat == key] = index
    return input_mat

if __name__ == "__main__":
    device = "cuda"
    RANDOM_SEED = 42
    data = open_hopv15_dataset("D:\\kan_encoder_architecture\\HOPV_15_revised_2.data")
    usable_labels = []
    molecule_counter = 1
    for molecule in data:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.PCE) or molecule_counter in [96, 105, 159, 284, 307, 350]:
            molecule_counter += 1
            continue
        for conformer in molecule.conformer_info:
            for calc in conformer.calculated_data:
                usable_labels.append([molecule.experimental_data.PCE, molecule.experimental_data.complement])
        molecule_counter += 1
    usable_data = pd.read_csv("D:\\kan_encoder_architecture\\saved_harvard_theory.csv").to_numpy()
    usable_labels = np.asarray(usable_labels)
    usable_labels[:, 1] = encode_mat(usable_labels[:, 1]).astype(np.int16)
    usable_labels = usable_labels.astype(np.float64)
    test_size = 0.2
    train_x, test_x, train_y, test_y = train_test_split(usable_data, usable_labels[:, 1], test_size=test_size, stratify=usable_labels[:, 1], shuffle=True, random_state=42)
    # Random Forest
    random_forest = RandomForestClassifier(random_state=42)
    random_forest.fit(train_x, train_y.astype(np.long))
    print(test_y)
    print(random_forest.predict(test_x))
    print(classification_report(test_y.astype(np.long), random_forest.predict(test_x)))
    print(log_loss(test_y.astype(np.long), random_forest.predict_proba(test_x)))
    # SVM
    svc = SVC(random_state=42, probability=True)
    svc.fit(train_x, train_y)
    print(classification_report(test_y.astype(np.long), svc.predict(test_x)))
    print(log_loss(test_y.astype(np.long), svc.predict_proba(test_x)))
    # Gradient Boosted Tree
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(train_x, train_y)
    print(classification_report(test_y.astype(np.long), lgbm.predict(test_x)))
    print(log_loss(test_y.astype(np.long), lgbm.predict_proba(test_x)))