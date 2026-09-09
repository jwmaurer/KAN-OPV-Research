"""
Jacob Maurer
9/4/2026
Purpose: Functions that are universal across all ablation files
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from test_file import encode_mat
import torch
import torch.nn as nn
from torch.utils.data import Dataset

DEVICE = "cuda"

def remove_nan_vals(dataset):
    usable_data = []
    usable_labels = []
    molecule_counter = 1
    saved_molecule_count = 0
    for molecule in dataset:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.PCE) or molecule_counter in [96, 105, 159, 284, 307, 350]:
            molecule_counter += 1
            continue
        for conformer in molecule.conformer_info:
            for calc in conformer.calculated_data:
                usable_data.append([calc.HOMO, calc.LUMO, calc.Gap])
                usable_labels.append([molecule.experimental_data.PCE, molecule.experimental_data.complement])
        saved_molecule_count += 1
        molecule_counter += 1
    print(saved_molecule_count)
    usable_data = pd.read_csv("./saved_harvard_theory.csv").to_numpy()
    usable_labels = np.array(usable_labels)
    return usable_data, usable_labels

def make_class_dataset(dataset, seed):
    data, labels = remove_nan_vals(dataset)
    labels[:, 1] = encode_mat(labels[:, 1]).astype(np.int16)
    labels = labels.astype(np.float64)
    test_size = 0.2
    train_set_x, test_x, train_set_y, test_y = train_test_split(data, labels[:, 1], test_size=test_size, stratify=labels[:, 1], shuffle=True, random_state=seed)
    train_x, valid_x, train_y, valid_y = train_test_split(train_set_x, train_set_y, test_size=0.1, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    pca = PCA(n_components=.9, random_state=seed)
    train_x = pca.fit_transform(scaler.fit_transform(train_x))
    valid_x = pca.transform(scaler.transform(valid_x))
    test_x = pca.transform(scaler.transform(test_x))
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def make_regress_dataset(dataset, seed):
    data, labels = remove_nan_vals(dataset)
    print(labels[:, 0])
    labels = labels[:, 0].astype(np.float64)
    test_size = 0.2
    train_set_x, test_x, train_set_y, test_y = train_test_split(data, labels, test_size=test_size, shuffle=True, random_state=seed)
    train_x, valid_x, train_y, valid_y = train_test_split(train_set_x, train_set_y, test_size=0.1, shuffle=True, random_state=seed)
    scaler = MinMaxScaler()
    pca = PCA(n_components=.9, random_state=seed)
    train_x = pca.fit_transform(scaler.fit_transform(train_x))
    valid_x = pca.transform(scaler.transform(valid_x))
    test_x = pca.transform(scaler.transform(test_x))
    return train_x, train_y, valid_x, valid_y, test_x, test_y

class ClassDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return torch.from_numpy(self.data[index]).type(torch.float).to(DEVICE), torch.from_numpy(self.labels[index, np.newaxis]).type(torch.long).to(DEVICE)

class RegressDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return torch.from_numpy(self.data[index]).type(torch.float).to(DEVICE), torch.from_numpy(self.labels[index, np.newaxis]).type(torch.float).to(DEVICE)
