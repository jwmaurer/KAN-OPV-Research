"""
Jacob Maurer
6/17/2026
Purpose: Determine if the scharber model can be predicted with KAN (a control, if you will)
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import torch
import torch.nn as nn
from kan import *
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from import_dataset import open_hopv15_dataset

device = "cuda"

class ScharberDataset(Dataset):
    def __init__(self, input_vars, pce_vars):
        self.scharber_vars = input_vars
        self.pces = pce_vars
    def __len__(self):
        return len(self.scharber_vars)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return torch.from_numpy(self.scharber_vars[index]).type(torch.float).to(device), torch.from_numpy(self.pces[index]).type(torch.float).to(device)

if __name__ == "__main__":
    RSEED = 42
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    scharber_vars = []
    scharber_pces = []
    for molecule in data:
        for conformer in molecule.conformer_info:
            for calc in conformer.calculated_data:
                if not np.isnan(calc.scharber_jsc) and not np.isnan(calc.scharber_voc) and not np.isnan(calc.scharber_pce):
                    scharber_vars.append([calc.scharber_jsc, calc.scharber_voc])
                    scharber_pces.append(calc.scharber_pce)
    test_split = 0.2
    dataset = create_dataset_from_data(torch.from_numpy(np.asarray(scharber_vars)).type(torch.float), torch.from_numpy(np.asarray(scharber_pces)).type(torch.float), device=device)
    kan = KAN([2, 10, 1], grid=4, k=5, seed=42, device=device)
    kan.fit(dataset, steps=50)
    print(f"Mean Squared Error: {mean_squared_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    print(f"Mean Absolute Error: {mean_absolute_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    print(f"Root Mean Squared Error {root_mean_squared_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    kan.prune()
    kan(dataset['test_input'])
    kan.auto_symbolic()
    print(kan.symbolic_formula()[0][0])
    