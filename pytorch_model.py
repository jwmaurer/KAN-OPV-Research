from math import nan
from typing import Any
from sympy import NDimArray
import torch
import torch.nn as nn
from import_dataset import FormattedData, open_hopv15_dataset, mol_atom_dist, open_csv_dataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from rdkit import Chem
import copy
import numpy as np
import math
device = 'cuda'

def get_hopv15(path):
    # Data [C_count, S count, O count, H count, N count, Si count, F count, Se count, experimental PCE]
    imported_data: list[FormattedData] = open_hopv15_dataset(path)
    x_list = []
    y_list = []
    index_convert = {"C": 0, "S": 1, "O": 2, "H": 3, "N": 4, "Si": 5, "F": 6, "Se": 7, "P": 8}
    for item in imported_data:
        #Check for nan in the Experimental PCE. Want to continue and skip that
        if math.isnan(item.experimental_data.PCE):
            continue
        atom_data = mol_atom_dist(Chem.AddHs(Chem.MolFromSmiles(item.smiles_molecule)))
        temp_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for key in atom_data:
            temp_list[index_convert[key]] = float(atom_data[key])
        y_list.append(item.experimental_data.PCE)
        x_list.append(copy.deepcopy(temp_list))
    return np.array(x_list), np.array(y_list)

def make_validation(path):
    # Data [C_count, S count, O count, H count, N count, Si count, F count, Se count, P count, experimental PCE]
    imported_data: list[FormattedData] = open_hopv15_dataset(path)
    x_list = []
    index_convert = {"C": 0, "S": 1, "O": 2, "H": 3, "N": 4, "Si": 5, "F": 6, "Se": 7}
    for item in imported_data:
        #Check for nan in the Experimental PCE. Want to continue and skip that
        if not math.isnan(item.experimental_data.PCE):
            continue
        atom_data = mol_atom_dist(Chem.AddHs(Chem.MolFromSmiles(item.smiles_molecule)))
        temp_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for key in atom_data:
            temp_list[index_convert[key]] = float(atom_data[key])
        x_list.append(copy.deepcopy(temp_list))
    return np.array(x_list)

def make_validation_2(fpath):
    imported_data = open_csv_dataset(fpath).select("smile").to_numpy()
    x_list = []
    index_convert = {"C": 0, "S": 1, "O": 2, "H": 3, "N": 4, "Si": 5, "F": 6, "Se": 7, "P": 8}
    for item in imported_data:
        #Check for nan in the Experimental PCE. Want to continue and skip that
        atom_data = mol_atom_dist(Chem.AddHs(Chem.MolFromSmiles(item[0])))
        temp_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for key in atom_data:
            temp_list[index_convert[key]] = float(atom_data[key])
        x_list.append(copy.deepcopy(temp_list))
    return np.array(x_list)

class HOPV15Dataset(Dataset):
    def __init__(self, x, y) -> None:
        self.data_set: tuple[np.ndarray, np.ndarray] = (x, y)
        
    def __len__(self) -> int:
        return len(self.data_set[0])
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = index.tolist()
        return self.data_set[0][index], np.float32(self.data_set[1][index])

class HOPV15Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x.type(torch.float))
        return logits

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        """
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        """

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            # pred = (pred > 0.5).type(torch.float)
            # correct += (pred == y.unsqueeze(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    gathered_set = get_hopv15("./HOPV_15_revised_2.data")
    validation_set = make_validation_2("OPV_Datasets/opv_db.csv")
    hopv_dataset_train = HOPV15Dataset(x=gathered_set[0][int(len(gathered_set[0])*.2):],y=gathered_set[1][int(len(gathered_set[1])*.2):])
    hopv_dataset_test = HOPV15Dataset(x=gathered_set[0][:int(len(gathered_set[0])*.2)],y=gathered_set[1][:int(len(gathered_set[1])*.2)])
    hopv_train_loader = DataLoader(hopv_dataset_train, batch_size=50, shuffle=True)
    hopv_test_loader = DataLoader(hopv_dataset_test, batch_size=50, shuffle=True)
    print("Cleaning and pre-processing of data completed...")
    hopv_model = HOPV15Network().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(hopv_model.parameters(), lr=5e-2)
    epochs = 10
    for i in range(epochs):
        train(hopv_train_loader, hopv_model, loss_fn=loss_fn, optimizer=optimizer)
        test(hopv_test_loader, hopv_model, loss_fn=loss_fn)
    hopv_model.eval()
    with torch.no_grad():
        for item in validation_set:
            input_item = torch.from_numpy(item).type(torch.float).to(device)
            pred = hopv_model(input_item)
            print(f"The molecule distribution {item} has a predicted experimental PCE: {pred}")
    # large_model_post = copy.deepcopy(hopv_model.cpu().state_dict())    
    # ID = 1
    # for item in large_model_post:
    #     if item[20:] == "weight":
    #         table = pd.DataFrame(large_model_post[item])
    #         table.to_csv("./hopv_model_weights_" + str(ID) + ".csv",index=False)
    #     if item[20:] == "bias":
    #         series = pd.Series(large_model_post[item])
    #         series.to_csv("./hopv_model_bias_" + str(ID) + ".csv",index=False)
    #     ID += 1
