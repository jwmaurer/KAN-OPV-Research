from kan import *
from sklearn.discriminant_analysis import StandardScaler
# from import_dataset import FormattedData, open_hopv15_dataset, mol_atom_dist, open_csv_dataset
import numpy as np
import copy
import math
# from rdkit import Chem
import torch

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

def get_csv_db(fpath):
    imported_data = open_csv_dataset(fpath).select("smile").to_numpy()
    x_list = []
    index_convert = {"C": 0, "S": 1, "O": 2, "H": 3, "N": 4, "Si": 5, "F": 6, "Se": 7, "P": 8}
    for item in imported_data:
        atom_data = mol_atom_dist(Chem.AddHs(Chem.MolFromSmiles(item[0])))
        temp_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for key in atom_data:
            temp_list[index_convert[key]] = float(atom_data[key])
        x_list.append(copy.deepcopy(temp_list))
    return np.array(x_list)

if __name__ == "__main__":
    # data = get_hopv15("./HOPV_15_revised_2.data")
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(f, n_var=2, device='cuda')
    model = KAN(width=[2,1,1], grid=5, k=3, seed=0, noise_scale=0.0, device='cuda')
    model.fit(dataset, opt="LBFGS", steps=20, lamb=0.001)
    model.prune()
    model(dataset["test_input"])
    model.auto_symbolic()
    print(model.symbolic_formula()[0][0])