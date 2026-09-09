"""
Jacob Maurer
8/16/2026
Prupose: Try different basis sets on our datasets to determine how well b-splines compare to other methods
"""
from typing import Any
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
from cheby import ChebyKANLayer
from rbf import RBFKANLayer
from wavlet import KAN
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
import pandas as pd

DEVICE = "cpu"
RANDOM_SEED = 42

class ChebyKAN(nn.Module):
    def __init__(self, layers: list[tuple[int, int]]):
        super().__init__()
        self.kan_layers = nn.Sequential(*[ChebyKANLayer(in_dim[0], out_dim[0], in_dim[1]) for in_dim, out_dim in zip(layers[:-1], layers[1:])])
    def forward(self, x):
        return self.kan_layers(x)

class RBFKAN(nn.Module):
    def __init__(self, layers: list[tuple[int, int, float]]):
        super().__init__()
        self.kan_layers = nn.Sequential(*[RBFKANLayer(in_dim[0], out_dim[0], in_dim[1], in_dim[2]) for in_dim, out_dim in zip(layers[:-1], layers[1:])])
    def forward(self, x):
        return self.kan_layers(x)
    
class PCEData(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y
    def __len__(self):
        return len(data)
    def __getitem__(self, index) -> Any:
        if torch.is_tensor(index):
            index = index.tolist()
        return (torch.from_numpy(self.data[index]).type(torch.float).to(DEVICE),
                torch.from_numpy(self.labels[index]).type(torch.float).to(DEVICE))
    
class ComplementData(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y
    def __len__(self):
        return len(data)
    def __getitem__(self, index) -> Any:
        if torch.is_tensor(index):
            index = index.tolist()
        return (torch.from_numpy(self.data[index]).type(torch.float).to(DEVICE),
                torch.from_numpy(self.labels[index]).type(torch.long).to(DEVICE))
   
def encode_mat(input_mat):
    unique_vals = np.unique(input_mat)
    print(unique_vals)
    for index, key in enumerate(unique_vals):
        input_mat[input_mat == key] = index
    return input_mat     

@dataclass
class ExperimentalInformation:
    DOI: str
    InChi_key: str
    construction: str
    architecture: str
    complement: str
    HOMO: float | str
    LUMO: float | str
    electorchemical_gap: float | str
    optical_gap: float | str
    PCE: float | str
    Voc: float | str
    Jsc: float | str
    fill_factor: float | str
    def has_nan(self) -> bool:
        check = (isinstance(self.HOMO, str) or isinstance(self.LUMO, str) or isinstance(self.electorchemical_gap, str)
        or isinstance(self.optical_gap, str) or isinstance(self.PCE, str) or isinstance(self.Voc, str) or isinstance(self.Jsc, str)
        or isinstance(self.fill_factor, str))
        if check:
            print(f"Nan value found in Experimental information of {self.DOI}")
        return check
    
@dataclass
class CalculatedInformation:
    basis_set_description: str
    HOMO: float | str
    LUMO: float | str
    Gap: float | str
    scharber_pce: float | str
    scharber_voc: float | str
    scharber_jsc: float | str
    def has_nan(self) -> bool:
        check = (isinstance(self.HOMO, str) or isinstance(self.LUMO, str) or isinstance(self.Gap, str) or 
                 isinstance(self.scharber_pce, str) or isinstance(self.scharber_voc, str) or isinstance(self.scharber_jsc, str))
        if check:
            print(f"Nan value found in the Calculated information with {self.basis_set_description}")
        return check

@dataclass
class ChemicalCoordinates:
    chemical_id: str
    x: float
    y: float
    z: float

@dataclass
class ConformerInformation:
    conformer_name: str
    number_atoms: int
    atom_coordinates: list[ChemicalCoordinates]
    calculated_data: list[CalculatedInformation]

@dataclass
class FormattedData:
    smiles_molecule: str
    InChl_molecule: str
    experimental_data: ExperimentalInformation
    pruned_smiles: str
    total_conformers: int
    conformer_info: list[ConformerInformation]
    
def convert_poss_nan(item: str)-> float | str:
    try:
        convert = float(item)
    except:
        convert = item
    return convert

def open_hopv15_dataset(fpath):
    imported_data: list[FormattedData] = []
    num_sections = 350
    with open(fpath, 'rt') as file:
        for i in range(num_sections):
            smiles = file.readline().strip()
            inchl = file.readline().strip()
            experimental_list = file.readline().strip().split(',')
            DOI_exp = experimental_list[0]
            inchl_exp = experimental_list[1]
            construct = experimental_list[2]
            architect = experimental_list[3]
            complement = experimental_list[4]
            homo = convert_poss_nan(experimental_list[5])
            lumo = convert_poss_nan(experimental_list[6])
            electrochem = convert_poss_nan(experimental_list[7])
            optical = convert_poss_nan(experimental_list[8])
            pce = convert_poss_nan(experimental_list[9])
            voc = convert_poss_nan(experimental_list[10])
            jsc = convert_poss_nan(experimental_list[11])
            fill_wid = convert_poss_nan(experimental_list[12])
            experiment = ExperimentalInformation(DOI_exp, inchl_exp, construct, architect, complement, homo, lumo, electrochem, optical, pce, voc, jsc, fill_wid)
            pruned_smiles = file.readline().strip()
            conformers_num = int(file.readline().strip())
            conformers = []
            for conformer in range(conformers_num):
                file.readline()
                num_atoms = int(file.readline().strip())
                chem_list = []
                for atom in range(num_atoms):
                    cur_line = file.readline().strip().split(' ')
                    chem_list.append(ChemicalCoordinates(cur_line[0], float(cur_line[1]), float(cur_line[2]), float(cur_line[3])))
                calc_data = []
                for calc in range(4):
                    new_line = file.readline().strip().split(',')
                    homo_calc = convert_poss_nan(new_line[1])
                    lumo_calc = convert_poss_nan(new_line[2])
                    gap_calc = convert_poss_nan(new_line[3])
                    pce_calc = convert_poss_nan(new_line[4])
                    voc_calc = convert_poss_nan(new_line[5])
                    jsc_calc = convert_poss_nan(new_line[6])
                    calc_data.append(CalculatedInformation(new_line[0], homo_calc, lumo_calc, gap_calc, pce_calc, voc_calc, jsc_calc))
                conformers.append(ConformerInformation(f'Conformer {conformer}', num_atoms, chem_list, calc_data))
            imported_data.append(FormattedData(smiles, inchl, experiment, pruned_smiles, conformers_num, conformers))
        return imported_data

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.squeeze())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        """
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        """

def test(dataloader, model, loss_fn, device):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.squeeze()).item()
            # pred = (pred > 0.5).type(torch.float)
            # correct += (pred == y.unsqueeze(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    scale = True
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    usable_labels = []
    usable_data = []
    molecule_counter = 1
    saved_molecule_count = 0
    for molecule in data:
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
    usable_labels[:, 1] = encode_mat(usable_labels[:, 1]).astype(np.int16)
    usable_labels = usable_labels.astype(np.float64)
    test_size = 0.2
    train_set_x, test_x, train_set_y, test_y = train_test_split(usable_data, usable_labels[:, 1], test_size=test_size, stratify=usable_labels[:, 1], shuffle=True, random_state=RANDOM_SEED)
    train_x, valid_x, train_y, valid_y = train_test_split(train_set_x, train_set_y, test_size=0.1, shuffle=True, random_state=RANDOM_SEED)
    if scale:
        scaler = StandardScaler()
        pca = PCA(n_components=0.90, random_state=RANDOM_SEED)
        train_x = pca.fit_transform(scaler.fit_transform(train_x))
        valid_x = pca.transform(scaler.transform(valid_x))
        test_x = pca.transform(scaler.transform(test_x))
    train_set = DataLoader(ComplementData(train_x, np.expand_dims(train_y, axis=1)), batch_size=100, shuffle=True)
    valid_set = DataLoader(ComplementData(valid_x, np.expand_dims(valid_y, axis=1)), batch_size=100, shuffle=True)
    test_set = DataLoader(ComplementData(test_x, np.expand_dims(test_y, axis=1)), batch_size=100, shuffle=True) 
    # model = KAN([len(train_x[0]), 50, 50, 4], wavelet_type='morlet').to(DEVICE)
    # model = RBFKAN([(len(train_x[0]), 18, 1.0), (50, 18, 0.5), (50, 18, 1.0), (4, 8, 1.0)]).to(DEVICE)
    model = ChebyKAN([(len(train_x[0]), 5), (20, 5), (20, 5), (20, 5), (4, 3)]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 30
    for _ in range(epochs):
        train(train_set, model, loss_fn, optimizer)
        test(valid_set, model, loss_fn)
    full_guess = []
    full_true_val = []
    for X, y in test_set:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        full_guess.extend(pred.cpu().detach().numpy())
        full_true_val.extend(y.cpu().detach().numpy())
    print("------------------------------------------")
    print(f"| Comp Predictions: {loss_fn(torch.from_numpy(np.asarray(full_guess)).to(DEVICE), torch.from_numpy(np.asarray(full_true_val)).squeeze().type(torch.long).to(DEVICE))}")
    print("------------------------------------------")
    
    