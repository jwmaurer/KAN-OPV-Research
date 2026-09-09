"""
Jacob Maurer
8/16/2026
Purpose: Curiosity. Want to try using the image based molecules in the wavelet kan, since the paper outlining these use image dataset
"""
from typing import Any
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from test_file import open_hopv15_dataset, train, test
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from wavlet import KAN
import torch.nn as nn

DEVICE = "cuda"
RANDOM_SEED = 42

class ImageData(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index) -> Any:
        if torch.torch.is_tensor(index):
            index = index.tolist()
        return (torch.from_numpy(self.data[index]).type(torch.float).to(DEVICE),
                torch.from_numpy(self.labels[index]).type(torch.float).to(DEVICE))

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    scale = True
    harvard_data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    harvard_chems = np.array([Chem.AddHs(Chem.MolFromSmiles(molecule.smiles_molecule)) for molecule in harvard_data if not(np.isnan(molecule.experimental_data.PCE))])
    harvard_labels = np.array([[molecule.experimental_data.PCE] for molecule in harvard_data if not(np.isnan(molecule.experimental_data.PCE))])
    imgs = np.array([np.array(ImageOps.grayscale(Draw.MolToImage(chem))).flatten() for chem in harvard_chems])
    train_set_x, test_x, train_set_y, test_y = train_test_split(imgs, harvard_labels, test_size = 0.2, shuffle=True, random_state = RANDOM_SEED)
    train_x, valid_x, train_y, valid_y = train_test_split(train_set_x, train_set_y, test_size=0.1, shuffle=True, random_state=RANDOM_SEED)
    if scale: 
        scaler = StandardScaler()
        pca = PCA(n_components=0.90, random_state=RANDOM_SEED)
        train_x = pca.fit_transform(scaler.fit_transform(train_x))
        valid_x = pca.transform(scaler.transform(valid_x))
        test_x = pca.transform(scaler.transform(test_x))
    train_set = DataLoader(ImageData(train_x, train_y),batch_size=100, shuffle=True) 
    valid_set = DataLoader(ImageData(valid_x, valid_y),batch_size=100, shuffle=True)
    test_set = DataLoader(ImageData(test_x, test_y),batch_size=100, shuffle=True)
    model = KAN([len(train_x[0]), 150, 100, 50, 1], 'shannon').to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 50
    for _ in range(epochs):
        train(train_set, model, loss_fn, optimizer)
        test(valid_set, model, loss_fn)
    print("Final Test Error: ")
    test(test_set, model, loss_fn)
    full_guess = []
    full_true_val = []
    for X, y in test_set:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        full_guess.extend(pred.cpu().detach().numpy())
        full_true_val.extend(y.cpu().detach().numpy())
    print("------------------------------------------")
    print(f"| PCE Predictions: {mean_absolute_error(full_true_val, full_guess):.4f} MAE, " + 
          f"{r2_score(full_true_val, full_guess):.4f} R^2, "+
          f"{root_mean_squared_error(full_true_val, full_guess):.4f} RMSE")
    print("------------------------------------------")