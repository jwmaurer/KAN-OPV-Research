"""
Jacob Maurer
6/16/2026
Purpose Test an autoencoder that worked. No idea why. Cannot replicate for whatever reason.
"""
import numpy as np
from zmq import device
from import_dataset import open_hopv15_dataset
from vautoencoder_experimental_data import VAE, ChemistryData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from molecule_autoencoder import create_layers

if __name__ == "__main__":
    np.random.default_rng(42)
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    experimental_data = []
    for molecule in data:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.Voc) or np.isnan(molecule.experimental_data.Jsc) or np.isnan(molecule.experimental_data.PCE):
            continue
        experimental_data.append([molecule.experimental_data.HOMO, molecule.experimental_data.LUMO, molecule.experimental_data.optical_gap, molecule.experimental_data.PCE])
    np.random.shuffle(experimental_data)
    imported_model = torch.load("./combined_model/vae_computation_model.pt", weights_only=False)
    layer_strs  = ("4|1097->relu->1097|956->relu->956|916->relu->916|1193->relu->1193|2", "2|988->relu->988|853->relu->853|1052->relu->1052|1071->relu->1071|4")
    encode_layer = create_layers(layer_strs[0], {"relu": nn.ReLU(), "silu": nn.SiLU(), "selu": nn.SELU()})
    decode_layer = create_layers(layer_strs[1], {"relu": nn.ReLU(), "silu": nn.SiLU(), "selu": nn.SELU()})
    model = VAE(encode_layer,decode_layer,2).to("cuda")
    model.load_state_dict(imported_model())
    model.eval()
    loader = DataLoader(ChemistryData(np.asarray(experimental_data)), batch_size=50, shuffle=True)
    with torch.no_grad():
        for batch, x in enumerate(loader):
            out = model(x, 0.1)
            print(out.loss.item())
            if out.loss.item() > 30.0:
                print(x)