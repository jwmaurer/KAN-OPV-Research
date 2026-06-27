"""
Jacob Maurer
6/15/2026
Purpose: Make an autoencoder that can find the latent space for experimental data
"""
from import_dataset import open_hopv15_dataset
from molecule_autoencoder import MoleculeAutoEncoderSigmoid, create_autoencoder, create_layers, MoleculeAutoEncoderTanh, train, test, save_model, CoordinateDataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
if __name__ == "__main__":
    device = "cuda"
    np.random.seed(42)
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    experimental_data = []
    for molecule in data:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.Voc) or np.isnan(molecule.experimental_data.Jsc) or np.isnan(molecule.experimental_data.PCE):
            continue
        experimental_data.append([molecule.experimental_data.HOMO, molecule.experimental_data.LUMO, molecule.experimental_data.optical_gap, molecule.experimental_data.PCE])
    np.random.shuffle(experimental_data)
    experimental_data = np.asarray(experimental_data)
    test_size = 0.1
    train_set, test_set = experimental_data[int(test_size*len(experimental_data)):], experimental_data[:int(test_size*len(experimental_data))]
    print(train_set.shape)
    print(test_set.shape)
    train_loader = DataLoader(CoordinateDataset(np.asarray(train_set)), batch_size=100, shuffle=True)
    test_loader = DataLoader(CoordinateDataset(np.asarray(test_set)), batch_size=10, shuffle=True)
    layer_strs = create_autoencoder(4, 2, 6, ["silu"], (100, 200))
    encoder_layers= create_layers(layer_strs[0], {"relu": nn.ReLU(), "silu": nn.SiLU()})
    decoder_layers= create_layers(layer_strs[1], {"relu": nn.ReLU(), "silu": nn.SiLU()})
    print(f"Encoder: {layer_strs[0]}")
    print(f"Decoder: {layer_strs[1]}")
    model = MoleculeAutoEncoderSigmoid(encoder_layers, decoder_layers).to(device)
    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 2000
    for i in range(epochs):
        train_loss = train(train_loader, model, loss, optimizer)
        test_loss = test(test_loader, model, loss)
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_loader):
            out = model(x)
            print(x)
            print(out)
    save_model_bool = input("Save (y or n)?: ")
    if save_model_bool == "y":
        print("Saving model...")
        save_model(model, "./combined_model/", "ae_experiment_model.pt", train_loss, test_loss, layer_strs[0], layer_strs[1])