"""
Jacob Maurer
6/13/2026
Purpose: Trian more autoencoders found from the jupyter notebook
"""
from import_dataset import open_csv_dataset, open_hopv15_dataset, mol_atom_dist
from rdkit import Chem
from rdkit.Chem import Draw
from molecule_autoencoder import MoleculeAutoEncoderSigmoid, create_autoencoder, create_layers, save_model, train, test, CoordinateDataset, MoleculeAutoEncoderTanh
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import ImageOps
import pandas as pd




if __name__ == "__main__":
    device = "cuda"
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    csv_data = open_csv_dataset("./OPV_Datasets/opv_db.csv")['smile']
    smiles = [molecule.smiles_molecule for molecule in data if not np.isnan(molecule.experimental_data.PCE)]
    smiles.extend(csv_data.to_list())
    smiles = np.unique(smiles)
    test_size = 0.2
    train_set, test_set = smiles[int(test_size*len(smiles)):], smiles[:int(test_size*len(smiles))]
    print("Imported All Data!")
    print(len(train_set))
    print(len(test_set))
    layer_strs = create_autoencoder(90000, 150, 3, ["relu", "silu"], (1000, 1500))
    encoder_layers= create_layers(layer_strs[0], {"relu": nn.ReLU(), "silu": nn.SiLU(), "tanh": nn.Tanh(), "sig": nn.Sigmoid()})
    decoder_layers= create_layers(layer_strs[1], {"relu": nn.ReLU(), "silu": nn.SiLU(), "tanh": nn.Tanh(), "sig": nn.Sigmoid()})
    model = MoleculeAutoEncoderSigmoid(encoder_layers, decoder_layers).to(device)
    print(f"Encoder: {layer_strs[0]}")
    print(f"Decoder: {layer_strs[1]}")
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        sample = np.random.choice(train_set, 2000)
        imgs = [np.asarray(ImageOps.grayscale(Draw.MolToImage(Chem.AddHs(Chem.MolFromSmiles(smile))))).flatten() for smile in sample]
        frame = pd.DataFrame(imgs)
        print(frame.head())
        formatted = drop_correlated_columns(drop_no_variance_columns(frame))
        print("All Data Formatted!")
        train_sample, valid_sample = formatted[int(test_size*len(formatted)):], formatted[:int(test_size*len(formatted))]
        train_loader = DataLoader(CoordinateDataset(np.asarray(train_sample)), batch_size=100, shuffle=True)
        test_loader = DataLoader(CoordinateDataset(np.asarray(valid_sample)), batch_size=100, shuffle=True)
        train_loss = train(train_loader, model, loss, optimizer)
        test_loss = test(test_loader, model, loss)
    print("Training Complete! running on the test set...")
    
    # with open("./image_model/image_data_autoencoder_non_clamp_2_report.txt", 'rt') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         if "Encoder" in line:
    #             encoder_layers = create_layers(line.strip().split(' ')[1], {"relu": nn.ReLU(), "silu": nn.SiLU(), "tanh": nn.Tanh(), "sig": nn.Sigmoid()})
    #             print(line)
    #         elif "Decoder" in line:
    #             decoder_layers = create_layers(line.strip().split(' ')[1], {"relu": nn.ReLU(), "silu": nn.SiLU(), "tanh": nn.Tanh(), "sig": nn.Sigmoid()})
    #             print(line)
    # state_dict = torch.load("./image_model/image_data_autoencoder_non_clamp_2.pt", weights_only=False)
    
    
    # model.load_state_dict(state_dict())
    # max_epochs = 100
    # min_MSE = 1000
    # train_loss = 20000
    # test_loss = 20000
    # curr_epoch = 0
    # print("Beginning Training!")
    # while test_loss > min_MSE:
    #     print(f"Epoch: {curr_epoch}")
    #     if max_epochs == curr_epoch:
    #         break
    #     train_loss = train(train_loader, model, loss, optimizer)
    #     test_loss = test(test_loader, model, loss)
    #     curr_epoch += 1
    # if max_epochs == curr_epoch:
    #     print(f"This parameter set could not acheive an MSE of {min_MSE} within {max_epochs} epochs. This would use too many resources, rerun to pursue a different path.")
    # else:
    #     print("Continued training resulted in lower MSE! reload to keep going")
    #     save_model(model, "./image_model/", "image_data_autoencoder_best",train_loss,test_loss, layer_strs[0], layer_strs[1])
