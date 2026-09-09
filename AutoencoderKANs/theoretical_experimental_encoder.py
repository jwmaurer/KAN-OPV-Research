"""
Jacob Maurer
6/22/2026
Purpose: Combine the enocder and decoder architectures from the variational autoencoders
using a go between network. Make sure the scalers match for decoding and encoding the input features.
Hope this works. The amount of new data could be worth it.
"""
from import_dataset import open_hopv15_dataset
from kan import *
from molecule_autoencoder import create_layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from AutoencoderKANs.vautoencoder_combined_data import VAE
import numpy as np
import torch
import torch.nn as nn

class TheoryEncodedDataset(Dataset):
    def __init__(self, theo_encoder, theo_data, exp_encoder, exp_data, device):
        self.theory_encoder = theo_encoder
        self.experimental_encoder = exp_encoder
        self.inputs = torch.from_numpy(theo_data).type(torch.float).to(device)
        self.outputs = torch.from_numpy(exp_data).type(torch.float).to(device)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        theo_encoded = None
        exp_encoded = None
        with torch.no_grad():
            mu, std = self.theory_encoder.encode(self.inputs[index])
            theo_encoded = self.theory_encoder.reparameterize(mu, std)
            mu, std = self.experimental_encoder.encode(self.outputs[index])
            exp_encoded = self.experimental_encoder.reparameterize(mu, std)
        return self.inputs[index], theo_encoded, exp_encoded, self.outputs[index]

if __name__ == "__main__":
    #Define the device, variables, and working data
    device = "cuda"
    version = "new_comp_enc_4"
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    usable_data = []
    usable_labels = []
    
    #Set up the original scaler used to train the autoencoder for the combined data
    scaler = StandardScaler()
    scaler.mean_ = np.asarray([2.31010435,-5.29527025,-2.25683534])
    scaler.scale_ = np.asarray([0.66287706,1.12101143,0.73894985])
    scaler.var_ = np.asarray([0.439406,1.25666663,0.54604688])
    
    #Set up the original scaler used to train the autoencoder for the experimental data
    exp_scaler = StandardScaler()
    exp_scaler.mean_ = np.asarray([-5.2848037, -3.37753148, 1.75974475, 3.84265556])
    exp_scaler.scale_ = np.asarray([0.22684783, 0.30689121, 0.25300771, 2.15390711])
    exp_scaler.var_ = np.asarray([0.05145994, 0.09418221, 0.0640129, 4.63931586])
    
    #Labels for the binary columns
    labels = ["camb3lyp/6-31g",
        "QChem PBE0/def2-SVP DFT",
        "QChem B3LYP/def2-SVP DFT",
        "b3lyp/6-31g(d)",
        "QChem BP86/def2-SVP DFT",
        "QChem M06-2X/def2-SVP DFT"]
    
    #Filter out all of the molecules without PCE values
    for molecule in data:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.PCE):
            continue
        for conformer in molecule.conformer_info:
            for calc in conformer.calculated_data:
                temp = [calc.HOMO, calc.LUMO, calc.Gap]
                usable_data.append(temp)
                usable_labels.append([molecule.experimental_data.HOMO, molecule.experimental_data.LUMO, molecule.experimental_data.optical_gap, molecule.experimental_data.PCE])
    
    #Scale the data
    usable_data = scaler.transform(usable_data)
    usable_labels = exp_scaler.transform(usable_labels)
    
    #load theoretical encoder
    theory_encoder_str = "3|563->tanh->563|532->tanh->532|523->tanh->523|535->tanh->535|2"
    theory_decoder_str = "2|533->tanh->533|584->tanh->584|594->tanh->594|562->tanh->562|3"
    theory_encoder_layer = create_layers(theory_encoder_str, {"tanh": nn.Tanh()})
    theory_decoder_layer = create_layers(theory_decoder_str, {"tanh": nn.Tanh()})
    theory_weights = torch.load("./combined_model/vae_computation_model_nonu_4.pt", weights_only=False)()
    
    #Load experimental model
    exp_encoder_str = "4|329->tanh->329|264->tanh->264|3"
    exp_decoder_str = "3|308->tanh->308|339->tanh->339|4"
    exp_encoder_layer = create_layers(exp_encoder_str, {"tanh": nn.Tanh()})
    exp_decoder_layer = create_layers(exp_decoder_str, {"tanh": nn.Tanh()})
    exp_weights = torch.load("./combined_model/vae_experimental_model_3.pt", weights_only=False)()
    
    #Create the autoencoders
    theo_model = VAE(theory_encoder_layer, theory_decoder_layer, 2).to(device=device)
    theo_model.load_state_dict(theory_weights)
    exp_model = VAE(exp_encoder_layer, exp_decoder_layer, 3).to(device=device)
    exp_model.load_state_dict(exp_weights)
    
    #Create the dataset
    dataset = TheoryEncodedDataset(theo_model, np.asarray(usable_data), exp_model, np.asarray(usable_labels), device)
    
    #Create the KAN models
    encoder_kan = KAN([3, 2, 2], grid=8, k=5, grid_eps=0, ckpt_path=f"./KAN_files/encoder_kan_{version}", device=device)
    transfer_kan = KAN([2, 4, 3], grid=8, k=5, grid_eps=0, ckpt_path=f"./KAN_files/transfer_kan_{version}", device=device)
    decoder_kan = KAN([3, 3, 4], grid=8, k=5, grid_eps=0, ckpt_path=f"./KAN_files/decoder_kan_{version}", device=device)
    
    #Train the KAN models
    for _ in range(2):
        loader = DataLoader(dataset, batch_size=2000, shuffle=True)
        for batch, (x, t_e, e_e, y) in enumerate(loader):
            current_dataset_encode = create_dataset_from_data(x, t_e, device=device)
            current_dataset_transfer = create_dataset_from_data(t_e, e_e, device=device)
            current_dataset_decode = create_dataset_from_data(e_e, y, device=device)
            encoder_kan.fit(current_dataset_encode, steps=50, update_grid=False) # https://github.com/KindXiaoming/pykan/issues/230
            transfer_kan.fit(current_dataset_transfer, steps=50, update_grid=False)
            decoder_kan.fit(current_dataset_decode, steps=50, update_grid=False)
            print("------------------------------------------")
            print(f"| Batch: {batch}")
            print(f"| Encoder KAN: {mean_squared_error(current_dataset_encode['test_label'].cpu().detach().numpy(), encoder_kan(current_dataset_encode['test_input']).cpu().detach().numpy()):.4f} MSE, {mean_absolute_error(current_dataset_encode['test_label'].cpu().detach().numpy(), encoder_kan(current_dataset_encode['test_input']).cpu().detach().numpy()):.4f} MAE")
            print(f"| Transfer KAN: {mean_squared_error(current_dataset_transfer['test_label'].cpu().detach().numpy(), transfer_kan(current_dataset_transfer['test_input']).cpu().detach().numpy()):.4f} MSE, {mean_absolute_error(current_dataset_transfer['test_label'].cpu().detach().numpy(), transfer_kan(current_dataset_transfer['test_input']).cpu().detach().numpy()):.4f} MAE")
            print(f"| Decoder KAN: {mean_squared_error(current_dataset_decode['test_label'].cpu().detach().numpy(), decoder_kan(current_dataset_decode['test_input']).cpu().detach().numpy()):.4f} MSE, {mean_absolute_error(current_dataset_decode['test_label'].cpu().detach().numpy(), decoder_kan(current_dataset_decode['test_input']).cpu().detach().numpy()):.4f} MAE")
            print("------------------------------------------")
        
    #Prune the models, if possible
    try:
        encoder_kan = encoder_kan.prune()
    except:
        print("Encoder cannot be pruned!")
    try:
        transfer_kan = transfer_kan.prune()
    except:
        print("Transfer cannot be pruned!")
    try:
        decoder_kan = decoder_kan.prune()
    except:
        print("Decoder cannot be pruned!")
        
    #Quick Test on a single batch
    for batch, (x, t_e, e_e, y) in enumerate(loader):
        test_dataset = create_dataset_from_data(x, y, device=device)
        print("------------------------------------------")
        print(f"| Full Model: {mean_squared_error(test_dataset['test_label'].cpu().detach().numpy(), decoder_kan(transfer_kan(encoder_kan(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MSE, {mean_absolute_error(test_dataset['test_label'].cpu().detach().numpy(), decoder_kan(transfer_kan(encoder_kan(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MAE")
        print("------------------------------------------")
    
    #Plot the models to check for symbolic substitution viability
    encoder_kan.plot(title="Encoder KAN", folder=f"./KAN_files/encoder_figs_{version}")
    transfer_kan.plot(title="Transfer KAN", folder=f"./KAN_files/transfer_figs_{version}")
    decoder_kan.plot(title="Decoder KAN", folder=f"./KAN_files/decoder_figs_{version}")
    
    encoder_kan.auto_symbolic()
    transfer_kan.auto_symbolic()
    decoder_kan.auto_symbolic()
    
    print("Symbolic Formulas found for each KAN. Represent the MLP autoencoders, which have errors. Interpret at your own risk")
    
    print("ENCODER KAN:")
    for eq in encoder_kan.symbolic_formula()[0]:
        print(f"\t{ex_round(eq, 4)}")
        
    print("TRANSFER KAN:")
    for eq in transfer_kan.symbolic_formula()[0]:
        print(f"\t{ex_round(eq, 4)}")
    
    print("DECODER KAN:")
    for eq in decoder_kan.symbolic_formula()[0]:
        print(f"\t{ex_round(eq, 4)}")
    
    plt.show()
        