"""
Jacob Maurer
6/24/2026
Purpose:Find the best hyperparameters for the theory, transfer, and experimental KAN models.
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
import optuna

RANDOM_SEED = 42

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

def format_data():
    device = "cuda"
    version = "new_loader"
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    usable_data = []
    usable_labels = []
    
    #Set up the original scaler used to train the autoencoder for the combined data
    scaler = StandardScaler()
    scaler.mean_ = np.asarray([2.0694012, -4.73794015, -2.0196512, 0.22442731, 0.03303103, 0.03303103, 0.64344856, 0.03303103, 0.03303103])
    scaler.scale_ = np.asarray([0.93441318, 1.91289926, 0.96923095, 0.41720462, 0.1787176, 0.1787176, 0.4789807, 0.1787176, 0.1787176])
    scaler.var_ = np.asarray([0.87312799, 3.65918358, 0.93940863, 0.1740597, 0.03193998, 0.03193998, 0.22942251, 0.03193998, 0.03193998])
    
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
                basis_set = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                basis_set[labels.index(calc.basis_set_description)] = 1
                temp.extend(basis_set)
                usable_data.append(temp)
                usable_labels.append([molecule.experimental_data.HOMO, molecule.experimental_data.LUMO, molecule.experimental_data.optical_gap, molecule.experimental_data.PCE])
    
    #Scale the data
    usable_data = scaler.transform(usable_data)
    usable_labels = exp_scaler.transform(usable_labels)
    def objective(trial):
        epochs = trial.suggest_int("epochs", 1, 10)
        encoding_model = trial.suggest_categorical("theory", ["./theory/vae_computation_model", "./theory/vae_computation_model_2"])
        decoding_model = trial.suggest_categorical("experimental", ["./experiment/vae_experimental_model", "./experiment/vae_experimental_model_2", "./experiment/vae_experimental_model_3", "./experiment/vae_experimental_model_4", "./experiment/vae_experimental_model_5"])
        encoder_hyperparams = {
            "lamb": trial.suggest_float("encoder_lambda", 0.0, 1.0),
            "lamb_ent": trial.suggest_float("encoder_lambda_entropy", 0.0, 10.0),
            "lamb_l1": trial.suggest_float("encoder_lambda_l1", 0.0, 1.0),
            "lamb_coef": trial.suggest_float("encoder_lambda_coefficient", 0.0, 5.0),
            "lamb_coefdiff": trial.suggest_float("encoder_lambda_coefficient_difference", 0.0, 5.0),
            "steps": trial.suggest_int("encoder_steps", 0, 50)
        }
        transfer_hyperparams = {
            "lamb": trial.suggest_float("transfer_lambda", 0.0, 1.0),
            "lamb_ent": trial.suggest_float("transfer_lambda_entropy", 0.0, 10.0),
            "lamb_l1": trial.suggest_float("transfer_lambda_l1", 0.0, 1.0),
            "lamb_coef": trial.suggest_float("transfer_lambda_coefficient", 0.0, 5.0),
            "lamb_coefdiff": trial.suggest_float("transfer_lambda_coefficient_difference", 0.0, 5.0),
            "steps": trial.suggest_int("transfer_steps", 0, 50)
        }
        decoder_hyperparams = {
            "lamb": trial.suggest_float("decoder_lambda", 0.0, 1.0),
            "lamb_ent": trial.suggest_float("decoder_lambda_entropy", 0.0, 10.0),
            "lamb_l1": trial.suggest_float("decoder_lambda_l1", 0.0, 1.0),
            "lamb_coef": trial.suggest_float("decoder_lambda_coefficient", 0.0, 5.0),
            "lamb_coefdiff": trial.suggest_float("decoder_lambda_coefficient_difference", 0.0, 5.0),
            "steps": trial.suggest_int("decoder_steps", 0, 50)
        }
        theory_encoder_str = ""
        theory_decoder_str = ""
        with open(encoding_model+"_report.txt", 'rt') as file:
            for line in file.readlines():
                if "Encoder" in line:
                    theory_encoder_str = line.strip().split(' ')[0]
                elif "Decoder" in line:
                    theory_decoder_str = line.strip().split(' ')[0]
        theory_encoder_layer = create_layers(theory_encoder_str, {"tanh": nn.Tanh(), "sig": nn.Sigmoid()})
        theory_decoder_layer = create_layers(theory_decoder_str, {"tanh": nn.Tanh(), "sig": nn.Sigmoid()})
        theory_weights = torch.load(encoding_model+".pt", weights_only=False)()
        
        exp_encoder_str = ""
        exp_decoder_str = ""
        with open(decoding_model+"_report.txt", 'rt') as file:
            for line in file.readlines():
                if "Encoder" in line:
                    exp_encoder_str = line.strip().split(' ')[0]
                elif "Decoder" in line:
                    exp_decoder_str = line.strip().split(' ')[0]
        exp_encoder_layer = create_layers(exp_encoder_str, {"tanh": nn.Tanh(), "sig": nn.Sigmoid(), "relu": nn.ReLU()})
        exp_decoder_layer = create_layers(exp_decoder_str, {"tanh": nn.Tanh(), "sig": nn.Sigmoid(), "relu": nn.ReLU()})
        exp_weights = torch.load(decoding_model+".pt", weights_only=False)()
        
        theo_model = VAE(theory_encoder_layer, theory_decoder_layer, 5).to(device=device)
        theo_model.load_state_dict(theory_weights)
        exp_model = VAE(exp_encoder_layer, exp_decoder_layer, 3).to(device=device)
        exp_model.load_state_dict(exp_weights)
        dataset = TheoryEncodedDataset(theo_model,usable_data,exp_model,usable_labels,device)
        
        encoder_grid_size = trial.suggest_int(0, 10)
        encoder_k = trial.suggest_int(0, 10)
        encoder_middle_set = trial.suggest_int(1, 20)
        
        transfer_grid_size = trial.suggest_int(0, 10)
        transfer_k = trial.suggest_int(0, 10)
        transfer_middle_set = trial.suggest_int(1, 20)
        
        decoder_grid_size = trial.suggest_int(0, 10)
        decoder_k = trial.suggest_int(0, 10)
        decoder_middle_set = trial.suggest_int(1, 20)
        
        encoder_kan = KAN([9, encoder_middle_set, 5], grid=encoder_grid_size, k=encoder_k, grid_eps=0, ckpt_path=f"./KAN_files/encoder_kan_{version}", device=device, seed=RANDOM_SEED)
        transfer_kan = KAN([5, transfer_middle_set, 3], grid=transfer_grid_size, k=transfer_k, grid_eps=0, ckpt_path=f"./KAN_files/transfer_kan_{version}", device=device, seed=RANDOM_SEED)
        decoder_kan = KAN([3, decoder_middle_set, 4], grid=decoder_grid_size, k=decoder_k, grid_eps=0, ckpt_path=f"./KAN_files/decoder_kan_{version}", device=device, seed=RANDOM_SEED)
        
        loader = DataLoader(dataset, batch_size=2000, shuffle=True)
        for epoch in range(epochs):
            loader = DataLoader(dataset, batch_size=2000, shuffle=True)
            for batch, (x, t_e, e_e, y) in enumerate(loader):
                current_dataset_encode = create_dataset_from_data(x, t_e, device=device)
                current_dataset_transfer = create_dataset_from_data(t_e, e_e, device=device)
                current_dataset_decode = create_dataset_from_data(e_e, y, device=device)
                encoder_kan.fit(current_dataset_encode, steps=encoder_hyperparams["steps"], lamb=encoder_hyperparams["lamb"], lamb_l1=encoder_hyperparams["lamb_l1"], lamb_entropy=encoder_hyperparams["lamb_ent"], lamb_coef=encoder_hyperparams["lamb_coef"], lamb_coefdiff=encoder_hyperparams["lamb_coefdiff"], update_grid=False) # https://github.com/KindXiaoming/pykan/issues/230
                transfer_kan.fit(current_dataset_transfer, steps=transfer_hyperparams["steps"], lamb=transfer_hyperparams["lamb"], lamb_l1=transfer_hyperparams["lamb_l1"], lamb_entropy=transfer_hyperparams["lamb_ent"], lamb_coef=transfer_hyperparams["lamb_coef"], lamb_coefdiff=transfer_hyperparams["lamb_coefdiff"], update_grid=False)
                decoder_kan.fit(current_dataset_decode, steps=decoder_hyperparams["steps"], lamb=decoder_hyperparams["lamb"], lamb_l1=decoder_hyperparams["lamb_l1"], lamb_entropy=decoder_hyperparams["lamb_ent"], lamb_coef=decoder_hyperparams["lamb_coef"], lamb_coefdiff=decoder_hyperparams["lamb_coefdiff"], update_grid=False)
        
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
        
        for batch, (x, t_e, e_e, y) in enumerate(loader):
            test_dataset = create_dataset_from_data(x, y, device=device)
            print("------------------------------------------")
            print(f"| Full Model: {mean_squared_error(test_dataset['test_label'].cpu().detach().numpy(), decoder_kan(transfer_kan(encoder_kan(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MSE, {mean_absolute_error(test_dataset['test_label'].cpu().detach().numpy(), decoder_kan(transfer_kan(encoder_kan(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MAE")
            print("------------------------------------------")
        return mean_absolute_error(test_dataset['test_label'].cpu().detach().numpy(), decoder_kan(transfer_kan(encoder_kan(test_dataset['test_input']))).cpu().detach().numpy())
    return objective
        
        

if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(format_data(), n_trials=1000)
    print(study.best_params)