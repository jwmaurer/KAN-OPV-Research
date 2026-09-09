"""
Jacob Maurer
6/26/2026
Purpose: To determine how the checkpoint loads work for 
"""
from kan import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
from import_dataset import open_csv_dataset, open_hopv15_dataset
from molecule_autoencoder import create_layers
from AutoencoderKANs.optuna_theory_experimental import TheoryEncodedDataset
from AutoencoderKANs.vautoencoder_combined_data import VAE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    device = "cuda"
    version = "new_comp_enc_4"
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
    
    #load theoretical encoder
    theory_encoder_str = "9|729->tanh->729|493->tanh->493|778->tanh->778|5"
    theory_decoder_str = "5|730->tanh->730|449->tanh->449|404->tanh->404|9"
    theory_encoder_layer = create_layers(theory_encoder_str, {"tanh": nn.Tanh()})
    theory_decoder_layer = create_layers(theory_decoder_str, {"tanh": nn.Tanh()})
    theory_weights = torch.load("./combined_model/vae_computation_model.pt", weights_only=False)()
    
    #Load experimental model
    exp_encoder_str = "4|338->tanh->338|281->tanh->281|3"
    exp_decoder_str = "3|373->tanh->373|281->tanh->281|4"
    exp_encoder_layer = create_layers(exp_encoder_str, {"tanh": nn.Tanh()})
    exp_decoder_layer = create_layers(exp_decoder_str, {"tanh": nn.Tanh()})
    exp_weights = torch.load("./combined_model/vae_experimental_model_4.pt", weights_only=False)()
    
    #Create the autoencoders
    theo_model = VAE(theory_encoder_layer, theory_decoder_layer, 5).to(device=device)
    theo_model.load_state_dict(theory_weights)
    exp_model = VAE(exp_encoder_layer, exp_decoder_layer, 3).to(device=device)
    exp_model.load_state_dict(exp_weights)
    
    dataset = TheoryEncodedDataset(theo_model, np.asarray(usable_data), exp_model, np.asarray(usable_labels), device)
    loader = DataLoader(dataset, batch_size=2000, shuffle=True)
    
    load_version = "_with_exp_4"
    
    encoder_str = f'./KAN_files/encoder_kan{load_version}/0.12'
    transfer_str = f'./KAN_files/transfer_kan{load_version}/0.12'
    decoder_str = f'./KAN_files/decoder_kan{load_version}/0.12'
    encoder = KAN.loadckpt(encoder_str)
    transfer = KAN.loadckpt(transfer_str)
    decoder = KAN.loadckpt(decoder_str)
    
    print("Model pre-prune")
    for batch, (x, t_e, e_e, y) in enumerate(loader):
        test_dataset = create_dataset_from_data(x, y, device=device)
        print("------------------------------------------")
        print(f"| Full Model: {mean_squared_error(test_dataset['test_label'].cpu().detach().numpy(), decoder(transfer(encoder(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MSE, {mean_absolute_error(test_dataset['test_label'].cpu().detach().numpy(), decoder(transfer(encoder(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MAE")
        print("------------------------------------------")
    
    encoder_str = f'./KAN_files/encoder_kan{load_version}/0.13'
    transfer_str = f'./KAN_files/transfer_kan{load_version}/0.13'
    decoder_str = f'./KAN_files/decoder_kan{load_version}/0.13'
    encoder = KAN.loadckpt(encoder_str)
    transfer = KAN.loadckpt(transfer_str)
    decoder = KAN.loadckpt(decoder_str)
    
    print("Model post-prune")
    for batch, (x, t_e, e_e, y) in enumerate(loader):
        test_dataset = create_dataset_from_data(x, y, device=device)
        print("------------------------------------------")
        print(f"| Full Model: {mean_squared_error(test_dataset['test_label'].cpu().detach().numpy(), decoder(transfer(encoder(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MSE, {mean_absolute_error(test_dataset['test_label'].cpu().detach().numpy(), decoder(transfer(encoder(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MAE")
        print("------------------------------------------")
    
    encoder_str = f'./KAN_files/encoder_kan{load_version}/0.14'
    transfer_str = f'./KAN_files/transfer_kan{load_version}/0.14'
    decoder_str = f'./KAN_files/decoder_kan{load_version}/0.14'
    encoder = KAN.loadckpt(encoder_str)
    transfer = KAN.loadckpt(transfer_str)
    decoder = KAN.loadckpt(decoder_str)
    
    print("Post-prune, Equations fixed")
    for batch, (x, t_e, e_e, y) in enumerate(loader):
        test_dataset = create_dataset_from_data(x, y, device=device)
        print("------------------------------------------")
        print(f"| Full Model: {mean_squared_error(test_dataset['test_label'].cpu().detach().numpy(), decoder(transfer(encoder(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MSE, {mean_absolute_error(test_dataset['test_label'].cpu().detach().numpy(), decoder(transfer(encoder(test_dataset['test_input']))).cpu().detach().numpy()):.4f} MAE")
        print("------------------------------------------")
    