from math import tanh
import torch
import torch.nn as nn
from import_dataset import open_hopv15_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import random
import os


# Following an auto-encoder construction found here: https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/

    
def save_model(model, model_path, file_name, train_loss, test_loss, encoder_string, decoder_string):
    if os.path.exists(model_path):
        with open(model_path+file_name+"_report.txt", "w") as writer:
            writer.write(f"Encoder: {encoder_string}\n")
            writer.write(f"Decoder: {decoder_string}\n")
            writer.write(f"Saved Train Loss: {train_loss}\n")
            writer.write(f"Saved Test Loss: {test_loss}\n")
        torch.save(model.to('cpu').state_dict, model_path+file_name+".pt")
    else:
        os.makedirs(model_path)
        with open(model_path+file_name+"_report.txt", "w") as writer:
            writer.write(f"Encoder: {encoder_string}\n")
            writer.write(f"Decoder: {decoder_string}\n")
            writer.write(f"Saved Train Loss: {train_loss}\n")
            writer.write(f"Saved Test Loss: {test_loss}\n")
        torch.save(model.to('cpu').state_dict, model_path+file_name+".pt")

def create_layers(layer_str: str, str_to_activ: dict[str, nn.Module]) -> list[nn.Module]:
    """Converts a model string into a list of layers to be applied to a SkeletonNetwork

    Args:
        layer_str (str): The description of the model as a string (ex. 128|1->relu->1|14)
        str_to_activ (dict[str, nn.Module]): a dictionary with a str to match a function in the string to a real activation function module (ex. {'relu': nn.Relu()})

    Returns:
        list[nn.Module]: the nn.Modules that are from the layer_str
    """
    layers_list: list[str] = layer_str.split('->')
    converted_layers: list[nn.Module] = []
    for command in layers_list:
        try:
            dimensions = tuple(command.split('|'))
            in_dim: int = int(dimensions[0])
            out_dim: int = int(dimensions[1])
            converted_layers.append(nn.Linear(in_dim, out_dim))
        except:
            converted_layers.append(str_to_activ[command])
    return converted_layers

def model_string_generator(input_dim: int, num_hidden: int, num_output: int, activations: list[str], size_range: tuple[int, int]) -> str:
    """Generates a model string from an input dimension, number of hidden layers, and output dimension to be used
    for creating models, with some control of activation functions and number of weights in each linear layer

    Args:
        input_dim (int): the size of the data input
        num_hidden (int): number of hidden layers (SHOULD NOT INCLUDE INPUT/OUTPUT LAYERS)
        num_output (int): the dimension of the final answer
        activations (list[str]): the activation functions that can be randomly chosen
        size_range (tuple[int, int]): the min and max size of randomly generated layers

    Returns:
        str: a model string that can be used in the create_layers function
    """
    output_str = f"{input_dim}|"
    prev_output = random.randint(size_range[0], size_range[1])
    output_str += f"{prev_output}->"
    for i in range(num_hidden):
        new_out = random.randint(size_range[0], size_range[1])
        output_str += f"{random.choice(activations)}->{prev_output}|{new_out}->"
        prev_output = new_out
    output_str += f"{random.choice(activations)}->{prev_output}|{num_output}"
    return output_str

def create_autoencoder(molecule_dim: int, encoded_dim: int, num_hidden: int, activations: list[str], size_range: tuple[int, int]) -> tuple[str, str]:
    """_summary_

    Args:
        molecule_dim (int): _description_
        encoded_dim (int): _description_
        num_hidden (int): _description_
        activations (list[str]): _description_
        size_range (tuple[int, int]): _description_

    Returns:
        _type_: _description_
    """
    encoder_struct = model_string_generator(molecule_dim, num_hidden, encoded_dim, activations, size_range)
    decoder_struct = model_string_generator(encoded_dim, num_hidden, molecule_dim, activations, size_range)
    return (encoder_struct, decoder_struct)


class CoordinateDataset(Dataset):
    def __init__(self, coords):
        self.coordinates = coords
    def __len__(self):
        return self.coordinates.shape[0]
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return (torch.from_numpy(self.coordinates[index]).type(torch.float).to(device), 
                torch.from_numpy(self.coordinates[index]).type(torch.float).to(device))

device = 'cuda'


class MoleculeAutoEncoderTanh(nn.Module):
    def __init__(self, encode_layers, decode_layers, limit=None):
        super().__init__()
        self.encoder = nn.Sequential(
            *encode_layers,
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            *decode_layers
        )
        self.limit = limit
    def forward(self, x):
        encode = self.encoder(x.type(torch.float))
        decode = self.decoder(encode)
        if self.limit:
            decode = torch.clamp(decode, self.limit[0], self.limit[1])
        return decode

class MoleculeAutoEncoderSigmoid(nn.Module):
    def __init__(self, encode_layers, decode_layers, limit=None):
        super().__init__()
        self.encoder = nn.Sequential(
            *encode_layers,
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            *decode_layers
        )
        self.limit = limit
    def forward(self, x):
        encode = self.encoder(x.type(torch.float))
        decode = self.decoder(encode)
        if self.limit:
            decode = torch.clamp(decode, self.limit[0], self.limit[1])
        return decode
        
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

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
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # pred = (pred > 0.5).type(torch.float)
            # correct += (pred == y.unsqueeze(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

def cosine_similarity(a, b):
    return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))

if __name__ == '__main__':
    #Assume radioactive elements are not used
    symbol_to_num = ['H','He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne','Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                     'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                     'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    atom_coordinates = []
    num_models = 100
    base_path = "D://MoleculeAutoEncoder//"
    str_to_activ = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "mish": nn.Mish}
    for molecule in data:
        for conformer in molecule.conformer_info:
            for atom in conformer.atom_coordinates:
                formatted_chemical = [symbol_to_num.index(atom.chemical_id)+1, atom.x, atom.y, atom.z]
                atom_coordinates.append(formatted_chemical)
    print(len(atom_coordinates), len(atom_coordinates[0]))
    atom_coordinates = np.unique(atom_coordinates, axis=0)
    print("Completed Data set load!")
    for model_id in range(num_models):
        tanh_model_path = base_path + "tanh_models//model_" + str(model_id) + "//"
        sigmoid_model_path = base_path + "sigmoid_models//model_" + str(model_id) + "//"
        file_name = "model_"+str(model_id)
        kf = KFold(n_splits=5)
        encoder_structures = create_autoencoder(4, 1, 3, ["relu", "silu", "gelu", "mish"], (300, 600))
        encoder_layers = create_layers(encoder_structures[0], str_to_activ)
        decoder_layers = create_layers(encoder_structures[1], str_to_activ)
        tanh_model = MoleculeAutoEncoderTanh(encoder_layers, decoder_layers).to(device)
        sigmoid_model = MoleculeAutoEncoderSigmoid(encoder_layers, decoder_layers).to(device)
        tanh_loss_fn = nn.MSELoss()
        sigm_loss_fn = nn.MSELoss()
        tanh_optimizer = torch.optim.Adam(tanh_model.parameters(), lr=1e-3)
        sigm_optimizer = torch.optim.Adam(sigmoid_model.parameters(), lr=1e-3)
        epochs = 2
        print("Entering Encoder Training loop...")
        for i in range(epochs):
            for train_index, test_index in kf.split(atom_coordinates):
                train_data = CoordinateDataset(atom_coordinates[train_index])
                test_data = CoordinateDataset(atom_coordinates[test_index])
                train_loader = DataLoader(train_data, batch_size=10000, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=10000, shuffle=True)
                train_tanh_loss = train(train_loader, tanh_model, tanh_loss_fn, tanh_optimizer)
                train_sigm_loss = train(train_loader, sigmoid_model, sigm_loss_fn, sigm_optimizer)
                print("TANH:")
                test_tanh_loss = test(test_loader, tanh_model, tanh_loss_fn)
                print("SIGMOID:")
                test_sigm_loss = test(test_loader, sigmoid_model, sigm_loss_fn)
        print("Saving Model...")
        save_model(tanh_model, tanh_model_path, file_name, train_tanh_loss, test_tanh_loss, encoder_structures[0], encoder_structures[1])
        save_model(sigmoid_model, sigmoid_model_path, file_name, train_sigm_loss, test_sigm_loss, encoder_structures[0], encoder_structures[1])
    
                