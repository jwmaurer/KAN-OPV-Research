"""
Jacob Maurer
6/10/2026
Purpose: To make an autoencoder that produces the experimental values
"""
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from import_dataset import FormattedData, open_hopv15_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from molecule_autoencoder import create_autoencoder, create_layers, save_model
from AutoencoderKANs.vautoencoder_combined_data import VAE, train_step

device = "cuda"
HOPV_AVG_POWER = 0.985641822292957
HOPV_AVG_FF = 53.603888888888875/100.0

"""
The following code was taken from: https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
"""
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class VAEOutput:
    z: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor

class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            *encoder,
            nn.Softmax(dim=-1)
        )
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_std = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            *decoder
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        # Softplus + epsilon for stable std deviation
        std = F.softplus(self.fc_std(h)) + 1e-6
        return mu, std

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, kl_weight=1.0):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        x_recon = self.decode(z)

        # 1. Reconstruction Loss (Binary Cross Entropy for MNIST)
        # Sum over features, mean over batch
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # 2. KL Divergence
        # Analytic KL for Normal distributions
        kl_loss = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=0).sum()

        # 3. Total Loss (ELBO)
        loss = recon_loss + (kl_weight * kl_loss)

        return VAEOutput(z, mu, std, x_recon, loss, recon_loss, kl_loss)

# --- Training Loop Example ---
def train_step(model, batch, optimizer, kl_weight=1.0):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(batch, kl_weight)
    
    # Backward pass
    output.loss.backward()

    # Gradient clipping (recommended)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return output.loss.item()
"""
end of code
"""

#
# PCA: Encodes to dimension 4 with 90% loss
# Autoencoder should reach this squish?
#

class ChemistryData(Dataset):
    def __init__(self, features):
        self.data = features
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return torch.from_numpy(self.data[index]).type(torch.float).to(device)
        
if __name__ == "__main__":
    np.random.default_rng(42)
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    experimental_data = []
    for molecule in data:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.Voc) or np.isnan(molecule.experimental_data.Jsc) or np.isnan(molecule.experimental_data.PCE):
            continue
        experimental_data.append([molecule.experimental_data.HOMO, molecule.experimental_data.LUMO, molecule.experimental_data.optical_gap, molecule.experimental_data.PCE])
    scaler = MinMaxScaler()
    experimental_data = scaler.fit_transform(experimental_data)
    pca = PCA(n_components=0.9)
    final = pca.fit_transform(experimental_data)
    print(pca.explained_variance_ratio_)
    print(pca.components_)
    print(final[:5])
    test_size = 0.3
    train_set, test_set = experimental_data[int(test_size*len(experimental_data)):], experimental_data[:int(test_size*len(experimental_data))]
    print(len(train_set))
    print(len(test_set))
    epochs = 100
    encode_dim = len(pca.explained_variance_ratio_)
    train_loader = DataLoader(ChemistryData(np.asarray(train_set)), batch_size=50, shuffle=True)
    test_loader = DataLoader(ChemistryData(np.asarray(test_set)), batch_size=50, shuffle=True)
    layer_strs = create_autoencoder(len(train_set[0]), encode_dim, 1, ["sig"], (3, 5))
    # layer_strs  = ("4|1097->relu->1097|956->relu->956|916->relu->916|1193->relu->1193|2", "2|988->relu->988|853->relu->853|1052->relu->1052|1071->relu->1071|4")
    encode_layer = create_layers(layer_strs[0], {"relu": nn.ReLU(), "silu": nn.SiLU(), "selu": nn.SELU(), "tanh": nn.Tanh(), "sig": nn.Sigmoid()})
    decode_layer = create_layers(layer_strs[1], {"relu": nn.ReLU(), "silu": nn.SiLU(), "selu": nn.SELU(), "tanh": nn.Tanh(), "sig": nn.Sigmoid()})
    print(f"Encoder: {layer_strs[0]}")
    print(f"Decoder: {layer_strs[1]}")
    model = VAE(encode_layer, decode_layer, encode_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    kl_w = 0.1
    training = "y"
    num_iters = 0
    while training == "y":
        for e in range(epochs):
            total_loss = 0.0
            batches = 0
            for batch, x in enumerate(train_loader):
                total_loss += train_step(model, x, optim, kl_w)
                batches += 1
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_batches = 0
                for batch, x in enumerate(test_loader):
                    out = model(x, kl_w)
                    test_loss += out.loss.item()
                    test_batches += 1
            print("----------------------------------------------")
            print(f"|  Epoch: {e + 100*num_iters}")
            print(f"|  Train Loss: {total_loss/batches:.4f}, which is approximately {total_loss/len(train_set):.6f} per molecule")
            print(f"|  Test Loss: {test_loss/test_batches:.4f}, which is approximately {test_loss/len(test_set):.6f} per molecule")
            print("----------------------------------------------")
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_batches = 0
            for batch, x in enumerate(test_loader):
                out = model(x, kl_w)
                print(out.x_recon)
                print(x)
                print(out.loss_recon.item())
        training = input("Continue (y or n)? ")
        num_iters += 1
    save_model_bool = input("Save (y or n)? ")
    if save_model_bool == "y":
        print("Saving model...")
        save_model(model, "./combined_model/", "vae_experimental_model_minmax", total_loss, test_loss, layer_strs[0], layer_strs[1])