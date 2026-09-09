"""
Jacob Maurer
6/10/2026
Purpose: Train encoders on the newly created combined dataset
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from import_dataset import FormattedData, open_csv_dataset, open_hopv15_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import polars as pl

from molecule_autoencoder import create_autoencoder, create_layers, save_model

device = "cuda"

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

class ChemistryData(Dataset):
    def __init__(self, features):
        self.data = features
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return torch.from_numpy(self.data[index]).type(torch.float).to(device), 0

def make_standard(mean, std):
    def inner(x):
        return (x - mean)/std
    return inner

if __name__ == "__main__":
    data = open_csv_dataset("./combined_data.csv")
    # data = data.with_columns((pl.col("is_camb3lyp/6-31g") + pl.col("is_b3lyp/6-31g(d)")).alias("is_majority"))
    # for col in data.columns:
    #     data = data.with_columns(data[col].map_elements(make_standard(data[col].mean(), data[col].std())).alias(col))
    test_size = 0.3
    numpy_data = data.select(pl.col("gap"), pl.col("homo"), pl.col("lumo")).unique().to_numpy()
    scaler = StandardScaler()
    numpy_data = scaler.fit_transform(numpy_data)
    print(scaler.mean_)
    print(scaler.scale_)
    print(scaler.var_)
    # train_set, test_set = numpy_data[int(test_size*numpy_data.shape[0]):], numpy_data[:int(test_size*numpy_data.shape[0])]
    # pca = PCA(n_components=0.9)
    # final = pca.fit_transform(numpy_data)
    # print(pca.explained_variance_ratio_)
    # print(pca.components_)
    # print(final[:5])
    # train_loader = DataLoader(ChemistryData(train_set), batch_size=1000, shuffle=True)
    # test_loader = DataLoader(ChemistryData(test_set), batch_size=1000, shuffle=True)
    # epochs = 10
    # encode_dim = len(pca.explained_variance_ratio_)
    # layer_strs = create_autoencoder(3, encode_dim, 3, ["tanh"], (500, 600))
    # encode_layer = create_layers(layer_strs[0], {"relu": nn.ReLU(), "silu": nn.SiLU(), "selu": nn.SELU(), "tanh": nn.Tanh(), "sig": nn.Sigmoid()})
    # decode_layer = create_layers(layer_strs[1], {"relu": nn.ReLU(), "silu": nn.SiLU(), "selu": nn.SELU(), "tanh": nn.Tanh(), "sig": nn.Sigmoid()})
    # print(f"Encoder: {layer_strs[0]}")
    # print(f"Decoder: {layer_strs[1]}")
    # model = VAE(encode_layer, decode_layer, encode_dim).to(device)
    # optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # training = "y"
    # kl_w = 0.01
    # curr_iter = 0
    # while training == "y":
    #     for e in range(epochs):
    #         total_loss = 0.0
    #         batches = 0
    #         for batch, (x, y) in enumerate(train_loader):
    #             total_loss += train_step(model, x, optim, kl_w)
    #             batches += 1
    #         model.eval()
    #         with torch.no_grad():
    #             test_loss = 0.0
    #             test_batches = 0
    #             for batch, (x, y) in enumerate(test_loader):
    #                 out = model(x, kl_w)
    #                 test_loss += out.loss.item()
    #                 test_batches += 1
    #         print("----------------------------------------------")
    #         print(f"|  Epoch: {e + epochs * curr_iter}")
    #         print(f"|  Train Loss: {total_loss:.4f}, which is approximately {total_loss/train_set.shape[0]:.6f} per molecule")
    #         print(f"|  Test Loss: {test_loss:.4f}, which is approximately {test_loss/test_set.shape[0]:.6f} per molecule")
    #         print("----------------------------------------------")
    #     model.eval()
    #     with torch.no_grad():
    #         test_loss = 0.0
    #         test_batches = 0
    #         for batch, (x, y) in enumerate(test_loader):
    #             out = model(x, kl_w)
    #             test_loss += out.loss.item()
    #             print(out.loss.item())
    #             print(out.x_recon)
    #             print(x)
                
    #     training = input("Continue (y or n)? ")
    #     curr_iter += 1
    # save_model_bool = input("Save (y or n)? ")
    # if save_model_bool == "y":
    #     print("Saving model...")
    #     save_model(model, "./combined_model/", "vae_computation_model_nonu_4", total_loss, test_loss, layer_strs[0], layer_strs[1])