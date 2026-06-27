"""
JAcob Maurer
6/5/2026
Purpose: Utilize a Variational Auto Encoder to convert theoretical data into experimental data. Mainly for PCE use,
But will also be applied for other important variables such as HOMO, LUMO and Optical Gap
"""
import torch
import torch.nn as nn
from import_dataset import FormattedData, open_csv_dataset, open_hopv15_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

device = 'cuda'

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
    def __init__(self, input_dim=6, hidden_dim=512, latent_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=-1)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
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
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=1).mean()

        # 2. KL Divergence
        # Analytic KL for Normal distributions
        kl_loss = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1).mean()

        # 3. Total Loss (ELBO)
        loss = recon_loss + (kl_weight * kl_loss)

        return VAEOutput(z, mu, std, x_recon, loss, recon_loss, kl_loss)

# --- Training Loop Example ---
def train_step(model, batch, optimizer, kl_weight=0.5):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(batch[0], kl_weight)
    
    # Backward pass
    output.loss.backward()

    # Gradient clipping (recommended)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

    optimizer.step()
    return output.loss.item()
"""
end of code
"""

class ChemistryData(Dataset):
    def __init__(self, features, labels):
        self.data = (features, labels)
    def __len__(self):
        return len(self.data[0])
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return (torch.from_numpy(self.data[0][index]).type(torch.float).to(device), 
                torch.from_numpy(self.data[1][index]).type(torch.float).to(device))

def check_nan(item: FormattedData):
    if np.isnan(item.experimental_data.optical_gap):
        return False
    if np.isnan(item.experimental_data.HOMO):
        return False
    if np.isnan(item.experimental_data.LUMO):
        return False
    if np.isnan(item.experimental_data.PCE):
        return False
    if np.isnan(item.experimental_data.Jsc):
        return False
    if np.isnan(item.experimental_data.Voc):
        return False
    return True

if __name__ == "__main__":
    harvard_data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    harvard_data = [molecule for molecule in harvard_data if check_nan(molecule)]
    B3LYP_data = []
    BP86_data = []
    M06_2X_data = []
    PBE0_data = []
    # labels = []
    for molecule in harvard_data:
        calc = molecule.conformer_info[0].calculated_data[0]
        B3LYP_data.append([calc.HOMO, calc.LUMO, calc.Gap, calc.scharber_jsc, calc.scharber_voc, calc.scharber_pce])
        calc = molecule.conformer_info[0].calculated_data[1]
        BP86_data.append([calc.HOMO, calc.LUMO, calc.Gap, calc.scharber_jsc, calc.scharber_voc, calc.scharber_pce])
        calc = molecule.conformer_info[0].calculated_data[2]
        M06_2X_data.append([calc.HOMO, calc.LUMO, calc.Gap, calc.scharber_jsc, calc.scharber_voc, calc.scharber_pce])
        calc = molecule.conformer_info[0].calculated_data[3]
        PBE0_data.append([calc.HOMO, calc.LUMO, calc.Gap, calc.scharber_jsc, calc.scharber_voc, calc.scharber_pce])
        # labels.append([molecule.experimental_data.HOMO, molecule.experimental_data.LUMO, molecule.experimental_data.optical_gap, molecule.experimental_data.Jsc, molecule.experimental_data.Voc, molecule.experimental_data.PCE])
    print("Data Split!")
    datasets = [B3LYP_data, BP86_data, M06_2X_data, PBE0_data]
    train_loaders = []
    test_loaders = []
    for dataset in datasets:
        train_x, test_x, train_y, test_y = train_test_split(dataset, dataset, test_size=.2, random_state=42)
        train_loaders.append(DataLoader(ChemistryData(np.asarray(train_x), np.asarray(train_y)), 50, shuffle=True))
        test_loaders.append(DataLoader(ChemistryData(np.asarray(test_x), np.asarray(test_y)), 50, shuffle=True))
    print("Converted to Pytorch readable data!")
    epochs = 50
    for i, loader in enumerate(train_loaders):
        print(f"On loader: {i}")
        vae_model = VAE().to(device)
        optim = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
        for j in range(epochs):
            for batch, data in enumerate(loader):
                train_step(vae_model, data, optim)
                print(f"\tError on Batch {batch}, Epoch {j}: {train_step(vae_model, data, optim)}")
        with torch.no_grad():
            vae_model.eval()
            for batch, data in enumerate(loader):
                out = vae_model(data[0], 1.0)
                print(f"\tTest loss for batch {batch}: {out.loss.item()}")
                print(f"\t Reconstruction: {out.x_recon}")
                print(f"\t True Results: {data[1]}")