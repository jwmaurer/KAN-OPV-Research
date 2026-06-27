"""
Jacob Maurer
6/2/2026
Purpose: Create an optuna optimizer for a KAN network for image dataset
"""
from kan import *
import optuna
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import ImageOps
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from import_dataset import open_hopv15_dataset

RANDOM_SEED = 42

def load_data():
    harvard_data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    harvard_chems = np.array([Chem.AddHs(Chem.MolFromSmiles(molecule.smiles_molecule)) for molecule in harvard_data])
    harvard_labels = np.array([molecule.experimental_data.PCE for molecule in harvard_data])
    harvard_chems = harvard_chems[~np.isnan(harvard_labels)]
    harvard_labels = harvard_labels[~np.isnan(harvard_labels)]
    harvard_labels = torch.from_numpy(harvard_labels).type(torch.float)
    imgs = np.array([np.array(ImageOps.grayscale(Draw.MolToImage(chem))).flatten() for chem in harvard_chems])
    scaler = StandardScaler()
    scaled_imgs = scaler.fit_transform(imgs)
    pca = PCA(n_components=0.9, random_state=RANDOM_SEED)
    final_imgs = torch.from_numpy(pca.fit_transform(scaled_imgs)).type(torch.float)
    dataset = create_dataset_from_data(final_imgs, harvard_labels, device='cuda')
    print(dataset)
    def objective(trial):
        depth = trial.suggest_int("depth", 3, 10)
        layers = [len(dataset["train_input"][0])]
        for i in range(depth-2):
            layers.append(trial.suggest_int(f"layer_{i+1}", 5, 100))
        layers.append(1)
        grid = trial.suggest_int("grid_size", 5, 10)
        k = trial.suggest_int("k", 3, 8)
        steps = trial.suggest_int("steps", 30, 100)
        lamb = trial.suggest_float("lamb", 0.01, 0.99)
        lamb_ent = trial.suggest_float("lamb_ent", 0.0, 10.0)
        print(f"Depth: {depth}")
        print(f"Layers: {layers}")
        print(f"Grid: {grid}")
        print(f"K: {k}")
        print(f"Steps: {steps}")
        print(f"Lambda: {lamb}")
        print(f"Lambda Entropy: {lamb_ent}")
        #Commented for non-input in case a faster alternative is found
        kan = KAN(width=layers, grid=grid, k=k, seed=RANDOM_SEED, device='cuda')
        kan.fit(dataset, opt='LBFGS', steps=steps, lamb=lamb, lamb_entropy=lamb_ent)
        pred = kan(dataset["test_input"]).cpu().detach().numpy()
        err = mean_squared_error(dataset["test_label"].cpu().detach().numpy(), pred) if not np.any(np.isnan(pred)) else float('nan')
        return err
    return objective
    
    
if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(load_data(), n_trials=500)
    print(study.best_params)