"""
Jacob Maurer
6/2/2026
Purpose: Create an optuna optimizer for a KAN network for summation dataset
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
SYMBOL_TO_NUM = ['H','He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne','Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                     'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                     'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']

def load_data():
    global SYMBOL_TO_NUM
    harvard_data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    harvard_labels = []
    molecules_with_conformers = []
    for molecule in harvard_data:
        for conformer in molecule.conformer_info:
            temp = np.zeros((4))
            for atom in conformer.atom_coordinates:
                formatted_chemical = np.array([SYMBOL_TO_NUM.index(atom.chemical_id)+1, atom.x, atom.y, atom.z])
                temp += formatted_chemical
            temp = temp.tolist()
            extras = []
            for calc in conformer.calculated_data:
                extras.extend([calc.HOMO, calc.LUMO, calc.Gap])
            temp.extend(extras)
            molecules_with_conformers.append(temp)
            harvard_labels.append(molecule.experimental_data.PCE)
    molecules_with_conformers = np.array(molecules_with_conformers) 
    harvard_labels = np.array(harvard_labels)
    molecules_with_conformers = molecules_with_conformers[~np.isnan(harvard_labels)]
    harvard_labels = harvard_labels[~np.isnan(harvard_labels)]
    molecules_with_conformers = torch.from_numpy(molecules_with_conformers).type(torch.float)
    harvard_labels = torch.from_numpy(harvard_labels).type(torch.float)
    dataset = create_dataset_from_data(molecules_with_conformers, harvard_labels, device='cuda')
    def objective(trial):
        global RANDOM_SEED
        depth = trial.suggest_int("depth", 3, 10)
        layers = [len(dataset["train_input"][0])]
        for i in range(depth-2):
            layers.append(trial.suggest_int(f"layer_{i+1}", 5, 100))
        layers.append(1)
        grid = trial.suggest_int("grid_size", 3, 10)
        k = trial.suggest_int("k", 3, 8)
        steps = 100
        lamb = trial.suggest_float("lamb", 0.01, 0.99)
        lamb_ent = trial.suggest_float("lamb_entropy", 0.0, 10.0)
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