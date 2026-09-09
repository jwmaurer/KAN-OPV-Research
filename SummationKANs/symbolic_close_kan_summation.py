"""
Jacob Maurer
6/5/2026
Purpose: Forgot to copy over the best params (will retrain to find the best after my computer has cooled down)
"""
from kan import *
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import ImageOps
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from import_dataset import open_hopv15_dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    SYMBOL_TO_NUM = ['H','He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne','Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                     'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                     'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
    harvard_data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    harvard_labels = []
    molecules_with_conformers = []
    for molecule in harvard_data:
        for conformer in molecule.conformer_info:
            temp = np.zeros((3))
            for atom in conformer.atom_coordinates:
                formatted_chemical = np.array([atom.x, atom.y, atom.z])
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
    harvard_labels = (harvard_labels - np.mean(harvard_labels))/np.std(harvard_labels)
    molecules_with_conformers = torch.from_numpy(molecules_with_conformers).type(torch.float)
    harvard_labels = torch.from_numpy(harvard_labels).type(torch.float)
    dataset = create_dataset_from_data(molecules_with_conformers, harvard_labels, device='cuda')
    kan = KAN(width=[15,5,1], grid=20, k=6, seed=42, device='cuda')
    kan.fit(dataset, opt='LBFGS', steps=75, lamb=0.08300226189851313, lamb_entropy=1.1371926643620405)
    kan = kan.prune()
    kan(dataset['test_input'])
    kan.plot()
    print(f"Test Error: {mean_squared_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    print(f"Mean Absolute Error: {mean_absolute_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    print(f"Root Mean Squared Error {root_mean_squared_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    kan.auto_symbolic()
    # kan.fit(dataset, steps=20)
    print(kan.symbolic_formula()[0][0])
    # print(f"Test Error: {mean_squared_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    # print(f"Mean Absolute Error: {mean_absolute_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    # print(f"Root Mean Squared Error {root_mean_squared_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    # kan.plot()
    plt.show()