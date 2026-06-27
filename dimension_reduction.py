from import_dataset import open_hopv15_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    symbol_to_num = ['H','He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne','Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                     'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                     'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    molecules_with_conformers = {}
    atom_coordinates = []
    for molecule in data:
        molecules_with_conformers.update({molecule.experimental_data.InChi_key: {}})
        for conformer in molecule.conformer_info:
            molecules_with_conformers[molecule.experimental_data.InChi_key].update({conformer.conformer_name: []})
            for atom in conformer.atom_coordinates:
                formatted_chemical = [symbol_to_num.index(atom.chemical_id)+1, atom.x, atom.y, atom.z]
                atom_coordinates.append(formatted_chemical)
                molecules_with_conformers[molecule.experimental_data.InChi_key][conformer.conformer_name].append(formatted_chemical)
    atom_coordinates = np.array(atom_coordinates)
    export_atoms = pd.DataFrame(atom_coordinates, columns = ['AtomicSymbol', 'X', 'Y', 'Z'])
    export_atoms.to_csv("./atoms.tsv", "\t", index=False)
    print("Completed the Data Conversion!")
    formatted_data = StandardScaler().fit_transform(atom_coordinates)
    reducer = PCA()
    fit_result = reducer.fit_transform(formatted_data)
    print(reducer.explained_variance_ratio_)
    print(fit_result[:20])
    x_vals = np.ones(fit_result.shape)
    plt.scatter(fit_result[:, 0], fit_result[:, 1])
    plt.show()