from import_dataset import open_hopv15_dataset
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
                formatted_chemical = np.array([atom.chemical_id, float(atom.x), float(atom.y), float(atom.z)])
                atom_coordinates.append(formatted_chemical)
                molecules_with_conformers[molecule.experimental_data.InChi_key][conformer.conformer_name].append(formatted_chemical)
    atom_coordinates = np.array(atom_coordinates)
    print(atom_coordinates.shape)
    print("Data set loaded in! making plot...")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs, ys, zs = [], [], []
    for row in atom_coordinates[:117]:
        xs.append(float(row[1]))
        ys.append(float(row[2]))
        zs.append(float(row[3]))
    ax.scatter(xs, ys, zs)
    print("Plot created!")
    plt.show()