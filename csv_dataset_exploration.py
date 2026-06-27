from import_dataset import open_csv_dataset, get_atom_dist
from rdkit import Chem
import numpy as np

if __name__ == "__main__":
    data = open_csv_dataset("./OPV_Datasets/opv_db.csv")
    print(data["basis"].unique())
    # chems = [Chem.AddHs(Chem.MolFromSmiles(molecule)) for molecule in data["smile"]]
    # atom_dist = get_atom_dist(chems)
    # total_atoms = np.sum(list(atom_dist.values()))
    # for key in atom_dist:
    #     print(f"% {key} atoms out of {total_atoms} total atoms: {(atom_dist[key]/total_atoms)*100:.2f}%")