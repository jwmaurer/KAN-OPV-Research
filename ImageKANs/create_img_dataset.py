"""
Jacob Maurer
6/13/2026
Purpose: Create the images of the molecules, dropping the no variance columns and the highly correlated columns.
"""
import csv

import numpy as np
from import_dataset import open_csv_dataset, open_hopv15_dataset
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import ImageOps

if __name__ == "__main__":
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    csv_data = open_csv_dataset("./OPV_Datasets/opv_db.csv")['smile']
    smiles = [molecule.smiles_molecule for molecule in data if not np.isnan(molecule.experimental_data.PCE)]
    smiles.extend(csv_data.to_list())
    smiles = np.unique(smiles)
    with open("./OPV_Datasets/molecule_images.csv", 'at') as file:
        writer = csv.writer(file, delimiter=',')
        for i, smile in enumerate(smiles):
            print(f"Item {i}, Smile: {smile}")
            molecule = Chem.AddHs(Chem.MolFromSmiles(smile))
            image = np.asarray(ImageOps.grayscale(Draw.MolToImage(molecule))).flatten()
            writer.writerow(image)
        