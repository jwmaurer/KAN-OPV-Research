"""
Lopez, S., Pyzer-Knapp, E., Simm, G. et al. The Harvard organic photovoltaic dataset. Sci Data 3, 160086 (2016). https://doi.org/10.1038/sdata.2016.86

The data set used, nicknamed HOPV15, can be found with the link above
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from dataclasses import dataclass
import polars as pl
import matplotlib.pyplot as plt

@dataclass
class ExperimentalInformation:
    DOI: str
    InChi_key: str
    construction: str
    architecture: str
    complement: str
    HOMO: float | str
    LUMO: float | str
    electorchemical_gap: float | str
    optical_gap: float | str
    PCE: float | str
    Voc: float | str
    Jsc: float | str
    fill_factor: float | str
    def has_nan(self) -> bool:
        check = (isinstance(self.HOMO, str) or isinstance(self.LUMO, str) or isinstance(self.electorchemical_gap, str)
        or isinstance(self.optical_gap, str) or isinstance(self.PCE, str) or isinstance(self.Voc, str) or isinstance(self.Jsc, str)
        or isinstance(self.fill_factor, str))
        if check:
            print(f"Nan value found in Experimental information of {self.DOI}")
        return check
    
@dataclass
class CalculatedInformation:
    basis_set_description: str
    HOMO: float | str
    LUMO: float | str
    Gap: float | str
    scharber_pce: float | str
    scharber_voc: float | str
    scharber_jsc: float | str
    def has_nan(self) -> bool:
        check = (isinstance(self.HOMO, str) or isinstance(self.LUMO, str) or isinstance(self.Gap, str) or 
                 isinstance(self.scharber_pce, str) or isinstance(self.scharber_voc, str) or isinstance(self.scharber_jsc, str))
        if check:
            print(f"Nan value found in the Calculated information with {self.basis_set_description}")
        return check

@dataclass
class ChemicalCoordinates:
    chemical_id: str
    x: float
    y: float
    z: float

@dataclass
class ConformerInformation:
    conformer_name: str
    number_atoms: int
    atom_coordinates: list[ChemicalCoordinates]
    calculated_data: list[CalculatedInformation]

@dataclass
class FormattedData:
    smiles_molecule: str
    InChl_molecule: str
    experimental_data: ExperimentalInformation
    pruned_smiles: str
    total_conformers: int
    conformer_info: list[ConformerInformation]
    
def convert_poss_nan(item: str)-> float | str:
    try:
        convert = float(item)
    except:
        convert = item
    return convert

def open_hopv15_dataset(fpath):
    imported_data: list[FormattedData] = []
    num_sections = 350
    with open(fpath, 'rt') as file:
        for i in range(num_sections):
            smiles = file.readline().strip()
            inchl = file.readline().strip()
            experimental_list = file.readline().strip().split(',')
            DOI_exp = experimental_list[0]
            inchl_exp = experimental_list[1]
            construct = experimental_list[2]
            architect = experimental_list[3]
            complement = experimental_list[4]
            homo = convert_poss_nan(experimental_list[5])
            lumo = convert_poss_nan(experimental_list[6])
            electrochem = convert_poss_nan(experimental_list[7])
            optical = convert_poss_nan(experimental_list[8])
            pce = convert_poss_nan(experimental_list[9])
            voc = convert_poss_nan(experimental_list[10])
            jsc = convert_poss_nan(experimental_list[11])
            fill_wid = convert_poss_nan(experimental_list[12])
            experiment = ExperimentalInformation(DOI_exp, inchl_exp, construct, architect, complement, homo, lumo, electrochem, optical, pce, voc, jsc, fill_wid)
            pruned_smiles = file.readline().strip()
            conformers_num = int(file.readline().strip())
            conformers = []
            for conformer in range(conformers_num):
                file.readline()
                num_atoms = int(file.readline().strip())
                chem_list = []
                for atom in range(num_atoms):
                    cur_line = file.readline().strip().split(' ')
                    chem_list.append(ChemicalCoordinates(cur_line[0], float(cur_line[1]), float(cur_line[2]), float(cur_line[3])))
                calc_data = []
                for calc in range(4):
                    new_line = file.readline().strip().split(',')
                    homo_calc = convert_poss_nan(new_line[1])
                    lumo_calc = convert_poss_nan(new_line[2])
                    gap_calc = convert_poss_nan(new_line[3])
                    pce_calc = convert_poss_nan(new_line[4])
                    voc_calc = convert_poss_nan(new_line[5])
                    jsc_calc = convert_poss_nan(new_line[6])
                    calc_data.append(CalculatedInformation(new_line[0], homo_calc, lumo_calc, gap_calc, pce_calc, voc_calc, jsc_calc))
                conformers.append(ConformerInformation(f'Conformer {conformer}', num_atoms, chem_list, calc_data))
            imported_data.append(FormattedData(smiles, inchl, experiment, pruned_smiles, conformers_num, conformers))
        return imported_data
    
def open_csv_dataset(fpath):
    return pl.read_csv(fpath) 

def open_image_data(fpath):
    low_variance_cols = np.ones((90000,), dtype=np.bool)
    line_num = 0
    with open(fpath, 'rt') as file:
        first_line = np.asarray(file.readline().strip().split(','), dtype=np.int16)
        for line in file:
            try:
                current = np.asarray(file.readline().split(','), dtype=np.int16)
                low_variance_cols = low_variance_cols & (current == first_line)
                print(f"Line: {line_num}")
                line_num += 1
            except: 
                return low_variance_cols
            

def mol_atom_dist(chemical: Mol)-> dict[str, int]:
    chemical_dist = {}
    for atom in chemical.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in list(chemical_dist.keys()):
            chemical_dist[symbol] += 1
        else:
            chemical_dist.update({symbol: 1})
    return chemical_dist

def get_atom_dist(chemicals: list[Mol]) -> dict[str, int]:
    chemical_dist: dict[str, int] = {}
    for chem in chemicals:
        for atom in chem.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in list(chemical_dist.keys()):
                chemical_dist[symbol] += 1
            else:
                chemical_dist.update({symbol: 1})
    return chemical_dist

def get_max_atoms(chemicals: list[Mol]) -> int:
    max_atoms = 0
    for chem in chemicals:
        if max_atoms < len(chem.GetAtoms()):
            max_atoms = len(chem.GetAtoms())
    return max_atoms

if __name__ == '__main__':
    print(open_image_data("./OPV_Datasets/molecule_images.csv")[10000:10100])
    # data = open_csv_dataset("OPV_Datasets/opv_db.csv").select("smile").to_numpy()
    # chem_data = [Chem.AddHs(Chem.MolFromSmiles(item[0])) for item in data]
    # print(get_atom_dist(chem_data))
    # data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    # print(len(data))
    # print([molecule.conformer_info for molecule in data if not molecule.experimental_data.has_nan()][:2])
    # chem_list = [Chem.AddHs(Chem.MolFromSmiles(chem.smiles_molecule)) for chem in data]
    # random_chem = chem_list[0]
    # print(get_atom_dist(chem_list))
    # print(mol_atom_dist(random_chem))
    # print(get_max_atoms(chem_list))
    # mols_with_elem: list[FormattedData] = []
    
    #Find the PCE vs. # of Atoms graph for molecules with a certain atom (ex. C)
    # atom = 'C'
    # for i, chem in enumerate(chem_list):
    #     atom_dist = mol_atom_dist(chem)
    #     if atom in list(atom_dist.keys()):
    #         mols_with_elem.append(data[i])
    # y = [item.experimental_data.PCE for item in mols_with_elem if not isinstance(item.experimental_data.PCE, str)]
    # x = [mol_atom_dist(Chem.AddHs(Chem.MolFromSmiles(item.smiles_molecule)))[atom] for item in mols_with_elem if not isinstance(item.experimental_data.PCE, str)]
    # plt.scatter(x, y)
    # plt.ylabel('PCE')
    # plt.xlabel(f'Number of Atoms of {atom}')
    # plt.show()