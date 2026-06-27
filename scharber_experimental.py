"""
Jacob Maurer
6/10/2026
Purpose: Could we use the Scharber Model on Experimental Data and get somewhat accurate results?
If the results are accurate, this could mean we train an encoder on these values, then accurately
calculate the PCE. A more accurate encoder leads to more accurate results 
"""
from import_dataset import open_hopv15_dataset
import numpy as np

def calculate_pce(jsc, voc):
    return ((jsc * voc * 53.603888888888875/100)/(0.985641822292957))

if __name__ == "__main__":
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    # for molecule in data:
    #     print((molecule.conformer_info[0].calculated_data[0].scharber_jsc * molecule.conformer_info[0].calculated_data[0].scharber_voc *.65 )/(molecule.conformer_info[0].calculated_data[0].scharber_pce))
    total_power = 0.0
    total_fill_factor = 0.0
    valid_mols = 0
    for molecule in data:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.PCE) or np.isnan(molecule.experimental_data.Jsc) or np.isnan(molecule.experimental_data.Voc) or np.isnan(molecule.experimental_data.fill_factor):
            continue
        truth = molecule.experimental_data.PCE
        calced = calculate_pce(molecule.experimental_data.Jsc, molecule.experimental_data.Voc)
        total_power += (molecule.experimental_data.Jsc * molecule.experimental_data.Voc * molecule.experimental_data.fill_factor/100.0)/molecule.experimental_data.PCE
        total_fill_factor += molecule.experimental_data.fill_factor
        print(f"True PCE: {truth} Calculated PCE: {calced:.4f} Difference {truth - calced}")
        valid_mols += 1
    print(f"Avg. Power: {total_power/valid_mols}")
    print(f"Avg. Fill Factor: {total_fill_factor/valid_mols}")