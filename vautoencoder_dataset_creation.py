"""
Jacob Maurer
6/10/2026
Purpose: Create an encoder dataset that compines opv_db.csv and HOPV_15 dataset.
"""
from import_dataset import open_csv_dataset, open_hopv15_dataset
import polars

#Features: HOMO, LUMO, Optical Gap, is_basis_1, is_basis_2, is_basis_3, is_basis_4, is_basis_5
#Additional Feature if this doesn't work: elemental composition of molecule

def encode_basis(basis: str):
    def inner(x):
        return 1 if x == basis else 0
    return inner

if __name__ == "__main__":
    harvard_data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    csv_data = open_csv_dataset("./OPV_Datasets/opv_db.csv")
    csv_data = csv_data.drop(["mol", "ctag", "total_energy", "optical_lumo", "spectral_overlap", "delta_homo", "delta_lumo", "delta_optical_lumo", "homo_extrapolated", "lumo_extrapolated", "gap_extrapolated", "optical_lumo_extrapolated", "smile"])
    formatted_harvard = []
    for molecule in harvard_data:
        for conformer in molecule.conformer_info:
            for calc in conformer.calculated_data:
                formatted_harvard.append([calc.basis_set_description, calc.Gap, calc.HOMO, calc.LUMO])
    harvard_frame = polars.DataFrame(formatted_harvard, {"basis":str, "gap": polars.Float64, "homo": polars.Float64, "lumo": polars.Float64})
    final_set = polars.concat([csv_data, harvard_frame])
    print(final_set["basis"].unique())
    for basis in final_set["basis"].unique():
        final_set = final_set.with_columns(final_set["basis"].map_elements(encode_basis(basis), return_dtype=polars.Int8).alias("is_"+basis))
    final_set = final_set.drop(["basis"])
    # final_set.write_csv("./combined_data.csv")
    