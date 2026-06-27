"""
Jacob Maurer
6/19/2026
Purpose: USe the Scharber pces to calculate the experimental values of PCE
"""
from math import isnan
from kan import *
from import_dataset import open_hopv15_dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    scharber_pces = []
    labels = []
    device = "cuda"
    for molecule in data:
        if isnan(molecule.experimental_data.PCE):
            continue
        for confomer in molecule.conformer_info:
            temp = []
            for calc in confomer.calculated_data:
               temp.append(calc.scharber_pce)
            scharber_pces.append(temp)
            labels.append(molecule.experimental_data.PCE/100.0)
    plt.hist(np.unique(labels))
    scaler = 
    dataset = create_dataset_from_data(torch.from_numpy(np.asarray(scharber_pces)).type(torch.float), torch.from_numpy(np.asarray(labels)).type(torch.float), 0.8, device)
    kan_model = KAN([4, 5, 5, 1], grid=20, k=5, seed=42, device=device)
    kan_model.fit(dataset, steps=300, lamb=0.1, lamb_entropy=3.51233)
    kan_model.plot()
    try:
        kan_model = kan_model.prune(0.01, 0.01)
    except:
        print("Cannot prune! not pruning...")
    kan_model.plot()
    kan_model.auto_symbolic()
    print(kan_model.symbolic_formula()[0][0])
    kan_model.plot()
    plt.show()
    
    