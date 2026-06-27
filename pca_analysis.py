from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    data = pd.read_csv('./atoms.tsv',delimiter='\t').iloc[:10000]
    symbol_to_num = ['H','He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne','Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                     'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                     'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
    symbols = data["AtomicSymbol"]
    data["AtomicSymbol"] = data["AtomicSymbol"].map(lambda x: symbol_to_num.index(x)+1)
    new_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    transformed_data = pd.DataFrame(pca.fit_transform(new_data), columns=["PCA 1", "PCA 2"])
    transformed_data["AtomicSymbol"] = symbols
    sns.scatterplot(data=transformed_data, x='PCA 1', y='PCA 2', hue='AtomicSymbol')
    plt.show()
    
    
    