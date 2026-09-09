"""
Jacob Maurer
6/4/2026
Purpose: Retrian the best kan network found, then convert it to symbolic
"""
from kan import *
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import ImageOps
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from import_dataset import open_hopv15_dataset

#Try 20 steps

if __name__ == "__main__":
    harvard_data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    harvard_chems = np.array([Chem.AddHs(Chem.MolFromSmiles(molecule.smiles_molecule)) for molecule in harvard_data])
    harvard_labels = np.array([molecule.experimental_data.PCE for molecule in harvard_data])
    harvard_chems = harvard_chems[~np.isnan(harvard_labels)]
    harvard_labels = harvard_labels[~np.isnan(harvard_labels)]
    harvard_labels = (harvard_labels - np.mean(harvard_labels))/np.std(harvard_labels)
    harvard_labels = torch.from_numpy(harvard_labels).type(torch.float)
    imgs = np.array([np.array(ImageOps.grayscale(Draw.MolToImage(chem))).flatten() for chem in harvard_chems])
    scaler = StandardScaler()
    scaled_imgs = scaler.fit_transform(imgs)
    pca = PCA(n_components=0.9, random_state=42)
    final_imgs = torch.from_numpy(pca.fit_transform(scaled_imgs)).type(torch.float)
    dataset = create_dataset_from_data(final_imgs, harvard_labels, device='cuda')
    print(dataset)
    kan = KAN(width=[214,39,97,1], grid=6, k=3, seed=42, device='cuda')
    kan.fit(dataset, opt='LBFGS', steps=200, lamb=0.7331999323645108, lamb_entropy=0.19713733118202736)
    kan = kan.prune()
    print(f"Mean Squared Error: {mean_squared_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    print(f"Mean Absolute Error: {mean_absolute_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    print(f"Root Mean Squared Error {root_mean_squared_error(dataset['test_label'].cpu().detach().numpy(), kan(dataset['test_input']).cpu().detach().numpy())}")
    kan.auto_symbolic(verbose=0)
    print(kan.symbolic_formula()[0][0])
    plt.show()