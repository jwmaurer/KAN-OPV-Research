"""
Jacob Maurer
8/17/2026
Purpose: Test the complement model for other metrics
"""
from sklearn.metrics import classification_report, f1_score
from kan import *
from sklearn.preprocessing import StandardScaler
from import_dataset import open_hopv15_dataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def encode_mat(input_mat):
    unique_vals = np.unique(input_mat)
    print(unique_vals)
    for index, key in enumerate(unique_vals):
        input_mat[input_mat == key] = index
    return input_mat

if __name__ == "__main__":
    device = "cuda"
    RANDOM_SEED = 42
    data = open_hopv15_dataset("D:\\kan_encoder_architecture\\HOPV_15_revised_2.data")
    usable_labels = []
    molecule_counter = 1
    for molecule in data:
        if np.isnan(molecule.experimental_data.HOMO) or np.isnan(molecule.experimental_data.LUMO) or np.isnan(molecule.experimental_data.optical_gap) or np.isnan(molecule.experimental_data.PCE) or molecule_counter in [96, 105, 159, 284, 307, 350]:
            molecule_counter += 1
            continue
        for conformer in molecule.conformer_info:
            for calc in conformer.calculated_data:
                usable_labels.append([molecule.experimental_data.PCE, molecule.experimental_data.complement])
        molecule_counter += 1
    usable_data = pd.read_csv("D:\\kan_encoder_architecture\\saved_harvard_theory.csv").to_numpy()
    usable_labels = np.asarray(usable_labels)
    usable_labels[:, 1] = encode_mat(usable_labels[:, 1]).astype(np.int16)
    usable_labels = usable_labels.astype(np.float64)
    test_size = 0.2
    train_x, test_x, train_y, test_y = train_test_split(usable_data, usable_labels, test_size=test_size, stratify=usable_labels[:, 1], shuffle=True, random_state=42)
    version = f"blank"
    scaler = StandardScaler()
    pca = PCA(n_components=0.9, random_state=RANDOM_SEED)
    comp_loss = nn.CrossEntropyLoss()
    encoded_x = pca.fit_transform(scaler.fit_transform(train_x))
    comp_kan = KAN([len(pca.components_), 5, 4], grid=27, k=6, ckpt_path=f"D:/KAN_files_class_regress_class/kan_{version}", device=device, seed=RANDOM_SEED)
    
    current_dataset_complement = create_dataset_from_data(torch.from_numpy(encoded_x).to(device).type(torch.float), torch.from_numpy(train_y[:, 1]).to(device).type(torch.long), train_ratio=0.9, device=device)
    
    comp_results = comp_kan.fit(current_dataset_complement, steps=57, lamb=0.0002685431942328297, loss_fn=comp_loss)
    
    try:
        comp_kan = comp_kan.prune()
    except:
        print("comp cannot be pruned!")
    else:
        try: 
            comp_kan.auto_symbolic(weight_simple=0.0, r2_threshold=0.85)
            comp_kan.fit(current_dataset_complement, steps=50, lamb=0.0002685431942328297, loss_fn=comp_loss)
            comp_kan.auto_symbolic(weight_simple=0.0, r2_threshold=0.85)
            comp_kan.fit(current_dataset_complement, steps=50, lamb=0.0002685431942328297, loss_fn=comp_loss)
            comp_kan.auto_symbolic(weight_simple=0.0, r2_threshold=0.85)
            comp_kan.fit(current_dataset_complement, steps=50, lamb=0.0002685431942328297, loss_fn=comp_loss)
            current_dataset_complement = create_dataset_from_data(torch.from_numpy(pca.transform(scaler.transform(test_x))).to(device).type(torch.float), torch.from_numpy(test_y[:, 1]).to(device).type(torch.long), train_ratio=0.0, device=device)
            full_guess_comp = np.argmax(comp_kan(current_dataset_complement['test_input']).type(torch.long).detach().cpu().numpy(), axis=1)
            full_true_comp = current_dataset_complement['test_label'].type(torch.long).detach().cpu().numpy()
            print(comp_loss(comp_kan(current_dataset_complement['test_input']).type(torch.long).detach(), current_dataset_complement['test_label'].type(torch.long).detach()))
            print(classification_report(full_true_comp, full_guess_comp))
        except Exception as e:
            print(f"Error has occured! {e}")