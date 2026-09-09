"""
Jacob Maurer
9/4/2026
Purpose: Hyperparameter tune for wavelet kans
"""
from wavlet import KANLinear
import optuna
from test_file import open_hopv15_dataset, train, test
from utils import make_class_dataset, make_regress_dataset, ClassDataset, RegressDataset
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error, r2_score
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np 

DEVICE = "cuda"
class ClassAblatWaveletKAN(nn.Module):
    def __init__(self, wavelet_type):
        super().__init__()
        self.model = nn.Sequential(
            KANLinear(4, 5, wavelet_type),
            KANLinear(5, 4, wavelet_type)
        )
    def forward(self, x):
        return self.model(x)

class RegressAblatWaveletKAN(nn.Module):
    def __init__(self, wavelet_type):
        super().__init__()
        self.model = nn.Sequential(
            KANLinear(4, 11, wavelet_type),
            KANLinear(11, 1, wavelet_type)
        )
    def forward(self, x):
        return self.model(x)    

def load_class_wavelet():
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    class_train_x, class_train_y, class_valid_x, class_valid_y, class_test_x, class_test_y = make_class_dataset(data, 42)
    train_loader = DataLoader(ClassDataset(class_train_x, class_train_y), batch_size=100, shuffle=True)
    valid_loader = DataLoader(ClassDataset(class_valid_x, class_valid_y), batch_size=100, shuffle=True)
    test_loader = DataLoader(ClassDataset(class_test_x, class_test_y), batch_size=100, shuffle=True)
    class_loss = nn.CrossEntropyLoss()
    string_to_optimizer = {
        "adam": torch.optim.Adam,
        "lbfgs": torch.optim.LBFGS
    }
    def objective(trial):
        train_settings = {
            "epochs": trial.suggest_int("epochs", 20, 100),
            "wavelet_type": trial.suggest_categorical("wavelet_type",["mexican_hat", "morlet", "dog", "meyer", "shannon"]),
            "lr": trial.suggest_float("lr", 0.0001, 0.3),
        }
        class_model = ClassAblatWaveletKAN(train_settings["wavelet_type"]).to(DEVICE)
        optimizer = string_to_optimizer["adam"](class_model.parameters(), lr=train_settings["lr"])
        for i in range(train_settings["epochs"]):
            train(train_loader, class_model, class_loss, optimizer, DEVICE)
            test(valid_loader, class_model, class_loss, DEVICE)
        return test(test_loader, class_model, class_loss, DEVICE)
    return objective

def load_regress_wavelet():
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    regress_train_x, regress_train_y, regress_valid_x, regress_valid_y, regress_test_x, regress_test_y = make_regress_dataset(data, 42)
    regress_train = nn.MSELoss()
    train_loader = DataLoader(RegressDataset(regress_train_x, regress_train_y), batch_size=100, shuffle=True)
    valid_loader = DataLoader(RegressDataset(regress_valid_x, regress_valid_y), batch_size=100, shuffle=True)
    test_loader = DataLoader(RegressDataset(regress_test_x, regress_test_y), batch_size=10000, shuffle=True)
    string_to_optimizer = {
        "adam": torch.optim.Adam,
        "lbfgs": torch.optim.LBFGS
    }
    def objective(trial):
        train_settings = {
            "epochs": trial.suggest_int("epochs", 20, 100),
            "wavelet_type": trial.suggest_categorical("wavelet_type",["mexican_hat", "morlet", "dog", "meyer", "shannon"]),
            "lr": trial.suggest_float("lr", 0.0001, 0.3)
        }
        regress_model = RegressAblatWaveletKAN(train_settings["wavelet_type"]).to(DEVICE)
        optimizer = string_to_optimizer["adam"](regress_model.parameters(), lr=train_settings["lr"])
        for i in range(train_settings["epochs"]):
            train(train_loader, regress_model, regress_train, optimizer, DEVICE)
            test(valid_loader, regress_model, regress_train, DEVICE)
        for x, y in test_loader:
            try:
                pred = regress_model(x).cpu().detach().numpy()
                return root_mean_squared_error(y.cpu().detach().numpy(), pred) - r2_score(y.cpu().detach().numpy(), pred)
            except:
                return np.nan
    return objective
    

if __name__ == "__main__":
    torch.manual_seed(42)
    study = optuna.create_study(
        storage="sqlite:///db_ablation.sqlite3",
        study_name="RBFClass",
        direction="minimize")
    study.optimize(load_class_wavelet(), 500)
    study2 = optuna.create_study(
        storage="sqlite:///db_ablation.sqlite3",
        study_name="RBFRegress",
        direction="minimize")
    study2.optimize(load_regress_wavelet(), 500)