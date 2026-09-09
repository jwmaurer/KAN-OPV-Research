"""
JAcob Maurer
9/8/2026
Purpose: classification report on the ablation study elements
"""
from sklearn.metrics import classification_report

from utils import make_class_dataset, ClassDataset
from import_dataset import open_hopv15_dataset
from rbf_ablation import ClassAblatRBFKAN
from wavelet_ablation import ClassAblatWaveletKAN
from cheby_ablation import ClassAblatChebyKAN
from torch.utils.data import DataLoader
from test_file import train, test
import torch.nn as nn
import torch
import numpy as np
device = "cuda"

if __name__ == "__main__":
    data = open_hopv15_dataset("./HOPV_15_revised_2.data")
    class_train_x, class_train_y, class_valid_x, class_valid_y, class_test_x, class_test_y = make_class_dataset(data, 42)
    train_loader = DataLoader(ClassDataset(class_train_x, class_train_y), batch_size=100, shuffle=True)
    valid_loader = DataLoader(ClassDataset(class_valid_x, class_valid_y), batch_size=100, shuffle=True)
    test_loader = DataLoader(ClassDataset(class_test_x, class_test_y), batch_size=10000, shuffle=True)
    rbf = ClassAblatRBFKAN(76, 1.0001115115417882).to(device)
    wavelet = ClassAblatWaveletKAN('mexican_hat').to(device)
    cheby = ClassAblatChebyKAN(24).to(device)
    rbf_loss = nn.CrossEntropyLoss()
    wav_loss = nn.CrossEntropyLoss()
    cheby_loss = nn.CrossEntropyLoss()
    rbf_opt = torch.optim.Adam(rbf.parameters(), lr=0.027842979205433393)
    wavelet_opt = torch.optim.Adam(wavelet.parameters(), lr=0.027842979205433393)
    cheby_opt = torch.optim.Adam(cheby.parameters(), lr=0.00692195365119616)
    for i in range(97):
        print(i)
        train(train_loader, rbf, rbf_loss, rbf_opt, device)
        test(valid_loader, rbf, rbf_loss, device)
        if i < 92:
            train(train_loader, wavelet, wav_loss, wavelet_opt, device)
            test(valid_loader, wavelet, wav_loss, device)
        if i < 76:
            train(train_loader, cheby, cheby_loss, cheby_opt, device)
            test(valid_loader, cheby, cheby_loss, device)
    for X, y in test_loader:
        rbf_guess = np.argmax(rbf(X).detach().cpu().numpy(), axis=1)
        wavelet_guess = np.argmax(wavelet(X).detach().cpu().numpy(), axis=1)
        cheby_guess = np.argmax(cheby(X).detach().cpu().numpy(), axis=1)
        true_val = y.detach().cpu().numpy()
        print(classification_report(true_val, rbf_guess))
        print(classification_report(true_val, wavelet_guess))
        print(classification_report(true_val, cheby_guess))
        