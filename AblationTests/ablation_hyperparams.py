"""
Jacob Maurer
9/4/2026
Purpose: File to run all hyperparameter tunes for the ablation study.
"""
import optuna
from rbf_ablation import load_class_rbf, load_regress_rbf
from wavelet_ablation import load_class_wavelet, load_regress_wavelet
from cheby_ablation import load_class_cheby, load_regress_cheby
import torch

if __name__ == "__main__":
    torch.manual_seed(42)
    study = optuna.create_study(
        storage="sqlite:///db_ablation.sqlite3",
        study_name="RBFClassProper",
        direction="minimize")
    study.optimize(load_class_rbf(), 500)
    #study2 = optuna.create_study(
    #    storage="sqlite:///db_ablation.sqlite3",
    #    study_name="RBFRegress",
    #    direction="minimize")
    #study2.optimize(load_regress_rbf(), 500)
    study3 = optuna.create_study(
        storage="sqlite:///db_ablation.sqlite3",
        study_name="WaveletClassProper",
        direction="minimize")
    study3.optimize(load_class_wavelet(), 500)
    #study4 = optuna.create_study(
    #    storage="sqlite:///db_ablation.sqlite3",
    #    study_name="WaveletRegress",
    #    direction="minimize")
    #study4.optimize(load_regress_wavelet(), 500)
    study5 = optuna.create_study(
        storage="sqlite:///db_ablation.sqlite3",
        study_name="ChebyClassProper",
        direction="minimize")
    study5.optimize(load_class_cheby(), 500)
    #study6 = optuna.create_study(
    #    storage="sqlite:///db_ablation.sqlite3",
    #    study_name="ChebyRegress",
    #    direction="minimize")
    #study6.optimize(load_regress_cheby(), 500)