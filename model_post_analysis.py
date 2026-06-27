import pandas as pd

if __name__ == "__main__":
    hopv_weight_l1 = pd.read_csv("./hopv_model_weights_1.csv")
    hopv_bias_l1 = pd.read_csv("./hopv_model_bias_2.csv")
    hopv_weight_l2 = pd.read_csv("./hopv_model_weights_3.csv")
    hopv_bias_l2 = pd.read_csv("./hopv_model_bias_4.csv")
    columns = ["C", "S", "O", "H", "N", "Si", "F", "Se"]
    columns_2 = [f"Neuron_{i}" for i in range(128)]
    hopv_weight_l1.rename(columns={"0": "C", "1": "S", "2": "O", "3": "H", "4": "N", "5": "Si", "6": "F", "7": "Se"}, inplace=True)
    hopv_weight_l2.columns = columns_2
    print(hopv_weight_l1.corr())
    print()
    for col in columns:
        print(f"Sum of weights for {col}: {hopv_weight_l1[col].sum()}")
        print(f"Mean of weights for {col}: {hopv_weight_l1[col].mean()}")
        print(f"Median of weights for {col}: {hopv_weight_l1[col].median()}")
        print()
    print()
    print(hopv_weight_l2)    

