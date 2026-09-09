from import_dataset import open_csv_dataset
from molecule_autoencoder import create_autoencoder, create_layers, MoleculeAutoEncoderTanh, train, test, save_model, CoordinateDataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

if __name__ == "__main__":
    device = "cuda"
    dataset = open_csv_dataset("./combined_data.csv")
    numpy_set = dataset.to_numpy()
    test_size = 0.2
    train_set, test_set = numpy_set[int(test_size*len(numpy_set)):], numpy_set[:int(test_size*len(numpy_set))]
    print(train_set.shape)
    print(test_set.shape)
    train_loader = DataLoader(CoordinateDataset(np.asarray(train_set)), batch_size=10000, shuffle=True)
    test_loader = DataLoader(CoordinateDataset(np.asarray(test_set)), batch_size=10000, shuffle=True)
    layer_strs = create_autoencoder(9, 7, 2, ["relu"], (200, 400))
    encoder_layers= create_layers(layer_strs[0], {"relu": nn.ReLU(), "silu": nn.SiLU()})
    decoder_layers= create_layers(layer_strs[1], {"relu": nn.ReLU(), "silu": nn.SiLU()})
    print(f"Encoder: {layer_strs[0]}")
    print(f"Decoder: {layer_strs[1]}")
    model = MoleculeAutoEncoderTanh(encoder_layers, decoder_layers).to(device)
    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    for i in range(epochs):
        train_loss = train(train_loader, model, loss, optimizer)
        test_loss = test(test_loader, model, loss)
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_loader):
            out = model(x)
            print(loss(y, out).item())
            print(out)
    save_model_bool = input("Save (y or n)?: ")
    if save_model_bool == "y":
        print("Saving model...")
        save_model(model, "./combined_model/", "combined_data_autoencoder",train_loss,test_loss, layer_strs[0], layer_strs[1])
    