import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from ..model.imputation_model import MAETransformerModel, HyperImpute
from ..dataset.imputation_dataset import get_housing_dataset

def preprocessing(path, encoder, scaler, missing_rate, batch_size):
    housing_data = get_housing_dataset(path)

    housing_cat = housing_data[["ocean_proximity"]]
    housing_cat_encoded = encoder.fit_transform(housing_cat)
    housing_encoded = pd.DataFrame(housing_cat_encoded, columns=encoder.get_feature_names_out(['ocean_proximity']))
    housing_encoded.index = housing_data.index
    housing_data = pd.concat([housing_data, housing_encoded], axis=1).drop('ocean_proximity', axis=1)

    data_scaled = scaler.fit_transform(housing_data.values)
    data_scaled = scaler.transform(data_scaled)

    data_scaled = torch.tensor(data_scaled, dtype=torch.float32)

    train_scaled, _ = train_test_split(data_scaled, test_size=0.2, random_state=42)

    # Mask the data: 0 where data is missing
    mask_train = torch.bernoulli(torch.full((train_scaled.shape[0], train_scaled.shape[1]), missing_rate))  # 80% data presence
    train_input = train_scaled * mask_train

    train_dataset = TensorDataset(train_input, train_scaled) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, housing_data.shape[1]


# Train model function
def train_mae(path, enc, scl, missing_rate, batch_size, masking_ratio, lr, epochs, model_path):
    if enc == "OneHot":
        encoder = OneHotEncoder(sparse=False)
    if enc == "Ordinal":
        encoder = OrdinalEncoder()
    if scl == "MinMax":
        scaler = MinMaxScaler()
    if scl == "Standard":
        scaler = StandardScaler()

    train_loader, input_dim = preprocessing(path, encoder, scaler, missing_rate, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MAETransformerModel(num_features=input_dim, masking_ratio=masking_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initialize a list to store the average loss per epoch
    epoch_losses = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()

        train_running_loss = []
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            reconstructed, masked_indices = model(inputs)
            # Calculate loss only on masked indices
            loss = criterion(reconstructed[:, masked_indices], inputs[:, masked_indices])
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_running_loss)
        epoch_losses.append(train_loss)
        time_taken = round(time.time() - epoch_start_time, 1)
        print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss:.4f}, time_taken: {time_taken}")

    torch.save(model, model_path)

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), epoch_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

def train_baseline(path, enc, scl, missing_rate, batch_size, lr, epochs, model_path):
    if enc == "OneHot":
        encoder = OneHotEncoder(sparse=False)
    if enc == "Ordinal":
        encoder = OrdinalEncoder()
    if scl == "MinMax":
        scaler = MinMaxScaler()
    if scl == "Standard":
        scaler = StandardScaler()

    train_loader, input_dim = preprocessing(path, encoder, scaler, missing_rate, batch_size)
    model = HyperImpute(input_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    # Initialize a list to store the average loss per epoch
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Use inputs as targets
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)  # Append the average loss
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    torch.save(model, model_path)

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), epoch_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='mae', choices=['mae','baseline'])
    parser.add_argument('--path', default='housing.csv')
    parser.add_argument('--enc', default='OneHot', choices=['OneHot','Ordinal'])
    parser.add_argument('--scl', default='MinMax', choices=['MinMax','Standard'])
    parser.add_argument('--missing_rate', type=float, default=0.8, help='missing rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='masking ratio')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='num epochs for training')
    parser.add_argument('--model_path', default='trained_mae_imputation.pth', choices=['trained_mae_imputation.pth', 'trained_baseline_imputation.pth'])

    opt = parser.parse_args()

    if opt.model_type == 'mae':
        train_mae(opt.path, opt.enc, opt.scl, opt.missing_rate, opt.batch_size, opt.mask_ratio, opt.lr, opt.epochs, opt.model_path)
        
    if opt.model_type == 'baseline':
        train_baseline(opt.path, opt.enc, opt.scl, opt.missing_rate, opt.batch_size, opt.lr, opt.epochs, opt.model_path)