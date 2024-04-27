import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
from scipy.stats import wasserstein_distance
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

    _, test_scaled = train_test_split(data_scaled, test_size=0.2, random_state=42)

    # Mask the data: 0 where data is missing
    mask_test = torch.bernoulli(torch.full((test_scaled.shape[0], test_scaled.shape[1]), missing_rate))  # 80% data presence
    test_input = test_scaled * mask_test

    test_dataset = TensorDataset(test_input, test_scaled) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_loader, housing_data.shape[1]


# Train model function
def eval_mae(path, enc, scl, missing_rate, batch_size, model_path):
    if enc == "OneHot":
        encoder = OneHotEncoder(sparse=False)
    if enc == "Ordinal":
        encoder = OrdinalEncoder()
    if scl == "MinMax":
        scaler = MinMaxScaler()
    if scl == "Standard":
        scaler = StandardScaler()

    test_loader, _ = preprocessing(path, encoder, scaler, missing_rate, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)

    model.eval()
    criterion = nn.MSELoss()
    total_mse_loss = 0.0
    total_wasserstein_distance = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs, _ = model(inputs)

            # Compute MSE loss
            mse_loss = criterion(outputs, targets)
            total_mse_loss += mse_loss.item()

            # Compute Wasserstein distance for each feature and average
            batch_wasserstein_distance = 0.0
            for i in range(inputs.shape[1]):  # Iterate over each feature
                batch_wasserstein_distance += wasserstein_distance(outputs[:, i].cpu().numpy(), targets[:, i].cpu().numpy())
            batch_wasserstein_distance /= inputs.shape[1]
            total_wasserstein_distance += batch_wasserstein_distance

            count += 1

    avg_mse_loss = total_mse_loss / count
    avg_wasserstein_distance = total_wasserstein_distance / count
    print(f"Avg MSE Loss: {avg_mse_loss:.4f}, Avg Wasserstein Distance: {avg_wasserstein_distance:.4f}")

def eval_baseline(path, enc, scl, missing_rate, batch_size, model_path):
    if enc == "OneHot":
        encoder = OneHotEncoder(sparse=False)
    if enc == "Ordinal":
        encoder = OrdinalEncoder()
    if scl == "MinMax":
        scaler = MinMaxScaler()
    if scl == "Standard":
        scaler = StandardScaler()

    test_loader, _ = preprocessing(path, encoder, scaler, missing_rate, batch_size)
    
    model = torch.load(model_path)

    criterion = nn.MSELoss()

    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_wasserstein_distance = 0.0
    n_batches = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in test_loader:
            outputs = model(inputs)
            print("targets vs outputs")
            print(targets, outputs)
            loss = criterion(outputs, targets)  # Use inputs as the target for loss calculation
            total_loss += loss.item()
            
            # Compute Wasserstein distance for each feature and average over all features
            wasserstein_distance_per_feature = [wasserstein_distance(outputs[:, i].cpu().numpy(), inputs[:, i].cpu().numpy()) 
                                                for i in range(inputs.shape[1])]
            total_wasserstein_distance += sum(wasserstein_distance_per_feature) / len(wasserstein_distance_per_feature)
            n_batches += 1
    
    average_loss = total_loss / n_batches
    avg_wasserstein_distance = total_wasserstein_distance / n_batches
    
    print(f"Avg MSE Loss: {average_loss:.4f}, Avg Wasserstein Distance: {avg_wasserstein_distance:.4f}")

# Example usage
# Note: You need your 'model', 'test_loader', and 'criterion' defined as per your setup
# average_loss, avg_wasserstein_distance = evaluate_model_with_wasserstein(model, test_loader, criterion)
# print(f'Average Loss: {average_loss:.4f}, Average Wasserstein Distance: {avg_wasserstein_distance:.4f}')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='mae', choices=['mae','baseline'])
    parser.add_argument('--path', default='housing.csv')
    parser.add_argument('--enc', default='OneHot', choices=['OneHot','Ordinal'])
    parser.add_argument('--scl', default='MinMax', choices=['MinMax','Standard'])
    parser.add_argument('--missing_rate', type=float, default=0.8, help='missing rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--model_path', default='trained_mae_imputation.pth', choices=['trained_mae_imputation.pth', 'trained_baseline_imputation.pth'])

    opt = parser.parse_args()

    if opt.model_type == 'mae':
        eval_mae(opt.path, opt.enc, opt.scl, opt.missing_rate, opt.batch_size, opt.model_path)
        
    if opt.model_type == 'baseline':
        eval_baseline(opt.path, opt.enc, opt.scl, opt.missing_rate, opt.batch_size, opt.model_path)