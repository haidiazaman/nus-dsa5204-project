import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time_serie_model import *
from time_serie_dataset import *
import matplotlib.pyplot as plt

def preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name):

    train_size,val_size,test_size = eval(train_val_test_split)

    #########################
    # DATASET PREPROCESSING
    #########################
    data = pd.read_csv(csvPath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = data[[date_col_name,value_col_name]]
    data[date_col_name] = pd.to_datetime(data[date_col_name])
    plt.plot(data[date_col_name], data[value_col_name])
    plt.show()


    shifted_df = prepare_dataframe_for_lstm(data, lookback, date_col_name, value_col_name)
    shifted_df

    # format X and y from df and scale it
    shifted_df_as_np = shifted_df.to_numpy()
    print(shifted_df_as_np.shape)

    # normalise the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    print(X.shape, y.shape)
    X = dc(np.flip(X, axis=1))

    # train test split
    X_test = X[int(len(X) * train_size)+int(len(X) * val_size):]
    y_test = y[int(len(y) * train_size)+int(len(y) * val_size):]
    print(X_test.shape)
    print(y_test.shape)

    X_test = X_test.reshape((-1, lookback, 1))
    y_test = y_test.reshape((-1, 1))
    print(X_test.shape)
    print(y_test.shape)

    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    print(X_test.shape)
    print(y_test.shape)

    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # print to check
    for _, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    return X_test,y_test


def evaluate_pretrain_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs):
    
    X_test,y_test = preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name)

    ############################
    # DEFINE MODEL AND EVALUATE
    ############################
    
    best_model_path = 'MAE_pretraining_best_model.pt'
    # best_model_path = f'/content/drive/MyDrive/uni acads/MASTERS/y1s2/dsa5204/weights/{best_model_path}'


    # Load the best model for evaluation
    model = MAETransformerModel().to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()


    reconstructed,masked_indices = model(X_test.to(device))

    print('only on masked indices')
    reconstruction_MSE_loss = nn.MSELoss()(reconstructed.cpu().detach()[:,masked_indices,:],X_test.cpu().detach()[:,masked_indices,:])
    reconstruction_MAE_loss = nn.L1Loss()(reconstructed.cpu().detach()[:,masked_indices,:],X_test.cpu().detach()[:,masked_indices,:])
    print(f'reconstruction_MSE_loss: {reconstruction_MSE_loss}')
    print(f'reconstruction_MAE_loss: {reconstruction_MAE_loss}')

    print()
    print('on full indices')
    reconstruction_MSE_loss = nn.MSELoss()(reconstructed.cpu().detach(),X_test.cpu().detach())
    reconstruction_MAE_loss = nn.L1Loss()(reconstructed.cpu().detach(),X_test.cpu().detach())
    print(f'reconstruction_MSE_loss: {reconstruction_MSE_loss}')
    print(f'reconstruction_MAE_loss: {reconstruction_MAE_loss}')

    # plot reconstruction

    indices = [0,10,210,300]
    reconstructed = reconstructed.cpu().detach()
    X_test = X_test.cpu().detach()

    for index in indices:
        fig,ax = plt.subplots(1,3,figsize=(15,5))
        reconstructed_ = reconstructed[index]
        X_test_ = X_test[index]

        plt.subplot(1,3,1)
        plt.plot(X_test_,label='original')
        plt.plot(reconstructed_,label='pred')
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(X_test_,label='original')
        plt.legend()

        plt.subplot(1,3,3)
        plt.plot(reconstructed_,label='pred')
        plt.legend()

        plt.show()
        
def evaluate_finetune_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs):
    
    X_test,y_test = preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name)

    ############################
    # DEFINE MODEL AND EVALUATE
    ############################
    
    
    # load best pretraining model
    best_model_path = 'MAE_pretraining_best_model.pt'
    # best_model_path = f'/content/drive/MyDrive/uni acads/MASTERS/y1s2/dsa5204/weights/{best_model_path}'
    pretraining_model = MAETransformerModel(masking_ratio=0.).to(device)
    pretraining_model.load_state_dict(torch.load(best_model_path,map_location=device))
    pretraining_model.eval()

    # load best pretraining model
    best_model_path = 'finetuning_best_model_2.pt'
    # best_model_path = f'/content/drive/MyDrive/uni acads/MASTERS/y1s2/dsa5204/weights/{best_model_path}'
    finetuning_model = FinetuningTransformerModel().to(device)
    finetuning_model.load_state_dict(torch.load(best_model_path,map_location=device))
    finetuning_model.eval()

    latent_vector,_ = pretraining_model.get_latent_vector(X_test.to(device))
    predictions = finetuning_model(latent_vector)
    forecast_MSE_loss = nn.MSELoss()(predictions.cpu().detach(),y_test.cpu().detach())
    print(f'forecast_MSE_loss: {forecast_MSE_loss}')
    forecast_MAE_loss = nn.L1Loss()(predictions.cpu().detach(),y_test.cpu().detach())
    print(f'forecast_MAE_loss: {forecast_MAE_loss}')
    print(predictions.shape,y_test.shape)

    # plot forecasting prediction

    y = y_test.cpu().detach()
    pred = predictions.cpu().detach()

    plt.plot(y,label='original')
    plt.plot(pred,label='pred')
    plt.legend()

    plt.show()
    
    

def evaluate_pretrain_no_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs):
    
    X_test,y_test = preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name)

    ############################
    # DEFINE MODEL AND EVALUATE
    ############################
    
    best_model_path = 'pretraining_best_model.pt'
    # best_model_path = f'/content/drive/MyDrive/uni acads/MASTERS/y1s2/dsa5204/weights/{best_model_path}'


    # Load the best model for evaluation
    model = TransformerModel().to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()


    predictions = model(X_test.to(device))

    reconstruction_MSE_loss = nn.MSELoss()(predictions.cpu().detach(),X_test.cpu().detach())
    reconstruction_MAE_loss = nn.L1Loss()(predictions.cpu().detach(),X_test.cpu().detach())
    print(f'reconstruction_MSE_loss: {reconstruction_MSE_loss}')
    print(f'reconstruction_MAE_loss: {reconstruction_MAE_loss}')

    # plot reconstruction

    indices = [0,10,210,450]
    for index in indices:
        fig,ax = plt.subplots(1,3,figsize=(15,5))
        x = X_test.cpu().detach()[index]
        pred = predictions.cpu().detach()[index]

        plt.subplot(1,3,1)
        plt.plot(x,label='original')
        plt.plot(pred,label='pred')
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(x,label='original')
        plt.legend()

        plt.subplot(1,3,3)
        plt.plot(pred,label='pred')
        plt.legend()

        plt.show()


def evaluate_finetune_no_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs):
    
    X_test,y_test = preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name)

    ############################
    # DEFINE MODEL AND EVALUATE
    ############################
    
    # load best pretraining model
    best_model_path = 'no_MAE_pretraining_best_model_final.pt'
    # best_model_path = f'/content/drive/MyDrive/uni acads/MASTERS/y1s2/dsa5204/weights/{best_model_path}'
    pretraining_model = TransformerModel().to(device)
    pretraining_model.load_state_dict(torch.load(best_model_path,map_location=device))
    pretraining_model.eval()

    # load best pretraining model
    best_model_path = 'finetuning_best_model.pt'
    # best_model_path = f'/content/drive/MyDrive/uni acads/MASTERS/y1s2/dsa5204/weights/{best_model_path}'
    finetuning_model = FinetuningTransformerModel().to(device)
    finetuning_model.load_state_dict(torch.load(best_model_path,map_location=device))
    finetuning_model.eval()

    latent_vector = pretraining_model.get_latent_vector(X_test.to(device))
    predictions = finetuning_model(latent_vector)
    forecast_MSE_loss = nn.MSELoss()(predictions.cpu().detach(),y_test.cpu().detach())
    print(f'forecast_MSE_loss: {forecast_MSE_loss}')
    forecast_MAE_loss = nn.L1Loss()(predictions.cpu().detach(),y_test.cpu().detach())
    print(f'forecast_MAE_loss: {forecast_MAE_loss}')
    print(predictions.shape,y_test.shape)

    # plot forecasting prediction

    y = y_test.cpu().detach()
    pred = predictions.cpu().detach()

    plt.plot(y,label='original')
    plt.plot(pred,label='pred')
    plt.legend()

    plt.show()
    
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='pretrain', choices=['pretrain','finetune'])
    parser.add_argument('--mae', default='mae', choices=['mae','no mae'])
    parser.add_argument('--csvPath', default='ETTh1.csv', help='the directory of training data')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training')
    parser.add_argument('--train_val_test_split', default='[0.7,0.2,0.1]', help='specify train-val-test split here, pass a string of list of floats')
    parser.add_argument('--lookback', type=int, default=100, help='indicate intended lookback period aka lags window')
    parser.add_argument('--date_col_name', default='date', help='specify date col name in data')
    parser.add_argument('--value_col_name', default='OT', help='specify value col name in data')
    parser.add_argument('--epochs', type=int, default=1000, help='num epochs for training')
    
    opt = parser.parse_args()

    if opt.phase == 'pretrain' and opt.mae == 'mae':
        evaluate_pretrain_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)

    elif opt.phase == 'finetune' and opt.mae == 'mae':
        evaluate_finetune_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)
        
    elif opt.phase == 'pretrain' and opt.mae == 'no mae':
        evaluate_pretrain_no_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)

    elif opt.phase == 'finetune' and opt.mae == 'no mae':
        evaluate_finetune_no_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)