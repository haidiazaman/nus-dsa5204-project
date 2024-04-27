import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


# POSITIONAL ENCODING FOR BOTH MAE AND NO MAE TRANSFORMER ARCHITECTURES
# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

# MODEL WITH MAE
# Model definition using Transformer
class MAETransformerModel(nn.Module):
    def __init__(self, masking_ratio=0.5, num_features=1, d_model=64, dropout=0.2, nhead=4, dim_feedforward=2048,num_layers=2):
        super(MAETransformerModel, self).__init__()

        self.masking_ratio = masking_ratio

        self.feature_projection = nn.Linear(num_features, d_model) # projects the input_num_features to d_model reqd for pos encoding
        self.positional_encoder = PositionalEncoding(d_model, dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_layers)


        transformer_decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_layers)

        self.layer_norm1 = nn.LayerNorm(d_model)

        self.reverse_projection = nn.Linear(d_model,num_features) # projects the d_model to input_num_features to reconstruct the input X

    def get_latent_vector(self, x):
        # input x is in shape (batch_size,seq_length,num_features)

        features = self.feature_projection(x) # Token generation we generate a token for every input patch (by linear projection with an added positional embedding).
        tokens = self.positional_encoder(features) # 512,100,63

        # Add input x to the first col of the positional encoding containing the original time series to maintain the original time series information
        # Extract the first slice of tokens (this will be of shape (512,100)) and reshape it to match the shape of input x (512,100,1)
        tokens_first_dim = tokens[:, :, 0].unsqueeze(-1)
        # Add input x to the first slice of tokens_first_dim
        tokens_first_dim += x

        # Extract the remaining slices of tokens
        tokens_rem = tokens[:, :, 1:]
        # Concatenate b_0th and b_rem along the last dimension
        tokens = torch.cat((tokens_first_dim, tokens_rem), dim=-1)

        # Random shuffling and masking
        shuffled_indices = torch.randperm(tokens.size(1)) # 1 as that is the sequence dimension
        num_unmasked = tokens.size(1)-int(self.masking_ratio * tokens.size(1))
        unmasked_indices = shuffled_indices[:num_unmasked]
        unmasked_tokens = tokens[:,unmasked_indices,:]

        # pass unmasked tokens to encoder
        encoded_tokens = self.transformer_encoder(unmasked_tokens)

        # get the masked tokens and mask them (set to zero)
        masked_indices = shuffled_indices[num_unmasked:]
        masked_tokens = tokens[:,masked_indices,:]
        masked_tokens[:,:,0] = 0. # only mask (set to 0) the first row

        # Appending mask tokens
        full_tokens = torch.cat((encoded_tokens, masked_tokens), dim=1)

        # Unshuffling and reconstructing the full vector
        unshuffled_indices = torch.argsort(shuffled_indices)
        latent_vector = full_tokens[:,unshuffled_indices,:] # unshuffled_full_tokens

        return latent_vector,masked_indices

    def forward(self, x):

        latent_vector,masked_indices = self.get_latent_vector(x)

        # pass to decoder
        x = self.transformer_decoder(latent_vector)
        x = self.layer_norm1(x)
        reconstructed = self.reverse_projection(x)
        # reconstructed = x[:,:,:1] # since we encoded the original time series information to the first col of the postional encoding vector

        # need to return the reconstructed and the masked indices, since the loss funciton should only calculate the loss on the masked patches
        return reconstructed,masked_indices
    

    
# MODEL WITH NO MAE
# Model definition using Transformer
class TransformerModel(nn.Module):
    def __init__(self, num_features=1, d_model=64, dropout=0.1, nhead=4, dim_feedforward=2048,num_layers=2):
        super(TransformerModel, self).__init__()

        self.feature_projection = nn.Linear(num_features, d_model) # projects the input_num_features to d_model reqd for pos encoding
        self.positional_encoder = PositionalEncoding(d_model, dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_layers)


        transformer_decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_layers)

        self.layer_norm1 = nn.LayerNorm(d_model)

        self.reverse_projection = nn.Linear(d_model,num_features) # projects the d_model to input_num_features to reconstruct the input X

    def forward(self, x):
        x = self.feature_projection(x)
        x = self.positional_encoder(x)
        latent_vector = self.transformer_encoder(x)
        x = self.transformer_decoder(latent_vector)
        x = self.layer_norm1(x)
        reconstructed = self.reverse_projection(x)

        return reconstructed


    def get_latent_vector(self, x):
        x = self.feature_projection(x)
        x = self.positional_encoder(x)
        latent_vector = self.transformer_encoder(x)

        return latent_vector
    
    
    
# FINETUNING MODEL AFTER PRETRAINING
class FinetuningTransformerModel(nn.Module):
    def __init__(self, d_model=64, num_features=1, seq_len=100, forecasting_window=1):
        super(FinetuningTransformerModel, self).__init__()

        self.seq_len = seq_len
        self.linear1 = nn.Linear(d_model,num_features)
        self.layer_norm = nn.LayerNorm(seq_len)
        self.linear2 = nn.Linear(seq_len,1)


    def forward(self, x):
      # input x is the latent vector from the pretraining model latent vector output

        x = F.tanh(self.linear1(x))
        x = x.view(x.size(0), -1) # flatten
        x = self.layer_norm(x)
        # x = F.relu(self.linear1(x))
        x = self.linear2(x) #apply tanh if the values go beyond (-1,1)

        return x
    
# # sample inference
# model = MAETransformerModel()
# for X,y in train_loader:
#     reconstructed,masked_indices = model(X)
#     break

# print(X.shape,reconstructed.shape)
# nn.MSELoss()(X[:,masked_indices,:],reconstructed[:,masked_indices,:])