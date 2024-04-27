import numpy as np
import torch
import torch.nn as nn
import numpy as np

# MAE Model using Transformer
class MAETransformerModel(nn.Module):
    def __init__(self, num_features, masking_ratio=0.75, d_model=128, dropout=0.1, nhead=4, dim_feedforward=2048, num_layers=2):
        super(MAETransformerModel, self).__init__()

        self.masking_ratio = masking_ratio

        self.feature_projection = nn.Linear(num_features, d_model)  # projects input features to d_model dimensions
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=num_layers)
        self.reverse_projection = nn.Linear(d_model, num_features)  # projects d_model back to input feature size

    def forward(self, x):
        # Masking part
        batch_size, num_features = x.shape
        num_masked = int(self.masking_ratio * num_features)  # Number of features to mask
        all_indices = torch.arange(num_features)
        masked_indices = np.random.choice(all_indices, num_masked, replace=False)  # Randomly choose indices to mask

        mask = torch.ones(num_features, device=x.device)
        mask[masked_indices] = 0  # Set masked indices to zero

        # Apply mask
        x_masked = x * mask

        # Project features
        x_projected = self.feature_projection(x_masked)
        # Encode
        encoded = self.transformer_encoder(x_projected.unsqueeze(1)).squeeze(1)  # unsqueeze and squeeze to add and remove sequence dimension
        # Reverse project
        reconstructed = self.reverse_projection(encoded)

        return reconstructed, masked_indices


## Baseline model
class HyperImpute(nn.Module):
    def __init__(self, input_dim):
        super(HyperImpute, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, input_dim)  # Predict the original input_dim
        )

    def forward(self, x):
        return self.network(x)