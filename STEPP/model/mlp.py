#Script to train a MLP network 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_classification
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import json
import argparse

#MLP encoder decoder architecture 
class ReconstructMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReconstructMLP, self).__init__()
        output_dim = input_dim
        layers = []
        for hd in hidden_dim[:]:
            layers.append(nn.Linear(input_dim, hd))
            layers.append(nn.ReLU())
            input_dim = hd
        layers.append(nn.Linear(input_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    # input_dim = 384
    # hidden layer dim = [input dim, 256, 64, 32, 16, 32, 64, 256, input_dim]


#VAE encoder decoder architecture
class ReconstructVAE (nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(ReconstructVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
            nn.Linear(hidden_dim[3], hidden_dim[4]),
            nn.ReLU(),
            nn.Linear(hidden_dim[4], hidden_dim[5]),
            nn.ReLU(),
            nn.Linear(hidden_dim[5], hidden_dim[6]),
            nn.ReLU(),
            nn.Linear(hidden_dim[6], latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim[6]),
            nn.ReLU(),
            nn.Linear(hidden_dim[6], hidden_dim[5]),
            nn.ReLU(),
            nn.Linear(hidden_dim[5], hidden_dim[4]),
            nn.ReLU(),
            nn.Linear(hidden_dim[4], hidden_dim[3]),
            nn.ReLU(),
            nn.Linear(hidden_dim[3], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], input_dim)
        )

    def forward(self, x):
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std