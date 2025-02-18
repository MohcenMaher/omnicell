
import os
import pickle
import logging

import numpy as np
from numpy import genfromtxt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from omnicell.processing.utils import to_dense
from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.utils.early_stopping import EarlyStopper

logger = logging.getLogger(__name__)


class VAE(nn.Module):
        
        
        def __init__(self, input_dim, hidden_dim, latent_dim, alpha, dropout_rate):

            super().__init__()
            self.input_dim = input_dim  
            self.latent_dim = latent_dim  # Dimension of the latent space
            self.dropout_rate = dropout_rate  
            self.alpha = alpha  # Weight for the KL divergence term in loss

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_rate))
            
            self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True) 
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True) 

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, input_dim, bias=True))
            
            
        def encode(self, x):

            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):

            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):

            h = torch.relu(self.decoder(z)) # Using ReLU as the final activation
            return h

        def forward(self, x):

            mu, logvar = self.encode(x.view(-1, self.input_dim))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

        def loss_function(self, recon_x, x, mu, logvar):

            recon_loss = F.mse_loss(recon_x, x.view(-1, self.input_dim), reduction="sum")
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + self.alpha * KLD

        def train_vae(self, train_X, valid_X, epochs, optimizer, trainlen, validlen, batsize, device):
           
            for e in range(epochs):
                logger.info(f"Epoch {e+1}/{epochs}")
                running_loss = 0
                self.train()  # Set model to training mode
        
                for lower in range(0, trainlen, batsize):
                    upper = min(lower + batsize, trainlen)
                    lower = min(trainlen - batsize, lower)
                    batch = train_X[lower:upper, :].to(device)

                    optimizer.zero_grad()  
                    out, mu, logvar = self.forward(batch)  # Forward pass
                    if lower == 0:
                        logger.debug(f"Batch shape: {batch.shape}")
                        logger.debug(f"Output shape: {out.shape} - mu shape: {mu.shape} - logvar shape: {logvar.shape}")
 
                    loss = self.loss_function(out, batch, mu, logvar) 
                    loss.backward() 
                    running_loss += loss.item()
                    optimizer.step() 

                logger.info(f"Train loss: {running_loss/1000000}")
        
                running_loss = 0
                self.eval()  # Set model to evaluation mode

                # Validation loop
                with torch.no_grad():
                    for lower in range(0, validlen, batsize):
                        upper = min(lower + batsize, validlen)
                        lower = min(validlen - batsize, lower)
                        batch = valid_X[lower:upper, :].to(device)

                        out, mu, logvar = self.forward(batch)
                        loss = self.loss_function(out, batch, mu, logvar)
                        running_loss += loss.item()

                    logger.info(f"Valid loss: {running_loss/1000000}")

                early_stopper = EarlyStopper(patience=10, min_delta=0.01)
                if early_stopper.early_stop(loss.item()):
                    logger.info(f"Early stopping after {e+1} epochs")
                    break
