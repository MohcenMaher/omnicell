import logging

import torch
from torch import nn, optim
import numpy as np
import scanpy as sc

from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.VAE.vae import VAE
from omnicell.models.utils.early_stopping import EarlyStopper
from omnicell.processing.utils import to_dense

logger = logging.getLogger(__name__)


class LinearRegression(nn.Module):
    
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    
    def forward(self, x):
        return self.linear(x)


class scVIDRPredictor():

    """

    Module for training and predicting with scVIDR: Single-Cell Variational Inference of Dose-Response  
    A VAE-based model that predicts gene expression perturbations across cell types.  
    Calculates delta using regression to adjust for cell-type specific effects, allowing predictions for unseen cell types.  
    
    """
 
    def __init__(self, config, input_dim, device, pert_ids):

        self.training_config = config['training']
        self.model_config = config['model']
        self.input_dim = input_dim
        self.device = device
        self.model = VAE(input_dim, self.model_config['hidden_dim'],
                         self.model_config['latent_dim'], 
                         self.training_config['alpha'],
                         self.training_config['dropout_rate'])
        
        self.epochs = self.training_config['epochs']
        self.delta_predictor_epochs = self.training_config['delta_predictor_epochs']
        self.batsize = self.training_config['batsize']
        self.model.to(device)

        self.perts = pert_ids
        self.deltas =  None #Stores the latent space shift associated which each perturbation
        self.perts_to_idx = {pert: idx for idx, pert in enumerate(pert_ids)}
        self.cell_ids = None
        self.invalid_perts = set()

        logger.debug(f"VAE predictor initialized, perturbations: {self.perts}, perts to idx: {self.perts_to_idx}")


    def train(self, adata):

        self.cell_ids = adata.obs[CELL_KEY].unique()
        device = self.device
        epochs = self.epochs
        batsize = self.batsize
        cell_types_with_pert = adata.obs[adata.obs[PERT_KEY] != CONTROL_PERT][CELL_KEY].unique()

        datalen = len(adata)
        indices = np.random.permutation(datalen)

        # Set up training data
        train = adata[indices[:np.int32(datalen*0.9)]]  # 90% training data
        valid = adata[indices[np.int32(datalen*0.9):]]         
        trainlen = train.shape[0]
        validlen = valid.shape[0]
        train_X = to_dense(train.X)
        valid_X = to_dense(valid.X)
        train_X = torch.from_numpy(train_X.astype(np.float32))  
        valid_X = torch.from_numpy(valid_X.astype(np.float32)) 

        logger.debug(f"Train data shape: {train.shape}")
        logger.debug(f"Validation data shape: {valid.shape}")
        logger.debug(f"Training data shape: {train_X.shape}")
        logger.debug(f"Validation data shape: {valid_X.shape}")
        logger.debug(f"Batch size: {batsize}")
        logger.info(f"Training VAE model for {epochs} epochs")

        # Train model
        vae = self.model
        vae.to(device)  
        optimizer = optim.Adam(vae.parameters(), lr=self.training_config["learning_rate"])  # Adam optimizer
        vae.train_vae(train_X, valid_X, epochs, optimizer, trainlen, validlen, batsize, device)

        # Compute mean latent space representation for each cell type in training for each perturbation
        vae.eval() 
        cell_means = {}
        for cell in cell_types_with_pert: 
            cell_ctrl_adata = train[(train.obs[CELL_KEY] == cell) & (train.obs[PERT_KEY] == CONTROL_PERT)]
            logger.debug(f"Shape of ctrl data for cell {cell}: {cell_ctrl_adata.shape}")
            cell_data = torch.from_numpy(to_dense(cell_ctrl_adata.X)).to(device)
            mu, _ = vae.encode(cell_data)
            cell_mean = mu.mean(axis=0).cpu().detach()
            cell_means[cell] = cell_mean

        self.regressors = []
        self.deltas = []
        # Altough scVIDR theoretically only uses one perturbation, we will use multiple for the sake of generality
        # Compute delta for each celltype for each perturbation:
        for i in range(len(self.perts)):
            
            self.deltas.append({})
            x_and_y_data = [] # Tuple for each cell type: (latent representstion mean, delta)
            for j, cell in enumerate(self.cell_ids):
                logger.debug(f'Computing mean for perturbation {self.perts[i]} and cell type {cell}')
                cell_pert_adata = adata[(adata.obs[CELL_KEY] == cell) & (adata.obs[PERT_KEY] == self.perts[i])]
                # Due to iid mode we might not have all perturbations for all cell types
                if len(cell_pert_adata) > 0:
                    pert_data = torch.from_numpy(to_dense(cell_pert_adata.X)).to(device)
                    pert_mu, _ = vae.encode(pert_data)
                    pert_mean = pert_mu.mean(axis=0).cpu().detach()
                    delta = pert_mean - cell_means[cell]
                    self.deltas[i][cell] = delta
                    x_and_y_data.append((cell_means[cell], delta))
                else:
                    logger.info(f'Cell {cell} does not have perturbation {self.perts[i]}')
               
            if len(x_and_y_data) == 0:
                self.invalid_perts.add(self.perts[i])
                logger.warning(f'Perturbation {self.perts[i]} is not applied to any cell in the training data - Delta linear model will be None and trying to transfer this perturbation to unseen cells will cause an undefined state.')
                self.regressors.append(None)
            else:
                # Regressor trained to enable prediction of delta for unseen celltypes
                logger.info(f"Training regressor to predict delta for perturbation {self.perts[i]}")
                model = LinearRegression(self.model_config['latent_dim'], self.model_config['latent_dim'])
                criterion = nn.MSELoss()
                optimizer = optim.SGD(model.parameters(), lr=0.01)
                for _ in range(self.delta_predictor_epochs):
                    logger.debug(f"Epoch {_+1}/{self.delta_predictor_epochs}")
                    for X, y in x_and_y_data:
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                self.regressors.append(model)

    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:

        assert len(adata.obs[CELL_KEY].unique()) == 1, 'Input data contains multiple cell types'
        assert len(adata.obs[PERT_KEY].unique()) == 1, 'Input data contains multiple perturbations'
        assert adata.obs[CELL_KEY].unique() == [cell_type], f'Cell type {cell_type} not in the provided data'
        assert adata.obs[PERT_KEY].unique() == [CONTROL_PERT], 'Input data contains non control perturbations'  
        
        if pert_id in self.invalid_perts:
            raise ValueError(f'Perturbation {pert_id} is not applied to any cell in the training data - Delta is NaN')

        logger.info(f'Predicting seen perturbation {pert_id} for unseen cell type {cell_type}')

        data = to_dense(adata.X)
        data = torch.from_numpy(data.astype(np.float32)).to(self.device)
        mu, logvar = self.model.encode(data)

        logger.debug(f"Pert idx = {self.perts_to_idx[pert_id]}")
        # If cell type seen in training, use delta directly, otherwise use regressor
        if cell_type in self.cell_ids:
            pert_delta = self.deltas[self.perts_to_idx[pert_id]][cell_type]

        else:
            with torch.no_grad():
                pert_predictor = self.regressors[self.perts_to_idx[pert_id]]
                pert_predictor.eval()
                mu, logvar = self.model.encode(data)
                pert_delta = pert_predictor(mu.mean(axis=0).cpu())

        pert_mu = mu + pert_delta.to(self.device)
        z = self.model.reparameterize(pert_mu, logvar)
        return self.model.decode(z).cpu().detach().numpy()

