import sys
import yaml
import torch
import logging
from pathlib import Path

# Add the path to the directory containing the omnicell package
# Assuming the omnicell package is in the parent directory of your notebook
sys.path.append('..')  # Adjust this path as needed

import yaml
import torch
import logging
from pathlib import Path
from omnicell.config.config import Config, ETLConfig, ModelConfig, DatasplitConfig, EvalConfig, EmbeddingConfig
from omnicell.data.loader import DataLoader
from omnicell.constants import PERT_KEY, GENE_EMBEDDING_KEY, CONTROL_PERT
from omnicell.models.selector import load_model as get_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure paths
MODEL_CONFIG = ModelConfig.from_yaml("/home/jason497/omnicell/configs/models/autoencoder.yaml")
ETL_CONFIG = ETLConfig(name = "no_preprocessing", log1p = False, drop_unmatched_perts = True)
EMBEDDING_CONFIG = EmbeddingConfig(pert_embedding='GenePT')

SPLIT_CONFIG = DatasplitConfig.from_yaml("/orcd/data/omarabu/001/njwfish/omnicell/configs/splits/repogle_k562_essential_raw/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1/split_0/split_config.yaml")
#SPLIT_CONFIG = DatasplitConfig.from_yaml("/orcd/data/omarabu/001/njwfish/omnicell/configs/splits/repogle_k562_essential_raw/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1/split_1/split_config.yaml")
EVAL_CONFIG = EvalConfig.from_yaml("/orcd/data/omarabu/001/njwfish/omnicell/configs/splits/repogle_k562_essential_raw/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1/split_0/eval_config.yaml")  # Set this if you want to run evaluations
#EVAL_CONFIG = EvalConfig.from_yaml("/orcd/data/omarabu/001/njwfish/omnicell/configs/splits/repogle_k562_essential_raw/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1/split_1/eval_config.yaml")

# Load configurations
config = Config(model_config=MODEL_CONFIG,
                 etl_config=ETL_CONFIG, 
                 datasplit_config=SPLIT_CONFIG, 
                 eval_config=EVAL_CONFIG)


#Alternatively you can initialize the config objects manually as follows:
# etl_config = ETLConfig(name = XXX, log1p = False, drop_unmatched_perts = False, ...)
# model_config = ...
# embedding_config = ...
# datasplit_config = ...
# eval_config = ...
# config = Config(etl_config, model_config, datasplit_config, eval_config)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize data loader and load training data
loader = DataLoader(config)
adata, pert_rep_map = loader.get_training_data()

# Get dimensions and perturbation IDs
input_dim = adata.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pert_ids = adata.obs[PERT_KEY].unique()
gene_emb_dim = adata.varm[GENE_EMBEDDING_KEY].shape[1] if GENE_EMBEDDING_KEY in adata.varm else None

print(f"Data loaded:")
print(f"- Number of cells: {adata.shape[0]}")
print(f"- Input dimension: {input_dim}")
print(f"- Number of perturbations: {len(pert_ids)}")
# get index of pert in adata.var_names
pert_list = adata.var_names.values.tolist()
pert_rep_map_idxs = {pert: pert_list.index(pert) for pert in adata.obs[PERT_KEY].unique() if pert != CONTROL_PERT}

import numpy as np



npz = np.load("/orcd/data/omarabu/001/Omnicell_datasets/repogle_k562_essential_raw/proportional_scot/synthetic_counterfactuals_0.pkl", allow_pickle=True)

from omnicell.models.utils.datamodules import StratifiedBatchSampler

class PairedStratifiedDataset(torch.utils.data.Dataset):
    def __init__(
            self, source_dict, target_dict, pert_map
    ):
        self.source = source_dict
        self.target = target_dict
        self.strata = np.array(list(self.source.keys()))
        print(self.strata)
        self.unique_pert_ids = np.array(list(self.target[self.strata[0]].keys()))
        print(self.unique_pert_ids)
        self.pert_map = pert_map
        self.ns = np.array([
            len(self.source[stratum]) for stratum in self.strata
        ])

        self.samples_per_epoch = len(self.unique_pert_ids) * self.source[self.strata[0]].shape[0]

    def __len__(self):
        return len(self.source_dict * self.target_dict)
    
    def __getitem__(self, strata_idx):
        (stratum_idx,), idx = strata_idx
        stratum = self.strata[stratum_idx]
        pert = np.random.choice(self.unique_pert_ids)
        return (
            self.source[stratum][idx],
            self.target[stratum][pert][idx],
            self.pert_map[pert]
        )
    
dset = PairedStratifiedDataset(
    source_dict=npz['source'],
    target_dict=npz['synthetic_counterfactuals'],
    pert_map=pert_rep_map_idxs
)

dl = torch.utils.data.DataLoader(
    dset, 
    batch_sampler=StratifiedBatchSampler(
        ns=dset.ns, batch_size=16, samples_per_epoch=dset.samples_per_epoch
    )
)


#################################
#  In[2]: Initialize the Model
#################################

model = get_model(config.model_config, dl, pert_rep_map, input_dim, device, pert_ids)

###############################################
#  In[3]: Train the Model on the Training Data
###############################################

model.train(dl)

#####################################################
#  In[4]: Simple Loop to Illustrate the Evaluation
#####################################################

import numpy as np

logger.info("Running evaluation")

# evaluate each pair of cells and perts
eval_dict = {}
for cell_id, pert_id, ctrl_data, gt_data in loader.get_eval_data():
    logger.debug(f"Making predictions for cell: {cell_id}, pert: {pert_id}")

    preds = model.make_predict(ctrl_data, pert_id, cell_id)
    eval_dict[(cell_id, pert_id)] = (ctrl_data.X.toarray(), gt_data.X.toarray(), preds)

    


#############################################################
#  In[5]: DEG Calculation and Additional Model Comparisons
#############################################################

import scanpy as sc
from omnicell.evaluation.utils import get_DEGs, get_eval, get_DEG_Coverage_Recall, get_DEGs_overlaps

pval_threshold = 0.05
log_fold_change_threshold = 0.0

results_dict = {}

for (cell, pert) in eval_dict:
    ctrl_data, gt_data, pred_pert = eval_dict[(cell, pert)]
    
    

    pred_pert_adat = sc.AnnData(X=pred_pert.copy())
    true_pert = sc.AnnData(X=gt_data.copy())
    control = sc.AnnData(X=ctrl_data.copy())
    

    sc.pp.normalize_total(control, target_sum=1e5)
    sc.pp.normalize_total(true_pert, target_sum=1e5)
    sc.pp.normalize_total(pred_pert_adat, target_sum=1e5)
    
    sc.pp.log1p(control)
    sc.pp.log1p(true_pert)
    sc.pp.log1p(pred_pert_adat)
    


    
    
    

    logger.debug(f"Getting ground truth DEGs for {pert} and {cell}")
    true_DEGs_df = get_DEGs(control, true_pert)
    signif_true_DEG = true_DEGs_df[true_DEGs_df['pvals_adj'] < pval_threshold]
    logger.debug(f"Number of significant DEGs from ground truth: {signif_true_DEG.shape[0]}")

    logger.debug(f"Getting predicted DEGs for {pert} and {cell}")
    pred_DEGs_df = get_DEGs(control, pred_pert_adat)

    logger.debug(f"Getting evaluation metrics for {pert} and {cell}")
    DEGs_overlaps = get_DEGs_overlaps(
        true_DEGs_df,
        pred_DEGs_df,
        [100, 50, 20],
        pval_threshold,
        log_fold_change_threshold
    )
    print((cell,pert))
    print(DEGs_overlaps)

    r2_and_mse = get_eval(
        control,
        true_pert,
        pred_pert_adat,
        true_DEGs_df,
        [100, 50, 20],
        pval_threshold,
        log_fold_change_threshold
    )

    logger.debug(f"Getting DEG overlaps for {pert} and {cell}")


    results_dict[(cell, pert)] = (r2_and_mse, DEGs_overlaps)


print(results_dict)
