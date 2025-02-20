import os
import argparse
import torch
import scanpy as sc
from omnicell.data.catalogue import Catalogue
from omnicell.data.loader import DataLoader, DatasetDetails
from omnicell.constants import DATA_CATALOGUE_PATH
import numpy as np
from sklearn.decomposition import PCA



EMBEDDING_NAME = 'GeneCorr_PCA256'
N_COMPONENTS = 256

def main():
    parser = argparse.ArgumentParser(description='Generating Embeddings based on correlations of raw count matrix, takign the top 256 components')
    parser.add_argument('--dataset_name', type=str, required=True, 
                       help='Name of the dataset')



    args = parser.parse_args()

    # Load dataset details
    ds_details = Catalogue.get_dataset_details(args.dataset_name)

    # Check if embedding already exists
    if EMBEDDING_NAME in ds_details.pert_embeddings:
        print(f"Embedding {EMBEDDING_NAME} already exists for dataset {args.dataset_name}")
        return

  
    
    # Load dataset
    adata = sc.read(ds_details.path)
    X = adata.X.toarray()

    print(f"Loaded dataset {args.dataset_name} with shape {X.shape}")

    pert_key = ds_details.pert_key
    control = ds_details.control
    
    # Get perturbations (excluding control)
    perts = [p for p in adata.obs[pert_key].unique() if p != control]

    corr_matrix = np.corrcoef(X.T)

  
    # Ensure the matrix is symmetric
    if not np.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("Input matrix is not symmetric")
    
    # Initialize PCA; note that n_components cannot exceed the number of observations.
    pca = PCA(n_components=min(N_COMPONENTS, corr_matrix.shape[0]))
    
    # Fit PCA and transform the correlation matrix
    embeddings = pca.fit_transform(corr_matrix)
    variance_ratio = pca.explained_variance_ratio_


    missing_count = 0
    found_perts = []
    found_perts_idx = []
    for p in perts:
        if p in adata.var_names:
            found_perts.append(p)
            found_perts_idx.append(adata.var_names.get_loc(p))
        else:
            missing_count += 1

    save_dir = f"{ds_details.folder_path}/pert_embeddings/"

    embeddings = torch.from_numpy(embeddings[found_perts_idx])
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Variance ratio: {variance_ratio}")

    torch.save({
        "embedding": embeddings,
        "pert_names": found_perts
    }, f"{save_dir}/{EMBEDDING_NAME}.pt")

    
    # Update catalogue
    Catalogue.register_new_pert_embedding(args.dataset_name, EMBEDDING_NAME)
    print(f"Successfully saved {len(found_perts)} embeddings in {EMBEDDING_NAME} for {args.dataset_name}")

if __name__ == '__main__':
    main()