#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -p newnodes
#SBATCH --mem=100GB
#SBATCH --array=0-9 # 10 datasets
hostname

source ~/.bashrc
conda activate omnicell

DATASETS=("PapalexiSatija2021_eccite_arrayed_RNA" 
    "PapalexiSatija2021_eccite_RNA" 
    "ReplogleWeissman2022_rpe1"
    "FrangiehIzar2021_RNA"
    "DatlingerBock2017"
    "DatlingerBock2021"
    "satija_IFNB_raw"
    "repogle_k562_essential_raw"
    "kang"
    "essential_gene_knockouts_raw")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python -m scripts.compute_DEGs.combine_parts \
    --dataset "${DATASET}" 
