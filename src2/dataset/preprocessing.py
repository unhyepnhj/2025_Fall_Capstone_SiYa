
# -*- coding: utf-8 -*-
"""HEST preprocessing with patches and compatibility fix"""

import os
import sys
import zipfile
import json
from tqdm import tqdm

# Install required packages (for server execution)
def install_packages():
    os.system("pip install scanpy==1.10.3")
    os.system("apt-get -y install openslide-tools")
    os.system("pip install openslide-python")
    os.system("pip install git+https://github.com/mahmoodlab/HEST.git")
    os.system("pip install huggingface-hub")
    os.system("pip install numba==0.60.0")
    os.system("pip install hdf5plugin")  # To prevent IOSpec-related errors

# install_packages()  # Uncomment if needed

from huggingface_hub import login, snapshot_download
import pandas as pd
import scanpy as sc
import shutil

# ============================================
# Configuration
# ============================================
HF_TOKEN = "YOUR_TOKEN"
DOWNLOAD_ALL = False
FOLDERS = ['metadata', 'st', 'patches']  # Added patches folder
LOCAL_DIR = 'hest_data'

# Login to Hugging Face
login(token=HF_TOKEN)

# ============================================
# 1. Download dataset
# ============================================
def download_hest(ids_to_query, local_dir, download_all=False):
    """Download HEST dataset"""
    repo_id = 'MahmoodLab/hest'
    
    if not download_all:
        folders = FOLDERS
        allow_patterns = []
        for fid in ids_to_query:
            for folder in folders:
                allow_patterns.append(f"{folder}/{fid}[._]*")
        print(f"Downloading {len(allow_patterns)} patterns...")
        snapshot_download(
            repo_id=repo_id, 
            allow_patterns=allow_patterns, 
            repo_type="dataset", 
            local_dir=local_dir
        )
    else:
        patterns = [f"*{id}[_.]**" for id in ids_to_query]
        snapshot_download(
            repo_id=repo_id, 
            allow_patterns=patterns, 
            repo_type="dataset", 
            local_dir=local_dir
        )
    
    # Unzip CellViT segmentation files
    seg_dir = os.path.join(local_dir, 'cellvit_seg')
    if os.path.exists(seg_dir):
        print('Unzipping CellViT segmentation...')
        for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith('.zip')]):
            path_zip = os.path.join(seg_dir, filename)
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(seg_dir)

# ============================================
# 1. Sample IDs to download (fixed list)
# ============================================

"""
ids_to_query = [
    'MEND124', 'MEND46', 'MEND63', 'MEND64', 'MEND65', 'MEND66', 'MEND67', 'MEND68',
    'MEND71', 'MEND77', 'MISC8', 'NCBI342', 'NCBI343', 'NCBI353', 'NCBI357',
    'NCBI359', 'NCBI360', 'NCBI370', 'NCBI371', 'NCBI372', 'NCBI385', 'NCBI386',
    'NCBI397', 'NCBI628', 'NCBI629', 'NCBI630', 'NCBI631', 'NCBI632', 'NCBI633',
    'NCBI634', 'NCBI635', 'NCBI636', 'NCBI637', 'NCBI638', 'NCBI639', 'NCBI640',
    'NCBI641', 'NCBI716', 'NCBI800', 'SPA11', 'SPA13', 'TENX138', 'TENX19',
    'TENX30', 'TENX31', 'TENX61', 'TENX73', 'TENX80',
]
"""

ids_to_query = ['MEND124', 'SPA11']

print(f"Total samples to download: {len(ids_to_query)}")
print(f"Sample IDs: {ids_to_query}")

# Execute download
print("\nStarting download...")
download_hest(ids_to_query, LOCAL_DIR, DOWNLOAD_ALL)
print("Download complete!")

# ============================================
# 2. Preprocessing
# ============================================
print("\n" + "="*50)
print("Starting preprocessing...")
print("="*50)

st_root = os.path.join(LOCAL_DIR, 'st')
meta_root = os.path.join(LOCAL_DIR, 'metadata')
processed_root = os.path.join(LOCAL_DIR, 'st_preprocessed')
os.makedirs(processed_root, exist_ok=True)

healthy_count = 0
cancer_count = 0
failed_samples = []

for fname in tqdm(os.listdir(st_root)):
    if not fname.endswith(".h5ad"):
        continue
    
    fpath = os.path.join(st_root, fname)
    sample_id = fname.replace('.h5ad', '')
    
    try:
        print(f"\nProcessing: {sample_id}")
        
        # Load AnnData
        adata = sc.read_h5ad(fpath)
        adata.layers["raw"] = adata.X.copy()  # Backup raw counts
        
        # Make variable names unique
        adata.var_names_make_unique()
        if adata.raw is not None:
            adata.raw.var_names_make_unique()
        
        # Load and attach metadata
        meta_path = os.path.join(meta_root, sample_id + '.json')
        with open(meta_path) as f:
            meta = json.load(f)
        
        organ = meta.get('organ')
        disease_state_raw = meta.get('disease_state')
        
        # Encode disease state
        if disease_state_raw in ['Tumor', 'Cancer']:
            disease_state = 1
            cancer_count += 1
        elif disease_state_raw == 'Healthy':
            disease_state = 0
            healthy_count += 1
        else:
            disease_state = None
        
        oncotree_code = meta.get('oncotree_code')
        species = meta.get('species')
        
        # Add metadata to obs
        adata.obs['sample_id'] = sample_id
        adata.obs['organ'] = organ
        adata.obs['disease_state'] = disease_state
        adata.obs['oncotree_code'] = oncotree_code
        adata.obs['species'] = species
        
        # Basic filtering
        print(f"  - Before filtering: {adata.n_obs} cells, {adata.n_vars} genes")
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_counts=1)
        print(f"  - After filtering: {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Normalization
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        
        # Highly variable gene selection
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']].copy()
        print(f"  - After HVG selection: {adata.n_vars} genes")
        
        # ★★★ Save with compatibility-friendly settings (prevent IOSpec errors) ★★★
        output_path = os.path.join(processed_root, fname)
        adata.write_h5ad(
            output_path,
            compression='gzip',  # Widely compatible compression
            compression_opts=9
        )
        print(f"  ✓ Saved to {output_path}")
        
    except Exception as e:
        print(f"  ✗ Failed {sample_id}: {e}")
        failed_samples.append(sample_id)
        continue

# ============================================
# 3. Summary
# ============================================
print("\n" + "="*50)
print("Preprocessing Summary")
print("="*50)
print(f"Healthy samples: {healthy_count}")
print(f"Cancer/Tumor samples: {cancer_count}")
print(f"Total processed: {healthy_count + cancer_count}")
if failed_samples:
    print(f"Failed samples ({len(failed_samples)}): {failed_samples}")
else:
    print("All samples processed successfully!")

# ============================================
# 4. Archive (optional)
# ============================================
print("\nCreating archive...")
shutil.make_archive('hest_data_preprocessed', 'zip', LOCAL_DIR)
print("Archive created: hest_data_preprocessed.zip")

print("\n✓ All done!")
