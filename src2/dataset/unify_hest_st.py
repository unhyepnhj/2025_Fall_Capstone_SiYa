
# -*- coding: utf-8 -*-
"""HEST preprocessing with GLOBAL HVG selection"""

import os
import json
from tqdm import tqdm
from collections import Counter

import pandas as pd
import scanpy as sc
import numpy as np

# ============================================
# Configuration
# ============================================
LOCAL_DIR = 'hest_data'  # Directory containing already-downloaded data with preprocessing.py

# ============================================
# 2-Step Preprocessing with Global HVG
# ============================================

print("\n" + "="*70)
print("STEP 1: Individual Normalization & HVG Scoring")
print("="*70)

st_root = os.path.join(LOCAL_DIR, 'st')
meta_root = os.path.join(LOCAL_DIR, 'metadata')
temp_root = os.path.join(LOCAL_DIR, 'st_temp')
os.makedirs(temp_root, exist_ok=True)

# Collect HVGs from all samples
all_hvg_genes = []
sample_metadata = []

for fname in tqdm(sorted(os.listdir(st_root)), desc="Step 1"):
    if not fname.endswith(".h5ad"):
        continue
    
    fpath = os.path.join(st_root, fname)
    sample_id = fname.replace('.h5ad', '')
    
    try:
        # Load AnnData
        adata = sc.read_h5ad(fpath)
        adata.layers["raw"] = adata.X.copy()
        
        # Make variable names unique
        adata.var_names_make_unique()
        if adata.raw is not None:
            adata.raw.var_names_make_unique()
        
        # Load metadata
        meta_path = os.path.join(meta_root, sample_id + '.json')
        with open(meta_path) as f:
            meta = json.load(f)
        
        organ = meta.get('organ')
        disease_state_raw = meta.get('disease_state')
        
        if disease_state_raw in ['Tumor', 'Cancer']:
            disease_state = 1
        elif disease_state_raw == 'Healthy':
            disease_state = 0
        else:
            disease_state = None
        
        # Add metadata to obs
        adata.obs['sample_id'] = sample_id
        adata.obs['organ'] = organ
        adata.obs['disease_state'] = disease_state
        adata.obs['oncotree_code'] = meta.get('oncotree_code')
        adata.obs['species'] = meta.get('species')
        
        # Preprocessing
        n_obs_before = adata.n_obs
        n_vars_before = adata.n_vars
        
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_counts=1)
        
        # Normalization
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        
        # HVG scoring (flavor changed: seurat_v3 → seurat)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
        
        # Collect HVGs selected in this sample
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
        all_hvg_genes.extend(hvg_genes)
        
        # Save sample metadata (explicit type casting)
        sample_metadata.append({
            'sample_id': str(sample_id),
            'disease_state': int(disease_state) if disease_state is not None else -1,
            'n_obs': int(adata.n_obs),
            'n_vars': int(adata.n_vars),
            'n_hvg': int(len(hvg_genes)),
        })
        
        # Temporary save (all genes kept, normalized state)
        temp_path = os.path.join(temp_root, fname)
        adata.write_h5ad(temp_path, compression='gzip', compression_opts=9)
        
        print(f"  {sample_id}: {n_obs_before}→{adata.n_obs} spots, "
              f"{n_vars_before}→{adata.n_vars} genes, {len(hvg_genes)} HVGs")
        
    except Exception as e:
        print(f"  ✗ Failed {sample_id}: {e}")
        continue

# ============================================
# STEP 2: Select Global HVG
# ============================================
print("\n" + "="*70)
print("STEP 2: Select Global HVG (Top 2000 genes)")
print("="*70)

# Select top 2000 genes most frequently chosen as HVGs across samples
gene_counter = Counter(all_hvg_genes)
most_common = gene_counter.most_common(2000)
global_hvg = sorted([gene for gene, count in most_common])

print(f"\n Global HVG Selection:")
print(f"  Total unique HVG candidates: {len(set(all_hvg_genes))}")
print(f"  Selected global HVGs: {len(global_hvg)}")
print(f"\nTop 10 genes by frequency:")
for i, (gene, count) in enumerate(most_common[:10], 1):
    pct = 100 * count / len(sample_metadata)
    print(f"  {i:2d}. {gene:15s}: {count:3d}/{len(sample_metadata):3d} samples ({pct:5.1f}%)")

# ============================================
# STEP 3: Filter to Global HVG & Save
# ============================================
print("\n" + "="*70)
print("STEP 3: Filter to Global HVG and Save")
print("="*70)

processed_root = os.path.join(LOCAL_DIR, 'st_preprocessed_global_hvg')
os.makedirs(processed_root, exist_ok=True)

healthy_count = 0
cancer_count = 0
failed_samples = []
gene_coverage = []

for fname in tqdm(sorted(os.listdir(temp_root)), desc="Step 3"):
    if not fname.endswith(".h5ad"):
        continue
    
    temp_path = os.path.join(temp_root, fname)
    sample_id = fname.replace('.h5ad', '')
    
    try:
        # Load temporary file
        adata = sc.read_h5ad(temp_path)
        
        # Intersection with global HVG
        available_genes = np.intersect1d(adata.var_names, global_hvg)
        coverage = len(available_genes) / len(global_hvg) * 100
        gene_coverage.append(coverage)
        
        if len(available_genes) == 0:
            print(f"  ✗ {sample_id}: No common genes!")
            failed_samples.append(sample_id)
            continue
        
        # Filter to global HVGs
        adata = adata[:, available_genes].copy()
        
        # Disease state counting
        disease_state = adata.obs['disease_state'].values[0]
        if disease_state == 1:
            cancer_count += 1
        elif disease_state == 0:
            healthy_count += 1
        
        # Final save
        output_path = os.path.join(processed_root, fname)
        adata.write_h5ad(output_path, compression='gzip', compression_opts=9)
        
        print(f"  ✓ {sample_id}: {adata.n_obs} spots, "
              f"{adata.n_vars}/{len(global_hvg)} genes ({coverage:.1f}%)")
        
    except Exception as e:
        print(f"  ✗ Failed {sample_id}: {e}")
        failed_samples.append(sample_id)
        continue

# ============================================
# 4. Cleanup & Summary
# ============================================
print("\n" + "="*70)
print("Cleaning up temporary files...")
print("="*70)

import shutil
shutil.rmtree(temp_root)
print("✓ Temporary files removed")

# Summary
print("\n" + "="*70)
print("PREPROCESSING SUMMARY")
print("="*70)
print(f"Total samples processed: {healthy_count + cancer_count}")
print(f"  - Healthy: {healthy_count}")
print(f"  - Cancer/Tumor: {cancer_count}")
print(f"\nGlobal HVG: {len(global_hvg)} genes")
print(f"Gene coverage across samples:")
print(f"  - Mean: {np.mean(gene_coverage):.2f}%")
print(f"  - Min:  {np.min(gene_coverage):.2f}%")
print(f"  - Max:  {np.max(gene_coverage):.2f}%")

if failed_samples:
    print(f"\nFailed samples ({len(failed_samples)}):")
    for s in failed_samples:
        print(f"  - {s}")
else:
    print("\nAll samples processed successfully!")

# Save global HVG list
hvg_path = os.path.join(LOCAL_DIR, 'global_hvg_genes.txt')
with open(hvg_path, 'w') as f:
    f.write('\n'.join(global_hvg))
print(f"\n✓ Global HVG list saved to: {hvg_path}")

# Save sample metadata
metadata_df = pd.DataFrame(sample_metadata)
metadata_path = os.path.join(LOCAL_DIR, 'sample_metadata.csv')
metadata_df.to_csv(metadata_path, index=False)
print(f"✓ Sample metadata saved to: {metadata_path}")

print("\n" + "="*70)
print("PREPROCESSING COMPLETE!")
print("="*70)
print(f"\nProcessed data location: {processed_root}")
print(f"Number of files: {len([f for f in os.listdir(processed_root) if f.endswith('.h5ad')])}")
