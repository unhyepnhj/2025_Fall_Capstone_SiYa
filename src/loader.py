import os
import h5py
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import gc

# -------------------------------------------------------
# Custom Sample class
# -------------------------------------------------------
class CustomSample:
    def __init__(self, root, sample_id):
        self.sample_id = sample_id
        self.st_path = os.path.join(root, "st_preprocessed", f"{sample_id}.h5ad")
        self.patch_path = os.path.join(root, "patches", f"{sample_id}.h5")
        
        if not os.path.exists(self.st_path):
            raise FileNotFoundError(f"{self.st_path} not found.")
        if not os.path.exists(self.patch_path):
            raise FileNotFoundError(f"{self.patch_path} not found.")

# -------------------------------------------------------
# WSI-level Dataset
# -------------------------------------------------------
class WSIDataset(Dataset):
    def __init__(self, samples, max_spots=2000, use_label=True, target_genes=None):
        self.samples = samples
        self.max_spots = max_spots
        self.use_label = use_label
        self.target_genes = target_genes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        adata = None
        
        try:
            # 1. Load AnnData (backed mode)
            adata = sc.read_h5ad(sample.st_path, backed='r')
            
            # 2. Load gene expression data
            if self.target_genes is not None:
                common_genes = np.intersect1d(adata.var_names, self.target_genes)
                
                if len(common_genes) == 0:
                    expr_vals = np.zeros((adata.n_obs, len(self.target_genes)))
                else:
                    raw_data = adata[:, common_genes].X
                    if hasattr(raw_data, "toarray"):
                        expr_vals = raw_data.toarray()
                    else:
                        expr_vals = np.array(raw_data)
                
                df = pd.DataFrame(expr_vals, columns=common_genes if len(common_genes) > 0 else [])
                df = df.reindex(columns=self.target_genes, fill_value=0.0)
                expr = df.values
            else:
                if hasattr(adata.X, 'toarray'):
                    expr = adata.X.toarray()
                else:
                    expr = np.array(adata.X[:])
            
            # 3. Metadata
            barcodes_st = adata.obs_names.to_numpy()
            coords_st = np.array(adata.obsm["spatial"][:])
            
            val = adata.obs['disease_state'].values[0]
            label_val = int(val) if not pd.isna(val) else 0
            
            # Immediately delete AnnData
            del adata
            adata = None
            
            # 4. Load patch data
            with h5py.File(sample.patch_path, "r") as patches:
                imgs = patches["img"][:]
                raw_bar = np.array(patches["barcode"])
            
            # 5. Barcode alignment
            patch_barcodes = [
                b.decode() if isinstance(b, bytes) else str(b)
                for b in raw_bar.squeeze()
            ]
            b2i = {b: i for i, b in enumerate(patch_barcodes)}

            patch_indices = []
            st_indices = []
            for i, b in enumerate(barcodes_st):
                if b in b2i:
                    patch_indices.append(b2i[b])
                    st_indices.append(i)

            if len(patch_indices) == 0:
                print(f"Warning: No aligned spots in {sample.sample_id}")
                return self.__getitem__((idx + 1) % len(self))

            # 6. Align data
            images = imgs[patch_indices]
            expr = expr[st_indices]
            coords = coords_st[st_indices]
            
            # 7. Sampling (memory-efficient)
            N = len(images)
            if self.max_spots is not None and N > self.max_spots:
                sample_indices = np.sort(np.random.choice(N, self.max_spots, replace=False))
                images = images[sample_indices]
                expr = expr[sample_indices]
                coords = coords[sample_indices]

            # 8. Convert to tensors
            label = torch.tensor(label_val).long() if self.use_label else torch.tensor(-1).long()
            
            # Image normalization & type conversion
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
            expr = torch.from_numpy(expr).float()
            coords = torch.from_numpy(coords).float()
            
            # 9. Coordinate normalization
            if coords.shape[0] > 1:
                c_min = coords.min(dim=0, keepdim=True)[0]
                c_max = coords.max(dim=0, keepdim=True)[0]
                c_range = c_max - c_min
                c_range[c_range == 0] = 1.0
                coords = (coords - c_min) / c_range
            else:
                coords = torch.zeros_like(coords)

            return {
                "images": images,
                "expr": expr,
                "coords": coords,
                "label": label,
                "sample_id": sample.sample_id,
                "num_spots": len(images)
            }
            
        except Exception as e:
            print(f"Error loading {sample.sample_id}: {e}")
            return self.__getitem__((idx + 1) % len(self))
            
        finally:
            if adata is not None:
                del adata
            gc.collect()

def wsi_collate_fn(batch):
    if len(batch) == 1:
        return batch[0]
    return batch

def create_wsi_dataloader(samples, batch_size=1, shuffle=True, max_spots=2000, target_genes=None):
    dataset = WSIDataset(samples, max_spots=max_spots, use_label=True, target_genes=target_genes)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,   # Prevent OOM: set workers to 0
        pin_memory=False,  # Prevent OOM: disable pin_memory
        collate_fn=wsi_collate_fn
    )
    return loader

# -------------------------------------------------------
# Get gene info (revised)
# -------------------------------------------------------
def get_gene_info(samples):
    """Get gene information from the first sample"""
    sample = samples[0]
    
    # directly load AnnData
    adata = sc.read_h5ad(sample.st_path, backed='r')
    num_genes = adata.n_vars
    gene_names = adata.var_names.tolist()
    del adata
    
    print(f"Gene info:")
    print(f"  Total genes: {num_genes}")
    print(f"  Gene names (first 10): {gene_names[:10]}")
    
    return num_genes, gene_names