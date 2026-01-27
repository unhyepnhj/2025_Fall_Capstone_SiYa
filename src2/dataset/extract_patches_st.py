import h5py
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import scanpy as sc

def extract_patches_to_h5(st_dir, img_dir, output_dir, patch_size=224):
    """
    Extract patches from STimage WSI and save them as H5 files
    in a HEST-compatible format
    
    Args:
        st_dir: Directory containing ST .h5ad files
        img_dir: Directory containing full WSI image files
        output_dir: Directory to save output H5 files
        patch_size: Size of extracted patches (default: 224)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = sorted([f for f in os.listdir(st_dir) if f.endswith('.h5ad')])
    
    print(f"Processing {len(files)} STimage samples...")
    
    for f_name in tqdm(files, desc="Extracting patches"):
        sample_id = f_name.replace(".h5ad", "")
        
        try:
            # 1. Load coordinates and barcodes from AnnData
            adata = sc.read_h5ad(os.path.join(st_dir, f_name))
            
            if 'spatial' not in adata.obsm:
                print(f"Skip {sample_id}: No spatial coordinates")
                continue
            
            coords = adata.obsm['spatial']  # (N, 2) - [x, y]
            barcodes = adata.obs_names.to_numpy()
            
            # 2. Load WSI image (try multiple extensions)
            img_path = None
            for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                candidate = os.path.join(img_dir, f"{sample_id}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            
            if img_path is None:
                print(f"Skip {sample_id}: Image not found")
                continue
            
            # Load image
            if img_path.endswith(('.tif', '.tiff')):
                # Load TIFF using PIL
                full_img = np.array(Image.open(img_path))
            else:
                # Load PNG/JPG using OpenCV
                full_img = cv2.imread(img_path)
                full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
            
            H, W = full_img.shape[:2]
            
            # 3. Extract patches at each spot coordinate
            patches = []
            valid_barcodes = []
            
            half = patch_size // 2
            
            for i, (x, y) in enumerate(coords):
                x, y = int(x), int(y)
                
                # Boundary check
                if (x - half < 0 or x + half > W or 
                    y - half < 0 or y + half > H):
                    continue
                
                # Extract patch
                patch = full_img[y-half:y+half, x-half:x+half]
                
                # Validate patch size
                if patch.shape == (patch_size, patch_size, 3):
                    patches.append(patch)
                    valid_barcodes.append(barcodes[i])
            
            if len(patches) == 0:
                print(f"Skip {sample_id}: No valid patches extracted")
                continue
            
            # 4. Save in HEST-compatible H5 format
            patches_array = np.array(patches, dtype=np.uint8)  # (N, H, W, 3)
            barcodes_array = np.array(valid_barcodes, dtype='S')
            
            h5_path = os.path.join(output_dir, f"{sample_id}.h5")
            
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('img', data=patches_array, compression='gzip')
                f.create_dataset('barcode', data=barcodes_array)
            
            print(f"  ✓ {sample_id}: {len(patches)} patches extracted")
            
        except Exception as e:
            print(f"  ✗ Error processing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

# Run
extract_patches_to_h5(
    st_dir='stimage_data/st',
    img_dir='stimage_data/patches',
    output_dir='merged_data/extracted',
    patch_size=224
)