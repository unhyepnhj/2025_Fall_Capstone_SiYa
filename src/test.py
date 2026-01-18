import torch
import torch.nn as nn
from loader import create_hest_dataloader, CustomSample  # import from loader.py
from model import MultiModalHestModel

def run_sanity_check():
    # 1. Load data 
    root = "/workspace/Temp"
    id_list = ["TENX24", "TENX39", "TENX97", "MISC61", "TENX153"]
    
    print(">>> 1. Loading Data...")
    samples = []
    for sample_id in id_list:
        try:
            s = CustomSample(root, sample_id)
            samples.append(s)
        except:
            pass
            
    # Find common genes across samples (must match loader.py logic)
    common_genes = set(samples[0].adata.var_names)
    for s in samples[1:]:
        common_genes = common_genes.intersection(set(s.adata.var_names))
    common_genes = sorted(list(common_genes))
    
    # Apply gene alignment to each sample
    for s in samples:
        s.adata = s.adata[:, common_genes]
        
    loader = create_hest_dataloader(samples, batch_size=2)
    
    # 2. Initialize model
    # num_genes must match the number of common genes (ex: 365)
    num_genes = len(common_genes)
    print(f">>> 2. Initializing Model with {num_genes} genes...")
    
    model = MultiModalHestModel(num_genes=num_genes, num_classes=2)
    
    # 3. Forward pass test
    print(">>> 3. Running Forward Pass (Sanity Check)...")
    model.train()
    
    try:
        batch = next(iter(loader))
        imgs = batch["image"]
        exprs = batch["expr"]
        labels = batch["label"]
        
        # Forward pass
        outputs = model(imgs, exprs)
        
        print("\n[Result]")
        print(f" - Input Image: {imgs.shape}")
        print(f" - Input Expr : {exprs.shape}")
        print(f" - Output Logits: {outputs.shape}")  # should be (B, 2)
        
        # Verify loss calculation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        print(f" - Loss calculation success: {loss.item()}")
        
        print("\n✅ System Check Passed! Model is structurally correct.")
        
    except Exception as e:
        print(f"\n❌ Error Occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_sanity_check()
