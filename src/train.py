import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import scanpy as sc
from collections import Counter
from tqdm import tqdm
import gc

from loader import CustomSample, create_wsi_dataloader
from model import MultiModalMILModel

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
CONFIG = {
    # "root_dir": "/workspace/Temp/ver1/hest_data",
    "root_dir": "/content/hest_data",
    "epochs": 50,
    "lr": 1e-4,
    "embed_dim": 64,
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vocab_size": 500,     
    "max_spots": 1000,      
    "batch_spots": 50,      # Move only 50 spots to GPU at a time
    "accum_steps": 8,       # Gradient accumulation steps
}

# -------------------------------------------------------
# Top-K gene list
# -------------------------------------------------------
def get_top_k_gene_list(samples, k=500):
    print(f"Scanning genes from all samples to find Top-{k}...")
    gene_counter = Counter()
    
    for s in tqdm(samples[:20], desc="Scanning Genes"):  # Scan only first 20 samples
        try:
            adata = sc.read_h5ad(s.st_path, backed='r')
            gene_counter.update(adata.var_names)
            del adata
        except Exception as e:
            print(f"Skipping {s.sample_id}: {e}")
    
    most_common = gene_counter.most_common(k)
    top_genes = sorted([gene for gene, _ in most_common])
    
    print(f" -> Selected {len(top_genes)} genes.")
    return top_genes

# -------------------------------------------------------
# Training Function (memory-optimized version)
# -------------------------------------------------------
def train(cfg):
    device = cfg["device"]
    torch.cuda.empty_cache()
    
    # 1. Load samples
    st_dir = os.path.join(cfg["root_dir"], "st_preprocessed")
    all_files = [f for f in os.listdir(st_dir) if f.endswith(".h5ad")]
    
    samples = []
    print(f"Loading {len(all_files)} sample headers...")
    for fname in tqdm(all_files):
        try:
            sid = fname.replace(".h5ad", "")
            samples.append(CustomSample(cfg["root_dir"], sid))
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
    
    print(f"Successfully loaded {len(samples)} samples")
    
    # 2. Get Top-K genes
    target_genes = get_top_k_gene_list(samples, k=cfg["vocab_size"])
    num_genes = len(target_genes)
    
    # 3. DataLoader
    loader = create_wsi_dataloader(
        samples,
        batch_size=1,
        shuffle=True,
        max_spots=cfg["max_spots"],
        target_genes=target_genes
    )
    
    # 4. Model
    model = MultiModalMILModel(
        num_genes=num_genes,
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
        head_use_ln=True    # <- 추가!!
    ).to(device)
    
    # model.py에 encoder freeze + eval 고정 함수 추가했음
    if hasattr(model, 'freeze_encoders'):
        model.freeze_encoders()

    # 인코더 제외 학습 파라미터 명시적으로 추가 (optional)
    trainable_params = (
        list(model.img_head.parameters()) +
        list(model.st_encoder.parameters()) +  # <- ST 인코더는 학습
        # list(model.st_head.parameters()) +
        list(model.fusion.parameters()) +
        list(model.mil_pooling.parameters()) +
        list(model.classifier.parameters())
    )

    optimizer = optim.AdamW(
        # trainable_params, # <- 필요 시 추가!!
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    # 5. Training loop
    print("\n=== Start Training (Memory-Safe Mode, Encoders Frozen) ===")
    
    for epoch in range(cfg["epochs"]):
        model.train()   # model.train() 내에서 인코더 eval 고정

        total_loss = 0
        correct = 0
        optimizer.zero_grad()
        
        loop = tqdm(loader, desc=f"Ep {epoch+1}/{cfg['epochs']}")
        
        for step, batch in enumerate(loop):
            if batch is None:
                continue
            
            try:
                # Prepare data on CPU
                images_cpu = batch["images"]
                expr_cpu = batch["expr"]
                coords_cpu = batch["coords"]
                label = batch["label"].unsqueeze(0).to(device)
                
                N = images_cpu.size(0)
                batch_spots = cfg["batch_spots"]
                
                # Core idea: process spot embeddings in small batches
                spot_embeds_list = []
                
                for i in range(0, N, batch_spots):
                    j = min(i + batch_spots, N)
                    
                    # Move to GPU
                    img_batch = images_cpu[i:j].to(device)
                    expr_batch = expr_cpu[i:j].to(device)
                    coord_batch = coords_cpu[i:j].to(device)
                    
                    with autocast('cuda'):
                        # Do not track gradients for images
                        with torch.no_grad():
                            img_feat = model.img_encoder(img_batch)
                        
                        # ST encoder is trainable
                        img_feat = model.img_head(img_feat)
                        st_feat = model.st_encoder(expr_batch, coord_batch)
                        fusion = model.fusion(img_feat, st_feat)
                    
                    # Move back to CPU and store (save GPU memory)
                    spot_embeds_list.append(fusion.detach().cpu())
                    
                    # Immediately free GPU memory
                    del img_batch, expr_batch, coord_batch, img_feat, st_feat, fusion
                    torch.cuda.empty_cache()
                
                # Concatenate on CPU, then move to GPU once
                spot_embeds = torch.cat(spot_embeds_list, dim=0).to(device)
                
                # MIL pooling & loss
                with autocast('cuda'):
                    wsi_embed, _ = model.mil_pooling(spot_embeds)
                    logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
                    loss = criterion(logits.unsqueeze(0), label)
                    loss = loss / cfg["accum_steps"]
                
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % cfg["accum_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Statistics
                total_loss += loss.item() * cfg["accum_steps"]
                pred = logits.argmax().item()
                correct += (pred == label.item())
                
                # Memory cleanup
                del spot_embeds, wsi_embed, logits, loss
                torch.cuda.empty_cache()
                
                loop.set_postfix(
                    acc=f"{100*correct/(step+1):.1f}%",
                    loss=f"{total_loss/(step+1):.4f}"
                )
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ OOM at step {step}! Skipping batch...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / len(loader)
        avg_acc = 100 * correct / len(loader)
        print(f"\nEpoch {epoch+1} Done | Loss: {avg_loss:.4f} | Acc: {avg_acc:.1f}%")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_epoch_{epoch+1}.pt")
            print(f"✓ Checkpoint saved")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    train(CONFIG)