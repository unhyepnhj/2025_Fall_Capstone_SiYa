import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np

import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset.loader import CustomSample, create_wsi_dataloader
from models.model import MultiModalMILModel

# ===============================================
# YAML Config Loader 
# ===============================================
def load_config(path="configs/train.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    CONFIG = {
        # Data
        "root_dir": cfg["data"]["root_dir"],
        "max_spots": cfg["data"]["max_spots"],

        # Model
        "num_genes": cfg["model"]["num_genes"],
        "num_classes": cfg["model"]["num_classes"],
        "embed_dim": cfg["model"]["embed_dim"],
        "fusion_option": cfg["model"]["fusion_option"],
        "top_k_genes": cfg["model"].get("top_k_genes"),

        # Training
        "epochs": cfg["training"]["epochs"],
        "lr": cfg["training"]["lr"],
        "weight_decay": cfg["training"]["weight_decay"],
        "batch_size": cfg["training"]["batch_size"],

        # Memory
        "batch_spots": cfg["memory"]["batch_spots"],
        "accum_steps": cfg["memory"]["accum_steps"],
        "freeze_image_encoder": cfg["memory"]["freeze_image_encoder"],

        # Misc
        "device": cfg["misc"]["device"],
        "seed": cfg["misc"]["seed"],
        "checkpoint_freq": cfg["misc"]["checkpoint_freq"],
    }
    return CONFIG


# ===============================================
# Utils
# ===============================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

def plot_confusion_matrix(
  cm, 
  class_names=('0', '1'),
  title="Confusion Matrix",
  save_path = None
):
    """
    cm: np.array shape (2, 2) [[TN, FP]. [FN, TP]]
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)

    # ticks / labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # 숫자 표시
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                fontsize=12
            )
    
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

# ===============================================
# Data Split
# ===============================================
def prepare_data_splits(root_dir, seed=42):
    st_dir = os.path.join(root_dir, "global_hvg_unified")
    patch_dir = os.path.join(root_dir, "patches")

    st_files = {f.replace('.h5ad', '') for f in os.listdir(st_dir) if f.endswith('.h5ad')}
    patch_files = {f.replace('.h5', '') for f in os.listdir(patch_dir) if f.endswith('.h5')}
    valid_ids = sorted(st_files & patch_files)

    print(f"Found {len(valid_ids)} valid samples")

    samples, labels = [], []
    for sid in valid_ids:
        try:
            sample = CustomSample(root_dir, sid)
            
            # Binary classification만 사용
            if sample.label in [0, 1]:
                samples.append(sample)
                labels.append(sample.label)
            else:
                print(f"⚠️  Skipping {sid} (label={sample.label})")
                
        except Exception as e:
            print(f"Failed to load {sid}: {e}")

    from collections import Counter
    label_counts = Counter(labels)
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} ({100*count/len(labels):.1f}%)")

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.3,
        stratify=labels,
        random_state=seed
    )

    print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples


# ===============================================
# Training / Validation 
# ===============================================
def train_epoch(model, loader, criterion, optimizer, scaler, config, device):
    model.train()
    if config["freeze_image_encoder"]:
        model.img_encoder.eval()

    epoch_loss, correct = 0.0, 0
    optimizer.zero_grad()

    loop = tqdm(loader, desc="Training")

    for step, batch in enumerate(loop):
        images = batch["images"]
        expr = batch["expr"]
        coords = batch["coords"]
        label = batch["label"].to(device)

        N = images.size(0)
        spot_embeds_list = []

        for i in range(0, N, config["batch_spots"]):
            j = min(i + config["batch_spots"], N)

            img_b = images[i:j].to(device)
            expr_b = expr[i:j].to(device)
            coord_b = coords[i:j].to(device)

            with autocast():
                if config["freeze_image_encoder"]:
                    with torch.no_grad():
                        img_feat = model.img_encoder(img_b)
                else:
                    img_feat = model.img_encoder(img_b)

                st_feat = model.st_encoder(expr_b, coord_b)
                fused = model.fusion(img_feat, st_feat)

            spot_embeds_list.append(fused.detach().cpu())

            del img_b, expr_b, coord_b, img_feat, st_feat, fused
            torch.cuda.empty_cache()

        spot_embeds = torch.cat(spot_embeds_list, dim=0).to(device)

        with autocast():
            wsi_embed, _ = model.mil_pooling(spot_embeds)
            logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss = loss / config["accum_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % config["accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * config["accum_steps"]
        correct += int(logits.argmax().item() == label.item())

        loop.set_postfix(
            loss=f"{epoch_loss/(step+1):.4f}",
            acc=f"{100*correct/(step+1):.1f}%"
        )

        del spot_embeds, wsi_embed, logits, loss
        torch.cuda.empty_cache()

    return epoch_loss / len(loader), 100 * correct / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, config, device):
    model.eval()
    val_loss, correct = 0.0, 0

    y_true = []
    y_score = []  # prob of class 1
    y_pred = []   # predicted label (0/1)

    for batch in tqdm(loader, desc="Validation"):
        images = batch["images"]
        expr = batch["expr"]
        coords = batch["coords"]
        label = batch["label"].to(device)

        spot_embeds_list = []
        for i in range(0, images.size(0), config["batch_spots"]):
            j = min(i + config["batch_spots"], images.size(0))

            with autocast():
                img_feat = model.img_encoder(images[i:j].to(device))
                st_feat = model.st_encoder(expr[i:j].to(device), coords[i:j].to(device))
                fused = model.fusion(img_feat, st_feat)

            spot_embeds_list.append(fused.cpu())

        spot_embeds = torch.cat(spot_embeds_list, dim=0).to(device)

        with autocast():
            wsi_embed, _ = model.mil_pooling(spot_embeds)
            logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))

        val_loss += loss.item()
        pred = logits.argmax().item()
        correct += int(pred == label.item())

        # ROC/AUC
        prob_pos = torch.softmax(logits, dim=0)[1].item()
        y_true.append(label.item())
        y_score.append(prob_pos)
        y_pred.append(pred)

    val_loss = val_loss / len(loader)
    val_acc = 100 * correct / len(loader)

    # AUC
    auc = float('nan')
    try:
      auc = roc_auc_score(y_true, y_score)
      # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    except Exception:
      pass
    
    # precision/recall/f1 (class 1=positive로)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    p, r, f1 = float(p), float(r), float(f1)

    # confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels = [0, 1])

    return val_loss, val_acc, auc, p, r, f1, cm


# ===============================================
# Main
# ===============================================
def main():
    CONFIG = load_config("configs/train.yaml")   
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])

    print("="*70)
    print("Multi-Modal MIL Training")
    print("="*70)
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print("="*70 + "\n")

    train_samples, val_samples = prepare_data_splits(
        CONFIG["root_dir"], CONFIG["seed"]
    )

    train_loader = create_wsi_dataloader(
        train_samples, 1, True, CONFIG["max_spots"], CONFIG["root_dir"]
    )
    val_loader = create_wsi_dataloader(
        val_samples, 1, False, CONFIG["max_spots"], CONFIG["root_dir"]
    )

    model = MultiModalMILModel(
        num_genes=CONFIG["num_genes"],
        num_classes=CONFIG["num_classes"],
        embed_dim=CONFIG["embed_dim"],
        fusion_option=CONFIG["fusion_option"],
        top_k_genes=CONFIG.get("top_k_genes"),
    ).to(device)

    if CONFIG["freeze_image_encoder"]:
        for p in model.img_encoder.parameters():
            p.requires_grad = False
        model.img_encoder.eval()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, CONFIG, device
        )
        val_loss, val_acc, val_auc, val_p, val_r, val_f1, cm = validate(
            model, val_loader, criterion, CONFIG, device
        )

        print(
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f} | "
            f"P/R/F1: {val_p:.3f}/{val_r:.3f}/{val_f1:.3f}"
        )

        if val_acc > best_val_acc:  # best model
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_attn.pt")
            print(f"✓ Saved best model to: {ckpt_path} (val_acc={val_acc:.2f}%)")

            # confusion matrix -> best model일 때만 plot
            plot_confusion_matrix(
                cm,
                class_names=("0", "1"),   # Healthy / Cancer면 바꿔도 됨
                title=f"Val Confusion Matrix (Epoch {epoch+1})",
                save_path=f"confusion_matrix_epoch_{epoch+1}.png"
            )

    print("\n TRAINING COMPLETE!!")


if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    main()