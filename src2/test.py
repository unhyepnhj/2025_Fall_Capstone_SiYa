import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from dataset.loader import CustomSample, create_wsi_dataloader, load_global_gene_order
from models.model import MultiModalMILModel

"""
- wsi에서 중요도 높은 패치 보여주기(img)
- 중요도 높은 gene들 뽑아주기(expr)
- 중요도 높은 스팟 보여주기 (st)
"""

# Detect samples -> return 
def discover_samples(root_dir):
    st_dir = os.path.join(root_dir, "st_preprocessed_global_hvg")
    patch_dir = os.path.join(root_dir, "patches")

    assert os.path.isdir(st_dir), f"ST dir not found: {st_dir}"
    assert os.path.isdir(patch_dir), f"Patch dir not found: {patch_dir}"

    sample_ids = []
    for fn in os.listdir(st_dir):
        if fn.endswith(".h5ad"):
            sid = fn[:-5]
            if os.path.exists(os.path.join(patch_dir, f"{sid}.h5")):
                sample_ids.append(sid)
    sample_ids.sort()

    samples = [CustomSample(root_dir, sid) for sid in sample_ids]
    return samples

# UMAP/PCA util
def compute_2d_embedding(X: np.ndarray, method: str = "umap", seed: int = 0):
    """
    X: (N, D)
    returns: (N, 2)
    """
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed)
            return reducer.fit_transform(X)
        except Exception as e:
            print(f"[warn] UMAP not available ({e}). Falling back to PCA.")
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)

    raise ValueError(f"Unknown method: {method}")

# IO utils
def save_patch_image(tensor_chw, out_path):
    """
    tensor_chw: torch.Tensor (3,H,W), range ~[0,1]
    """
    x = tensor_chw.detach().cpu().clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(x).save(out_path)

# attention score 높은 patch plot
def plot_attention_scatter(coords_raw, attn, top10_idx, out_path, title="Patch importance (MIL attn)"):
    """
    coords_raw: (N,2) torch.Tensor  (original spatial coords)
    attn: (N,) torch.Tensor
    top10_idx: list[int]
    """
    c = coords_raw.detach().cpu().numpy()
    a = attn.detach().cpu().numpy()

    a_min, a_max = float(a.min()), float(a.max())
    denom = (a_max - a_min) if (a_max - a_min) > 1e-12 else 1.0
    a_n = (a - a_min) / denom
    sizes = 10 + 200 * a_n

    plt.figure()
    plt.scatter(c[:, 0], c[:, 1], s=sizes)  # 색 지정 안 함

    if top10_idx is not None and len(top10_idx) > 0:
        sel = np.array(top10_idx, dtype=np.int64)
        plt.scatter(c[sel, 0], c[sel, 1], s=250, marker="x")
        for rank, i in enumerate(sel.tolist(), start=1):
            plt.text(c[i, 0], c[i, 1], f"Top{rank}", fontsize=10)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()  # 필요 없으면 제거
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# 중요도 높은 gene 집계
def aggregate_top_genes(gene_attn, gene_indices, mil_attn, gene_order, topk=30):
    """
    gene_attn: (N, G) torch.Tensor (per-spot attention over gene tokens)
    gene_indices: (N, G) torch.LongTensor
    mil_attn: (N,) torch.Tensor  (spot importance)
    gene_order: list[str] length K_global
    """
    N, G = gene_attn.shape
    mil_w = mil_attn.view(N, 1)

    contrib = (gene_attn * mil_w).detach().cpu().numpy()  # (N,G)
    gidx = gene_indices.detach().cpu().numpy().astype(np.int64)

    scores = {}  # gene_id -> sum score
    for i in range(N):
        for j in range(G):
            gid = int(gidx[i, j])
            scores[gid] = scores.get(gid, 0.0) + float(contrib[i, j])

    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    out = []
    for gid, sc in items:
        gname = gene_order[gid] if (0 <= gid < len(gene_order)) else f"gene_{gid}"
        out.append((gname, sc))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./xai_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model params (일단 train과 통일)
    parser.add_argument("--num_genes", type=int, default=2000)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--fusion_option", type=str, default="concat")  # <- ver2_concat이 best여서 concat으로 진행
    parser.add_argument("--top_k_genes", type=int, default=512)
    parser.add_argument("--freeze_image_encoder", action="store_true")

    # data params
    parser.add_argument("--max_spots", type=int, default=2000)
    parser.add_argument("--topk_patches", type=int, default=12)
    parser.add_argument("--topk_genes", type=int, default=30)

    args = parser.parse_args(args=[
    "--root_dir", r"C:\Users\rdh08\Desktop\Capstone\src2\hest_data",
    "--ckpt", r"C:\Users\rdh08\Desktop\Capstone\src2\best_model_concat.pt",
])

    os.makedirs(args.out_dir, exist_ok=True)

    # samples + loader
    samples = discover_samples(args.root_dir)
    if len(samples) == 0:
        raise RuntimeError("No samples discovered. Check root_dir structure.")
    loader = create_wsi_dataloader(
        samples,
        batch_size=1,
        shuffle=False,
        max_spots=args.max_spots,
        root_dir=args.root_dir,
        return_trace=True,  # loader 수정본 기준
    )

    # gene order (fallback)
    global_gene_order = load_global_gene_order(args.root_dir)
    if global_gene_order is None:
        global_gene_order = []

    # model
    model = MultiModalMILModel(
        num_genes=args.num_genes,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        fusion_option=args.fusion_option,
        top_k_genes=args.top_k_genes,
        freeze_image_encoder=args.freeze_image_encoder,

    ).to(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.eval()

    with torch.inference_mode():
        for batch in loader:
            sample_id = batch["sample_id"]
            out_dir = os.path.join(args.out_dir, sample_id)
            os.makedirs(out_dir, exist_ok=True)

            images = batch["images"].to(args.device)     # (N,3,H,W)
            expr = batch["expr"].to(args.device)         # (N,K)
            coords = batch["coords"].to(args.device)     # (N,2) normalized (model input)
            label = int(batch["label"].item())

            # XAI metadata (loader 수정본 기준 key들)
            barcodes = batch.get("barcodes", None)            # list[str]
            patch_indices = batch.get("patch_indices", None)  # np.ndarray
            coords_raw = batch.get("coords_raw", None)        # torch.Tensor (N,2)
            gene_order = batch.get("gene_order", global_gene_order)

            # forward
            outputs = model(
                images, 
                expr, 
                coords, 
                return_gene_attn=True, 
                return_spot_embeds=True
            )

            spot_embeds = outputs["spot_embeds"]  # (N, D)
            mil_attn = outputs["mil_attn"]      # (N, )

            # UMAP 코드
            X = spot_embeds.detach().cpu().numpy()
            Z = compute_2d_embedding(X, method="umap", seed=0)  # umap or pca

            a = mil_attn.detach().cpu().numpy()
            a_min, a_max = float(a.min()), float(a.max())
            denom = (a_max - a_min) if (a_max - a_min) > 1e-12 else 1.0
            a_n = (a - a_min) / denom
            sizes = 10 + 200 * a_n

            # 단색 ver(size only)
            plt.figure()
            plt.scatter(Z[:, 0], Z[:, 1], s=sizes)  # 색 지정 안 함
            plt.title("UMAP of fused spot embeddings(size=mil_attn)")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "spot_embeds_umap.png"), dpi=200)
            plt.close()

            # 색 + 크기 모두
            plt.figure()
            plt.scatter(
                Z[:, 0], Z[:, 1],
                c=a_n,
                s=10+200*a_n, 
                cmap="viridis"
            )
            plt.colorbar(label="MIL attention")
            plt.title("UMAP of fused spot embeddings(color+size=mil_attn)")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "spot_embeds_umap_color.png"), dpi=200)
            plt.close()

            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("out", None))
                if logits is None:
                    raise ValueError("Model dict output must contain 'logits' (or 'out').")
                mil_attn = outputs.get("mil_attn", None)
                gene_attn = outputs.get("gene_attn", None)
                gene_indices = outputs.get("gene_indices", None)
            elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                logits, mil_attn = outputs[0], outputs[1]
                gene_attn, gene_indices = None, None
            else:
                logits = outputs
                mil_attn = None
                gene_attn, gene_indices = None, None

            probs = F.softmax(logits, dim=-1).detach().cpu().numpy().tolist()
            pred = int(np.argmax(probs))
            n_spots = int(images.shape[0])

            # attention이 없으면 XAI가 불가하므로 최소한 pred만 저장
            if mil_attn is None:
                summary = {
                    "sample_id": sample_id,
                    "gt_label": label,
                    "pred_label": pred,
                    "probs": probs,
                    "num_spots_used": n_spots,
                    "mil_attn_available": False,
                    "note": "Model did not return mil_attn. Patch ranking/heatmap skipped."
                }
                with open(os.path.join(out_dir, "pred.json"), "w") as f:
                    json.dump(summary, f, indent=2)
                continue

            mil_attn = mil_attn.view(-1)

            # Top-10 important spots
            top10 = torch.topk(mil_attn, k=min(10, n_spots)).indices.detach().cpu().tolist()

            summary = {
                "sample_id": sample_id,
                "gt_label": label,
                "pred_label": pred,
                "probs": probs,
                "num_spots_used": n_spots,
                "top10_spot_indices": top10,
                "barcodes_available": barcodes is not None,
                "patch_indices_available": patch_indices is not None,
                "coords_raw_available": coords_raw is not None,
                "gene_xai_available": (gene_attn is not None and gene_indices is not None),
            }
            with open(os.path.join(out_dir, "pred.json"), "w") as f:
                json.dump(summary, f, indent=2)

            # (1) Top-k patches 저장
            k = min(args.topk_patches, n_spots)
            topk = torch.topk(mil_attn, k=k).indices.detach().cpu().tolist()

            patch_dir = os.path.join(out_dir, "top_patches")
            os.makedirs(patch_dir, exist_ok=True)

            for rank, i in enumerate(topk, start=1):
                fn = f"rank{rank:02d}_idx{i}"
                if barcodes is not None:
                    fn += f"_bc{barcodes[i]}"
                if patch_indices is not None:
                    fn += f"_pidx{int(patch_indices[i])}"
                fn += ".png"
                save_patch_image(images[i], os.path.join(patch_dir, fn))

            # (2) attention scatter (coords_raw 기반)
            if coords_raw is not None:
                plot_attention_scatter(
                    coords_raw=coords_raw,
                    attn=mil_attn.detach().cpu(),
                    top10_idx=top10,
                    out_path=os.path.join(out_dir, "patch_attn_scatter_top10.png"),
                    title="Patch importance (MIL attn) + Top10"
                )

            # (3) gene top list (가능할 때만)
            if gene_attn is not None and gene_indices is not None and len(gene_order) > 0:
                # gene_attn: (N,1,G) or (N,G)
                if gene_attn.dim() == 3:
                    gene_attn2 = gene_attn.squeeze(1)
                else:
                    gene_attn2 = gene_attn

                top_genes = aggregate_top_genes(
                    gene_attn=gene_attn2,
                    gene_indices=gene_indices,
                    mil_attn=mil_attn.detach().cpu(),
                    gene_order=gene_order,
                    topk=args.topk_genes
                )
                with open(os.path.join(out_dir, "top_genes.csv"), "w") as f:
                    f.write("rank,gene,score\n")
                    for r, (g, sc) in enumerate(top_genes, start=1):
                        f.write(f"{r},{g},{sc:.6f}\n")

    print(f"Done. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
