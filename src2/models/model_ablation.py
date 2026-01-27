import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.performer_pytorch import Performer

# =======================================================
# 1. Image Encoder (Patch → Spot-level visual feature)
# =======================================================
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, x):
        """
        x: (N_spots, 3, 224, 224)
        return: (N_spots, embed_dim)
        """
        return self.backbone(x)


# =======================================================
# 2. Spatial ST Encoder (HVG-only, scBERT-style)
# =======================================================
class SpatialSTEncoder(nn.Module):
    """
    scBERT-style encoder with explicit spatial token

    Input:
      - expr   : (N_spots, K)   [already HVG-filtered]
      - coords : (N_spots, 2)   [normalized]

    Output:
      - (N_spots, embed_dim) spot-level ST embedding
    """

    def __init__(
        self,
        num_genes,        # K = number of HVGs (e.g., 2000)
        embed_dim=256,
        num_layers=2,
        num_heads=4,
        top_k_genes=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_genes = num_genes
        self.top_k_genes = top_k_genes

        # Gene identity embedding (HVG-only)
        self.gene_embedding = nn.Embedding(num_genes, embed_dim)

        # Gene positional embedding (gene order)
        self.gene_pos_embedding = nn.Embedding(num_genes, embed_dim)

        # Expression value embedding
        self.value_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Spatial token embedding
        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Performer (efficient transformer)
        self.transformer = Performer(
            dim=embed_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=embed_dim // num_heads,
            causal=False,
            ff_mult=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        # Spatial-query pooling
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, expr, coords, return_gene_attn=False):
        """
        expr   : (N, K)
        coords : (N, 2)

        if return_gene_attn:
          returns:
            pooled: (N, D)
            gene_attn: (N, G_used)          # G_used = top_k_genes or K
            gene_indices: (N, G_used) (long) # global gene ids
        else:
          returns:
            pooled: (N, D)
        """
        N, K = expr.shape
        device = expr.device

        # ✅ Top-K gene selection (메모리 절약)
        if self.top_k_genes and self.top_k_genes < K:
            # 각 spot에서 expression 값이 높은 상위 K개만 선택
            topk_values, topk_indices = torch.topk(expr, k=self.top_k_genes, dim=1)
            gene_indices = topk_indices.long()  # global gene ids

            gene_embed = self.gene_embedding(gene_indices)  # (N, top_k, D)
            gene_pos = self.gene_pos_embedding(gene_indices)
            value_emb = self.value_embedding(topk_values.unsqueeze(-1))
            gene_tokens = gene_embed + gene_pos + value_emb
        else:
            # 원래 방식: 모든 gene 사용
            gene_ids = torch.arange(K, device=device).unsqueeze(0).expand(N, -1).long()
            gene_embed = self.gene_embedding(gene_ids)
            gene_pos = self.gene_pos_embedding(gene_ids)
            value_emb = self.value_embedding(expr.unsqueeze(-1))
            gene_tokens = gene_embed + gene_pos + value_emb

        spatial_token = self.spatial_embedding(coords).unsqueeze(1)
        tokens = torch.cat([spatial_token, gene_tokens], dim=1)

        tokens = self.transformer(tokens)

        spatial_out = tokens[:, :1]
        gene_out    = tokens[:, 1:]

        q = self.q_proj(spatial_out)
        k = self.k_proj(gene_out)
        v = self.v_proj(gene_out)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5),
            dim=-1
        )

        pooled = torch.matmul(attn, v).squeeze(1)
        pooled = self.out_proj(pooled)

        if return_gene_attn:
            gene_attn = attn.squeeze(1)
            return pooled, gene_attn, gene_indices
        else:
            return pooled


# =======================================================
# 3. Spot Fusion Module (4 options: concat, attn, sim, gate)
# =======================================================
class SpotFusionModule(nn.Module):
    """
    Fusion options:
    - 'concat': Simple concatenation + MLP
    - 'attn': Cross-attention between img and st
    - 'sim': Similarity-based fusion (cosine, product, diff)
    - 'gate': Gated fusion with learnable weights
    """
    def __init__(
        self,
        embed_dim=256,
        fusion_option='concat',
        attn_heads=4,
        dropout=0.2,
        use_l2norm_for_sim=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_option = fusion_option
        self.dropout = dropout
        self.use_l2norm_for_sim = use_l2norm_for_sim
        
        # Pre-normalization
        self.pre_norm_img = nn.LayerNorm(embed_dim)
        self.pre_norm_st = nn.LayerNorm(embed_dim)

        if fusion_option == 'concat':
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif fusion_option == 'attn':
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        elif fusion_option == 'sim':
            # 4D + 1 = img, st, product, abs_diff, cosine_sim
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 4 + 1, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif fusion_option == 'gate':
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
                nn.Softmax(dim=-1),
            )
            self.proj = nn.Linear(embed_dim, embed_dim)
        
        else:
            raise ValueError(f"Unknown fusion_option: {fusion_option}")

    def forward(self, img_feat, st_feat):
        """
        img_feat: (N, D)
        st_feat: (N, D)
        return: (N, D)
        """
        # Pre-norm
        img_feat = self.pre_norm_img(img_feat)
        st_feat = self.pre_norm_st(st_feat)

        if self.fusion_option == 'concat':
            # Simple concatenation
            x = torch.cat([img_feat, st_feat], dim=-1)  # (N, 2D)
            return self.fuse(x)  # (N, D)

        elif self.fusion_option == 'attn':
            # Cross-attention: [img, st] as 2 tokens
            tokens = torch.stack([img_feat, st_feat], dim=1)  # (N, 2, D)
            
            # Self-attention
            attn_out, _ = self.attn(tokens, tokens, tokens)  # (N, 2, D)
            tokens = self.norm1(tokens + attn_out)
            
            # FFN
            ffn_out = self.ffn(tokens)  # (N, 2, D)
            tokens = self.norm2(tokens + ffn_out)
            
            # Pool (average)
            pooled = tokens.mean(dim=1)  # (N, D)
            return self.out_proj(pooled)

        elif self.fusion_option == 'sim':
            # Similarity-based features
            if self.use_l2norm_for_sim:
                img_n = F.normalize(img_feat, p=2, dim=-1, eps=1e-8)
                st_n = F.normalize(st_feat, p=2, dim=-1, eps=1e-8)
            else:
                img_n = img_feat
                st_n = st_feat
            
            # Cosine similarity
            sim = F.cosine_similarity(img_n, st_n, dim=-1, eps=1e-8).unsqueeze(-1)  # (N, 1)
            
            # Element-wise product
            prod = img_n * st_n  # (N, D)
            
            # Absolute difference
            abs_diff = torch.abs(img_n - st_n)  # (N, D)
            
            # Concatenate all features
            x = torch.cat([img_n, st_n, prod, abs_diff, sim], dim=-1)  # (N, 4D+1)
            return self.fuse(x)  # (N, D)

        elif self.fusion_option == 'gate':
            # Gated fusion
            x = torch.cat([img_feat, st_feat], dim=-1)  # (N, 2D)
            weights = self.gate(x)  # (N, 2)
            
            # Weighted sum
            fused = weights[:, 0:1] * img_feat + weights[:, 1:2] * st_feat  # (N, D)
            return self.proj(fused)  # (N, D)


# =======================================================
# 4. MIL Attention Pooling (Spot → WSI)
# =======================================================
class MILAttentionPooling(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=128):
        super().__init__()
        self.attn_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, spot_embeds):
        """
        spot_embeds: (N_spots, D)
        """
        A = self.attn_w(self.attn_V(spot_embeds) * self.attn_U(spot_embeds))
        weights = F.softmax(A, dim=0)
        wsi_embed = torch.sum(weights * spot_embeds, dim=0)
        return wsi_embed, weights


# =======================================================
# 5. Linear Head (Image Encoder 뒤에 붙일 FC layer)
# =======================================================
class LinearHead(nn.Module):
    def __init__(self, dim: int, use_ln: bool=True):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.fc = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.ln(x))
    
    
# =======================================================
# 6. Full Multi-Modal MIL Model (Freeze 지원)
# =======================================================
class MultiModalMILModel(nn.Module):
    def __init__(
        self,
        num_genes=2000,
        num_classes=2,
        embed_dim=256,
        fusion_option='concat',
        top_k_genes=None,
        dropout=0.3,
        freeze_image_encoder=True,
        mil_hidden_dim=128,
        mil_dropout=0.0,
        fusion_dropout=0.2,
        head_use_ln=True,

        # ablation용
        use_image =True,
        use_st=True
    ):
        super().__init__()

        assert use_image or use_st, "At least one modality must be used!!"
        
        self.use_image = use_image
        self.use_st = use_st
        self.fusion_option = fusion_option
        self.freeze_image_encoder = freeze_image_encoder

        # Ablation: 조건부 인코더 생성
        if self.use_image:
            self.img_encoder = ImageEncoder(embed_dim)
            self.img_head = LinearHead(dim=embed_dim, use_ln=head_use_ln)
        else:
            self.img_encoder = None
            self.img_head = None

        if self.use_st:
            self.st_encoder = SpatialSTEncoder(
                num_genes=num_genes,
                embed_dim=embed_dim,
                top_k_genes=top_k_genes,
            )
        else:
            self.st_encoder = None
        self.st_head = nn.Identity()

        # Freeze 적용
        if self.use_image and freeze_image_encoder:
            self.freeze_encoders()
            
        # Fusion
        # Ablation: multimodal일 때만 fusion 생성
        if self.use_image and self.use_st:
            self.fusion = SpotFusionModule(
                embed_dim=embed_dim,
                fusion_option=fusion_option,
                dropout=fusion_dropout,
            )
        else:
            self.fusion = None

        # MIL Pooling
        self.mil_pooling = MILAttentionPooling(
            embed_dim=embed_dim,
            hidden_dim=mil_hidden_dim,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        print(f"✓ Model 1 initialized with fusion_option='{fusion_option}'")
        if freeze_image_encoder:
            print(f"✓ Image Encoder frozen (only img_head trainable)")

    def freeze_encoders(self):
        """ResNet backbone만 freeze"""
        if self.img_encoder is None:    # Ablation용 수정
            return
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.img_encoder.eval()
    
    def train(self, mode: bool=True):
        """Training 모드에서도 Image Encoder는 eval 유지"""
        super().train(mode)
        if self.use_image and self.freeze_image_encoder:
            self.img_encoder.eval()

    def forward(self, images, expr, coords, return_gene_attn=True, return_spot_embeds=True):
        """
        images: (N_spots, 3, 224, 224)
        expr  : (N_spots, K)
        coords: (N_spots, 2)
        
        Returns:
          logits: (num_classes,)
          attn: (N_spots, 1)
        """
        spot_embeds = None

        # Ablation: 조건부 인코딩
        if self.use_image:  # img branch
            if self.freeze_image_encoder:
                with torch.no_grad():
                    img_feat = self.img_encoder(images)
            else:
                img_feat = self.img_encoder(images)
            img_feat = self.img_head(img_feat)  # FC layer (trainable)

        if self.use_st:     # st branch
            if return_gene_attn:
                st_feat, gene_attn, gene_indices = self.st_encoder(expr, coords, return_gene_attn=True)
            else:
                st_feat = self.st_encoder(expr, coords, return_gene_attn=False)
                gene_attn, gene_indices = None, None
        
        # Ablation: modality별 spot embedding 처리
        if self.use_image and self.use_st:  # Both modalities: Fusion
            spot_embeds = self.fusion(img_feat, st_feat)
        elif self.use_image:    # Image only
            spot_embeds = img_feat
        elif self.use_st:       # ST only
            spot_embeds = st_feat

        # 이후 동일 ==================================================

        # MIL Pooling
        wsi_embed, mil_attn = self.mil_pooling(spot_embeds)
        mil_attn = mil_attn.squeeze(-1)  # (N_spots,)

        # Classification
        logits = self.classifier(wsi_embed)
        
        out = {
            "logits": logits,
            "mil_attn": mil_attn,
            "gene_attn": gene_attn,
            "gene_indices": gene_indices,
        }
        if return_spot_embeds:
            out["spot_embeds"] = spot_embeds  # <- 추가

        return out