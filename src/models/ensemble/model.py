"""
Modelo temporal backbone-agnóstico para classificação de culturas.

Mesma arquitetura do V7 (FiLM + Temporal Attention) mas aceita qualquer
backbone do timm. O feature_dim e num_heads são detectados automaticamente.

Backbones suportados:
    efficientnet_b0  →  1280 features
    resnet50         →  2048 features
    convnext_tiny    →   768 features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# Configuração por backbone: quantas camadas descongelar no fine-tuning
BACKBONE_CONFIG = {
    'efficientnet_b0': {'fine_tune_layers': 20},
    'resnet50':        {'fine_tune_layers': 30},
    'convnext_tiny':   {'fine_tune_layers': 25},
}

MES_EMBED_DIM = 8


class TemporalCulturaModel(nn.Module):
    """
    [Backbone] + FiLM(dia, mes) + 2x Temporal Attention + Classification Head.

    Funciona com qualquer backbone do timm. A dimensão de features, FiLM,
    attention e head se adaptam automaticamente.
    """

    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone_name = backbone_name

        # Backbone (qualquer modelo timm, sem cabeça de classificação)
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0
        )
        self.feature_dim = self.backbone.num_features

        # Congelar backbone inicialmente
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Calcular num_heads compatível com feature_dim
        num_heads = 8
        while self.feature_dim % num_heads != 0:
            num_heads -= 1

        # FiLM conditioning: concat(dia, mes_emb) → gamma, beta
        self.mes_embedding = nn.Embedding(12, MES_EMBED_DIM)
        self.film_hidden = nn.Linear(1 + MES_EMBED_DIM, 64)
        self.film_gamma = nn.Linear(64, self.feature_dim)
        self.film_beta = nn.Linear(64, self.feature_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # 2 camadas de temporal self-attention
        self.attn1 = nn.MultiheadAttention(
            self.feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.attn2 = nn.MultiheadAttention(
            self.feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(self.feature_dim)

        # Cabeça de classificação
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _temporal_forward(self, images, dias, mes, mask):
        """Forward completo até o vetor pooled (antes da cabeça)."""
        B, T = images.shape[:2]

        # Backbone
        imgs_flat = images.reshape(B * T, *images.shape[2:])
        feats_flat = self.backbone(imgs_flat)
        features = feats_flat.reshape(B, T, -1)

        # FiLM
        mes_emb = self.mes_embedding(mes).unsqueeze(1).expand(-1, T, -1)
        context = torch.cat([dias.unsqueeze(-1), mes_emb], dim=-1)
        film_h = F.relu(self.film_hidden(context))
        gamma = self.film_gamma(film_h)
        beta = self.film_beta(film_h)
        tokens = features * (1.0 + gamma) + beta

        # Temporal attention
        key_pad_mask = (mask == 0)
        attn_out1, _ = self.attn1(tokens, tokens, tokens,
                                   key_padding_mask=key_pad_mask)
        tokens = self.norm1(tokens + attn_out1)
        attn_out2, _ = self.attn2(tokens, tokens, tokens,
                                   key_padding_mask=key_pad_mask)
        tokens = self.norm2(tokens + attn_out2)

        # Mean pooling sobre timesteps válidos
        mask_exp = mask.unsqueeze(-1)
        pooled = (tokens * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)

        return pooled

    def forward(self, images, dias, mes, mask):
        """Retorna logits para treinamento."""
        pooled = self._temporal_forward(images, dias, mes, mask)
        return self.head(pooled)

    def forward_features(self, images, dias, mes, mask):
        """Retorna (pooled, logits) para extração de features."""
        pooled = self._temporal_forward(images, dias, mes, mask)
        logits = self.head(pooled)
        return pooled, logits

    def descongelar_ultimas_camadas(self, n_camadas: int):
        """Descongela as últimas n_camadas do backbone para fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        all_params = list(self.backbone.named_parameters())
        for _, param in all_params[-n_camadas:]:
            param.requires_grad = True
