import math

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sig": nn.Sigmoid,
}


class YieldMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_size: int,
        activation: str,
        dropout_rate: float,
        use_batch_norm: bool,
    ):
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_cls())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _FeatureTokenizer(nn.Module):
    """Projects each tabular feature into a shared d_token-dimensional space.

    Numerical: token_i = x_i · W_i + b_i  (per-feature weight vector).
    Categorical: standard embedding lookup.
    """

    def __init__(self, n_num: int, cat_cardinalities: list[int], d_token: int):
        super().__init__()
        self.n_num = n_num
        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            self.num_bias = nn.Parameter(torch.zeros(n_num, d_token))
            nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in cat_cardinalities]
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num: (B, n_num)  x_cat: (B, n_cat)
        parts: list[torch.Tensor] = []
        if self.n_num > 0:
            # (B, n_num, 1) * (n_num, d_token) → (B, n_num, d_token)
            parts.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        for i, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_cat[:, i]).unsqueeze(1))  # (B, 1, d_token)
        return torch.cat(parts, dim=1)  # (B, n_num + n_cat, d_token)


class YieldFTTransformer(nn.Module):
    """FT-Transformer for tabular yield prediction (Gorishniy et al., 2021).

    Input contract: flat float32 tensor where the first n_num_features columns are
    continuous and the remaining len(cat_cardinalities) columns are integer-encoded
    categorical features (rm_id last by convention from dataset.encode_rm).

    The backbone (tokenizer + transformer + norm) is decoupled from the head so
    freeze_backbone / unfreeze_backbone support the VT → Planting Date transfer.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        d_token: int,
        n_heads: int,
        n_layers: int,
        d_ffn_factor: float,
        dropout: float,
    ):
        super().__init__()
        assert d_token % n_heads == 0, (
            f"d_token ({d_token}) must be divisible by n_heads ({n_heads})"
        )
        self.n_num = n_num_features
        self.n_cat = len(cat_cardinalities)

        self.tokenizer = _FeatureTokenizer(n_num_features, cat_cardinalities, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * d_ffn_factor),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_num = x[:, : self.n_num]
        x_cat = x[:, self.n_num :].long()

        tokens = self.tokenizer(x_num, x_cat)            # (B, n_features, d_token)
        cls = self.cls_token.expand(x.size(0), -1, -1)   # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, 1+n_features, d_token)

        tokens = self.transformer(tokens)
        cls_out = self.norm(tokens[:, 0])                 # (B, d_token)
        return self.head(cls_out).squeeze(-1)              # (B,)

    def freeze_backbone(self) -> None:
        """Freeze every parameter except the regression head."""
        for name, p in self.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True
