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
