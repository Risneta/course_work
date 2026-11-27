"""Transformer Encoder regressor (trích xuất được attention) và LSTM baseline."""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):  # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer; có thể lưu attention weights (B, heads, S, S)."""

    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(dim_ff, d_model))
        self.drop = nn.Dropout(dropout)
        self.store_attn = False
        self.last_attn = None

    def forward(self, x):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=self.store_attn, average_attn_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        if self.store_attn:
            self.last_attn = w.detach()
        return x


def _head(d, dropout):
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))


class TransformerRegressor(nn.Module):
    """Input (B, L, F) -> dự đoán (B,). pooling='cls' thêm 1 token CLS ở đầu chuỗi."""

    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1,
                 pooling="cls", max_len=512):
        super().__init__()
        assert pooling in ("cls", "mean", "last")
        self.pooling = pooling
        self.has_cls = pooling == "cls"
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        if self.has_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = _head(d_model, dropout)

    def forward(self, x):
        h = self.pos(self.input_proj(x))
        if self.has_cls:
            h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], dim=1)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if self.pooling == "cls":
            z = h[:, 0]
        elif self.pooling == "mean":
            z = h.mean(dim=1)
        else:
            z = h[:, -1]
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def forward_with_attention(self, x):
        """Trả về (pred, [attn_layer_0, ...]); mỗi attn có shape (B, heads, S, S), S = L (+1 nếu có CLS).
        Gọi sau model.eval() để không bị dropout làm nhiễu attention."""
        for layer in self.layers:
            layer.store_attn = True
        try:
            out = self.forward(x)
            attn = [layer.last_attn for layer in self.layers]
        finally:
            for layer in self.layers:
                layer.store_attn = False
                layer.last_attn = None
        return out, attn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = _head(hidden, dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def build_model(name, cfg, n_features):
    if name == "transformer":
        return TransformerRegressor(n_features, **cfg["transformer"])
    if name == "lstm":
        return LSTMRegressor(n_features, **cfg["lstm"])
    raise ValueError(f"Model không hợp lệ: {name}")


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = build_model(ck["model_name"], {ck["model_name"]: ck["model_cfg"]}, ck["n_features"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
