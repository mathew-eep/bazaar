from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x) * torch.sigmoid(self.gate(x))
        return self.dropout(out)


class GRN(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_out, dropout=dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Input shape: (B, T, F)
    Output shape: (B, T, D), weights: (B, T, F)
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.feature_grns = nn.ModuleList([GRN(1, d_model, dropout=dropout) for _ in range(n_features)])
        self.weight_grn = GRN(n_features, n_features, dropout=dropout)
        self.weight_proj = nn.Linear(n_features, n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        b, t, f = x.shape
        if f != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {f}")

        x = self.dropout(x)
        weights_logits = self.weight_proj(self.weight_grn(x))
        weights = torch.softmax(weights_logits, dim=-1)
        weights = self.dropout(weights)

        transformed = []
        for i in range(self.n_features):
            xi = x[..., i : i + 1]
            transformed.append(self.feature_grns[i](xi))

        stacked = torch.stack(transformed, dim=2)  # (B, T, F, D)
        selected = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)
        return selected, weights


class BazaarTFT(nn.Module):
    def __init__(
        self,
        n_past_features: int,
        n_future_features: int,
        n_static_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        lookback: int = 168,
        horizon: int = 24,
        n_quantiles: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.dropout_rate = dropout

        self.input_dropout = nn.Dropout(dropout)
        self.static_dropout = nn.Dropout(dropout)
        self.post_lstm_dropout = nn.Dropout(dropout)
        self.post_attn_dropout = nn.Dropout(dropout)
        self.head_dropout = nn.Dropout(dropout)

        self.past_vsn = VariableSelectionNetwork(n_past_features, d_model, dropout=dropout)
        self.future_vsn = VariableSelectionNetwork(n_future_features, d_model, dropout=dropout)

        self.static_grn = GRN(n_static_features, d_model, dropout=dropout)

        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_gate = GLU(d_model, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.quantile_head = nn.Linear(d_model, n_quantiles)

    def forward(
        self,
        past_obs: torch.Tensor,
        future_known: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        # past_obs: (B, lookback, n_past_features)
        # future_known: (B, horizon, n_future_features)
        # static: (B, n_static_features)
        past_obs = torch.nan_to_num(past_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        future_known = torch.nan_to_num(future_known, nan=0.0, posinf=1e6, neginf=-1e6)
        static = torch.nan_to_num(static, nan=0.0, posinf=1e6, neginf=-1e6)

        past_obs = self.input_dropout(past_obs)
        future_known = self.input_dropout(future_known)
        static = self.static_dropout(static)

        past_sel, _ = self.past_vsn(past_obs)
        future_sel, _ = self.future_vsn(future_known)

        static_ctx = self.static_grn(static).unsqueeze(1)  # (B,1,D)

        enc_out, (h, c) = self.encoder(past_sel)
        dec_out, _ = self.decoder(future_sel, (h, c))
        enc_out = self.post_lstm_dropout(enc_out)
        dec_out = self.post_lstm_dropout(dec_out)

        full_seq = torch.cat([enc_out, dec_out], dim=1)  # (B, lookback+horizon, D)
        full_ctx = full_seq + static_ctx

        attn_out, _ = self.attn(full_ctx, full_ctx, full_ctx, need_weights=False)
        full_seq = self.attn_norm(full_seq + self.post_attn_dropout(attn_out))

        future_slice = full_seq[:, -self.horizon :, :]
        ffn_out = self.ffn(future_slice)
        ffn_out = self.ffn_gate(ffn_out)
        future_slice = self.ffn_norm(future_slice + self.post_attn_dropout(ffn_out))

        future_slice = self.head_dropout(future_slice)

        out = self.quantile_head(future_slice)
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
