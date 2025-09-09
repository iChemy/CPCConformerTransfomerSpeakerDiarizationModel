from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import Swish
from .convolution import StreamingConvolutionModule


class StreamingConformerBlock(nn.Module):
    """アテンションのブロックレベル再帰を実装したConformerブロック"""

    PastState = Tuple[
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
    ]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        kernel_size: int,
        dropout: float = 0.1,
        attention_context_size: int = 256,
    ):
        super().__init__()
        self.attention_context_size = attention_context_size
        self.d_model = d_model
        self.nhead = nhead
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm_mhsa = nn.LayerNorm(d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout_mhsa = nn.Dropout(dropout)
        self.conv_module = StreamingConvolutionModule(d_model, kernel_size, dropout)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, past_state: Optional[PastState] = None
    ) -> Tuple[torch.Tensor, PastState]:
        if past_state is None:
            past_state = (None, None)
        x = x + 0.5 * self.ffn1(x)
        res_mhsa = x
        x = self.norm_mhsa(x)
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        past_kv = past_state[0]
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        if k.size(1) > self.attention_context_size:
            k = k[:, -self.attention_context_size :, :]
            v = v[:, -self.attention_context_size :, :]
        new_kv = (k.detach(), v.detach())
        B, T_q, D = q.shape
        T_k = k.size(1)
        q = q.view(B, T_q, self.nhead, D // self.nhead).transpose(1, 2)
        k = k.view(B, T_k, self.nhead, D // self.nhead).transpose(1, 2)
        v = v.view(B, T_k, self.nhead, D // self.nhead).transpose(1, 2)
        attn_mask = torch.triu(
            torch.ones(T_q, T_k, device=x.device, dtype=torch.bool),
            diagonal=1 - (T_k - T_q),
        )
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, D)
        x = self.linear_out(attn_output)
        x = res_mhsa + self.dropout_mhsa(x)
        res_conv = x
        past_conv_context = past_state[1]
        x, new_conv_context = self.conv_module(x, past_conv_context)
        x = res_conv + x
        x = x + 0.5 * self.ffn2(x)
        x = self.final_norm(x)
        new_state = (new_kv, new_conv_context)
        return x, new_state
