import torch
import torch.nn as nn

from .activations import Swish


class TransformerDecoderBlock(nn.Module):
    """標準的なTransformerデコーダブロック"""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = Swish()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        tgt_norm = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt_norm, tgt_norm, tgt_norm)
        tgt = tgt + self.dropout1(tgt2)

        # Cross-Attention (Encoder-Decoder Attention)
        tgt_norm = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(query=tgt_norm, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)

        # Feed Forward
        tgt_norm = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class TransformerDecoder(nn.Module):
    """Transformerデコーダ層のスタック"""

    def __init__(self, decoder_layer: TransformerDecoderBlock, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.self_attn.embed_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(output, memory)
        return self.norm(output)
