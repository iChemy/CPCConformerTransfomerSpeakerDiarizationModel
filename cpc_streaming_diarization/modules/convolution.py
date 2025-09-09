from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import Swish


class StreamingConvolutionModule(nn.Module):
    """ストリーミング対応の畳み込みモジュール"""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.layer_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, padding=0, groups=channels
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, past_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.layer_norm(x)
        if past_context is None:
            past_context = torch.zeros(
                x.shape[0], self.padding, x.shape[2], device=x.device
            )
        conv_input = torch.cat([past_context, x_norm], dim=1)
        new_context = conv_input[:, -self.padding :, :]
        x_conv = conv_input.transpose(1, 2)
        x_conv = self.pointwise_conv1(x_conv)
        x_conv = self.glu(x_conv)
        x_conv = F.pad(x_conv, (self.padding, 0))
        x_conv = self.depthwise_conv(x_conv)
        x_conv = self.batch_norm(x_conv)
        x_conv = self.swish(x_conv)
        x_conv = self.pointwise_conv2(x_conv)
        output = self.dropout(x_conv.transpose(1, 2))
        return output, new_context
