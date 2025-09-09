import argparse
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from huggingface_hub import PyTorchModelHubMixin

from cpc.feature_loader import get_default_cpc_config, getAR, getEncoder, loadArgs
from cpc.model import CPCModel

from .config import CPCStreamingDiarizationModelConfig
from .modules.conformer import StreamingConformerBlock
from .modules.decoder import TransformerDecoder, TransformerDecoderBlock
from .utils import get_default_device, apply_min_duration


class CPCStreamingDiarizationModel(nn.Module, PyTorchModelHubMixin):
    """
    BW-EDA-EENDの堅牢なストリーミング処理と、CPC/Conformer Encoder/Transformer Decoderを統合したモデル。
    """

    # 5 x 4 x 2 x 2 x 2 = 1600 のストライド
    # 利用する CPC モデルのサブサンプリング率に応じて変更する必要がある
    feature_extractor_subsampling: int = 160

    def __init__(self, config: CPCStreamingDiarizationModelConfig):
        super().__init__()
        self.config = config
        self.device = get_default_device()
        self.hidden_dim = config.hidden_dim
        self.max_speakers = config.max_speakers
        self.num_conformer_layers = config.num_conformer_layers
        self.threshold = config.threshold
        self.use_reordering = config.use_reordering
        self.use_averaging = config.use_averaging
        self.attractor_averaging_weight = config.attractor_averaging_weight
        self.downsampling_factor = config.downsampling_factor

        # 特徴量抽出器 (CPC)
        self.waveform_encoder = self.load_feature_extractor()
        for param in self.waveform_encoder.parameters():
            param.requires_grad = False
        self.waveform_encoder.eval()

        # ★★★ 時間軸ダウンサンプリング層を追加 ★★★
        if self.downsampling_factor > 1:
            self.downsampler = nn.Sequential(
                nn.Conv1d(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    kernel_size=config.downsampling_factor,
                    stride=config.downsampling_factor,
                ),
                nn.GELU(),  # 活性化関数を追加
            )

        # Conformer Encoder
        self.conformer_encoder = nn.ModuleList(
            [
                StreamingConformerBlock(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.conformer_ffn_dim,
                    kernel_size=config.conformer_kernel_size,
                )
                for _ in range(config.num_conformer_layers)
            ]
        )

        # --- ★★★ ここからが修正箇所 (LSTMをTransformerデコーダに置換) ★★★ ---
        # 学習可能な話者クエリ (Transformerデコーダへの初期入力)
        self.speaker_queries = nn.Parameter(
            torch.zeros(1, self.max_speakers, config.hidden_dim)
        )

        # Transformer Decoder
        decoder_layer = TransformerDecoderBlock(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.conformer_ffn_dim,
        )
        self.conformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )

        self.linear_attractor = nn.Linear(config.hidden_dim, 1)

        self.to(self.device)

    def get_frame_hop_sec(self, sample_rate: int = 16000) -> float:
        base = self.feature_extractor_subsampling
        return (base * self.config.downsampling_factor) / sample_rate

    def load_feature_extractor(self):
        url = self.config.cpc_checkpoint_url
        try:
            locArgs = get_default_cpc_config()
            checkpoint = load_state_dict_from_url(url, map_location=self.device)
            loadArgs(locArgs, argparse.Namespace(**checkpoint["config"]))
            locArgs.hiddenEncoder = self.hidden_dim
            model = CPCModel(getEncoder(locArgs), getAR(locArgs))
            model.load_state_dict(checkpoint["weights"], strict=False)
            print("CPC model loaded and frozen.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load CPC model from URL '{url}'. Error: {e}")

    def extract_feature(self, wavform: torch.Tensor) -> torch.Tensor:
        """
        `(B, T, D)` を返す．
        `T` は `downsampling_factor` の値によって決まる．
        """
        with torch.no_grad():
            cpc_features = self.waveform_encoder.gEncoder(
                wavform.unsqueeze(1)
            ).transpose(1, 2)

        if self.downsampling_factor > 1:
            cpc_features_transposed = cpc_features.transpose(1, 2)
            downsampled_transposed = self.downsampler(cpc_features_transposed)
            return downsampled_transposed.transpose(1, 2)
        else:
            return cpc_features

    def _reorder_attractors(self, current_attractors, prev_attractors):
        S_prev, _ = prev_attractors.shape
        S_curr = current_attractors.shape[0]
        sim = F.cosine_similarity(
            current_attractors.unsqueeze(1),
            prev_attractors.unsqueeze(0),
            dim=-1,
            eps=1e-8,
        )
        cost_matrix = 1 - sim
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        if not np.all(np.isfinite(cost_matrix_np)):
            return current_attractors
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
        reordered_attractors = torch.zeros_like(current_attractors)
        reordered_attractors[col_ind] = current_attractors[row_ind]
        unassigned_mask = torch.ones(S_curr, dtype=torch.bool, device=self.device)
        unassigned_mask[row_ind] = False
        if torch.any(unassigned_mask):
            reordered_attractors[len(col_ind) :] = current_attractors[unassigned_mask]
        return reordered_attractors

    def postprocess(
        self,
        predictions: torch.Tensor,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        frame_hop_sec = self.get_frame_hop_sec(sample_rate)
        return apply_min_duration(
            predictions, frame_hop_sec, min_duration_on, min_duration_off
        )

    def forward_block(
        self,
        x_block_wav: torch.Tensor,
        prev_conformer_states: Optional[List],
        prev_attractors: torch.Tensor,
        prev_num_speakers: torch.Tensor,
    ) -> Tuple:
        B, _ = x_block_wav.shape

        x_block_wav = x_block_wav / (
            torch.max(torch.abs(x_block_wav), dim=-1, keepdim=True)[0] + 1e-8
        )

        conformer_input = self.extract_feature(x_block_wav)

        next_conformer_states = []
        if prev_conformer_states is None:
            prev_conformer_states = [None] * self.num_conformer_layers

        for i, layer in enumerate(self.conformer_encoder):
            conformer_output, new_state = layer(
                conformer_input, prev_conformer_states[i]
            )
            next_conformer_states.append(new_state)
            conformer_input = conformer_output
        current_embeddings = conformer_output

        queries = self.speaker_queries.repeat(B, 1, 1)

        current_attractors = self.conformer_decoder(queries, current_embeddings)

        final_attractors_list = []
        next_num_speakers_list = []
        for i in range(B):
            num_s_prev = int(prev_num_speakers[i].item())

            probs_i = torch.sigmoid(
                self.linear_attractor(current_attractors[i])
            ).squeeze(-1)
            num_s_curr = torch.sum(probs_i >= self.threshold).item()
            num_attractors_i = max(num_s_prev, num_s_curr)
            if num_attractors_i == 0 and len(probs_i) > 0:
                num_attractors_i = 1

            attractors_i = current_attractors[i, :num_attractors_i]
            if num_s_prev > 0 and attractors_i.numel() > 0:
                if self.use_reordering:
                    attractors_i = self._reorder_attractors(
                        attractors_i, prev_attractors[i, :num_s_prev]
                    )
                if self.use_averaging:
                    w = self.attractor_averaging_weight
                    len_to_avg = min(num_s_prev, attractors_i.shape[0])
                    attractors_i[:len_to_avg] = (1 - w) * prev_attractors[
                        i, :len_to_avg
                    ] + w * attractors_i[:len_to_avg]

            final_attractors_list.append(attractors_i)
            next_num_speakers_list.append(attractors_i.shape[0])

        final_attractors_padded = nn.utils.rnn.pad_sequence(
            final_attractors_list, batch_first=True
        )
        next_num_speakers = torch.tensor(next_num_speakers_list, device=self.device)

        logits = torch.bmm(current_embeddings, final_attractors_padded.transpose(1, 2))
        p_blk_downsampled = torch.sigmoid(logits)
        p_blk = p_blk_downsampled

        if p_blk.shape[2] < self.max_speakers:
            padding = torch.zeros(
                B,
                p_blk.shape[1],
                self.max_speakers - p_blk.shape[2],
                device=self.device,
            )
            p_blk = torch.cat([p_blk, padding], dim=2)

        return (
            p_blk,
            next_conformer_states,
            final_attractors_padded,
            next_num_speakers,
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        block_size_sec: float,
        hop_size_sec: float,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        B, T_total_wav = waveforms.shape
        device = self.device
        block_size = int(block_size_sec * sample_rate)
        hop_size = int(hop_size_sec * sample_rate)

        prev_conformer_states = None
        # --- LSTM state is no longer needed ---
        prev_attractors = torch.zeros(
            B, self.max_speakers, self.hidden_dim, device=device
        )
        prev_num_speakers = torch.zeros(B, dtype=torch.long, device=device)

        results = []
        for start in range(0, T_total_wav, hop_size):
            end = min(start + block_size, T_total_wav)
            if end <= start:
                continue

            x_blk_wav = waveforms[:, start:end]
            if x_blk_wav.shape[1] < self.feature_extractor_subsampling:
                continue

            (p_blk, conf_states, attractors, num_speakers) = self.forward_block(
                x_block_wav=x_blk_wav,
                prev_conformer_states=prev_conformer_states,
                prev_attractors=prev_attractors,
                prev_num_speakers=prev_num_speakers,
            )

            num_cpc_frames_in_hop = hop_size // self.feature_extractor_subsampling
            num_frames_to_add = min(p_blk.shape[1], num_cpc_frames_in_hop)
            if num_frames_to_add > 0:
                results.append(p_blk[:, :num_frames_to_add, :])

            prev_conformer_states = conf_states

            padded_attractors = torch.zeros(
                B, self.max_speakers, self.hidden_dim, device=device
            )
            for i in range(B):
                num_s = attractors[i].shape[0]
                if num_s > 0:
                    padded_attractors[i, :num_s] = attractors[i]
            prev_attractors = padded_attractors
            prev_num_speakers = num_speakers

        if not results:
            return torch.zeros(B, 0, self.max_speakers, device=device)

        all_prob = torch.cat(results, dim=1)
        return all_prob
