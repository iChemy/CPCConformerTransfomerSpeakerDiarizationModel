from dataclasses import dataclass


@dataclass
class CPCStreamingDiarizationModelConfig:
    hidden_dim: int = 256
    num_heads: int = 4
    num_conformer_layers: int = 4
    num_decoder_layers: int = 2
    conformer_ffn_dim: int = 1024
    conformer_kernel_size: int = 31
    threshold: float = 0.5
    max_speakers: int = 4
    downsampling_factor: int = 5
    use_reordering: bool = True
    use_averaging: bool = True
    attractor_averaging_weight: float = 0.5
    cpc_checkpoint_url: str = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"
