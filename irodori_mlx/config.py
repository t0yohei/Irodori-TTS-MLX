from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Minimal Irodori-TTS model config needed by the MLX encoders."""

    latent_dim: int = 32
    latent_patch_size: int = 1
    model_dim: int = 1280
    num_layers: int = 12
    num_heads: int = 20
    mlp_ratio: float = 2.875
    text_mlp_ratio: float | None = 2.6
    speaker_mlp_ratio: float | None = 2.6
    dropout: float = 0.0
    text_vocab_size: int = 99574
    text_tokenizer_repo: str = "sbintuitions/sarashina2.2-0.5b"
    text_add_bos: bool = True
    text_dim: int = 512
    text_layers: int = 10
    text_heads: int = 8
    use_caption_condition: bool = False
    caption_vocab_size: int | None = None
    caption_tokenizer_repo: str | None = None
    caption_add_bos: bool | None = None
    caption_dim: int | None = None
    caption_layers: int | None = None
    caption_heads: int | None = None
    caption_mlp_ratio: float | None = None
    speaker_dim: int = 768
    speaker_layers: int = 8
    speaker_heads: int = 12
    speaker_patch_size: int = 1
    timestep_embed_dim: int = 512
    adaln_rank: int = 192
    norm_eps: float = 1e-5

    @property
    def patched_latent_dim(self) -> int:
        return self.latent_dim * self.latent_patch_size

    @property
    def speaker_patched_latent_dim(self) -> int:
        return self.patched_latent_dim * self.speaker_patch_size

    @property
    def use_speaker_condition(self) -> bool:
        return not bool(self.use_caption_condition)

    @property
    def text_mlp_ratio_resolved(self) -> float:
        return self.mlp_ratio if self.text_mlp_ratio is None else float(self.text_mlp_ratio)

    @property
    def speaker_mlp_ratio_resolved(self) -> float:
        return self.mlp_ratio if self.speaker_mlp_ratio is None else float(self.speaker_mlp_ratio)

    @property
    def caption_vocab_size_resolved(self) -> int:
        return self.text_vocab_size if self.caption_vocab_size is None else int(self.caption_vocab_size)

    @property
    def caption_tokenizer_repo_resolved(self) -> str:
        return self.text_tokenizer_repo if self.caption_tokenizer_repo is None else str(self.caption_tokenizer_repo)

    @property
    def caption_add_bos_resolved(self) -> bool:
        return self.text_add_bos if self.caption_add_bos is None else bool(self.caption_add_bos)

    @property
    def caption_dim_resolved(self) -> int:
        return self.text_dim if self.caption_dim is None else int(self.caption_dim)

    @property
    def caption_layers_resolved(self) -> int:
        return self.text_layers if self.caption_layers is None else int(self.caption_layers)

    @property
    def caption_heads_resolved(self) -> int:
        return self.text_heads if self.caption_heads is None else int(self.caption_heads)

    @property
    def caption_mlp_ratio_resolved(self) -> float:
        return self.text_mlp_ratio_resolved if self.caption_mlp_ratio is None else float(self.caption_mlp_ratio)
