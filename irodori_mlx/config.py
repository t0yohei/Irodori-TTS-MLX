from __future__ import annotations

from dataclasses import dataclass

CHECKPOINT_FAMILY_BASE_V2 = "base_v2"
CHECKPOINT_FAMILY_VOICEDESIGN_V2 = "voicedesign"
CHECKPOINT_FAMILY_V3 = "v3"


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
    use_duration_predictor: bool = False
    duration_aux_dim: int = 14
    duration_hidden_dim: int = 1024
    duration_layers: int = 3
    duration_dropout: float = 0.1
    duration_attention_heads: int = 8
    duration_architecture: str = "token_sum_adarn_zero_no_aux"
    duration_token_init_frames: float = 9.0
    duration_speaker_fusion: str = "adarn_zero"

    def __post_init__(self) -> None:
        architecture = str(self.duration_architecture).strip().lower()
        speaker_fusion = str(self.duration_speaker_fusion).strip().lower()
        object.__setattr__(self, "duration_architecture", architecture)
        object.__setattr__(self, "duration_speaker_fusion", speaker_fusion)

        if not self.use_duration_predictor:
            return
        if self.duration_aux_dim <= 0:
            raise ValueError(f"duration_aux_dim must be > 0, got {self.duration_aux_dim}")
        if self.duration_hidden_dim <= 0:
            raise ValueError(f"duration_hidden_dim must be > 0, got {self.duration_hidden_dim}")
        if self.duration_layers <= 0:
            raise ValueError(f"duration_layers must be > 0, got {self.duration_layers}")
        if not (0.0 <= self.duration_dropout <= 1.0):
            raise ValueError(f"duration_dropout must be in [0, 1], got {self.duration_dropout}")
        if self.duration_attention_heads <= 0:
            raise ValueError(
                f"duration_attention_heads must be > 0, got {self.duration_attention_heads}"
            )
        if self.text_dim % self.duration_attention_heads != 0:
            raise ValueError(
                "text_dim must be divisible by duration_attention_heads: "
                f"text_dim={self.text_dim} duration_attention_heads={self.duration_attention_heads}"
            )
        if self.duration_token_init_frames <= 0:
            raise ValueError(
                "duration_token_init_frames must be > 0, "
                f"got {self.duration_token_init_frames}"
            )
        if architecture != "token_sum_adarn_zero_no_aux":
            raise ValueError(
                "Unsupported duration_architecture for MLX: "
                f"{self.duration_architecture!r}"
            )
        if speaker_fusion != "adarn_zero":
            raise ValueError(
                "Unsupported duration_speaker_fusion for MLX: "
                f"{self.duration_speaker_fusion!r}"
            )
        if not self.use_speaker_condition:
            raise ValueError(
                "MLX duration predictor currently requires speaker-conditioned configs "
                "(use_caption_condition must be false)."
            )

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
    def checkpoint_family(self) -> str:
        if self.use_caption_condition:
            return CHECKPOINT_FAMILY_VOICEDESIGN_V2
        if self.use_duration_predictor:
            return CHECKPOINT_FAMILY_V3
        return CHECKPOINT_FAMILY_BASE_V2

    @property
    def checkpoint_family_label(self) -> str:
        labels = {
            CHECKPOINT_FAMILY_BASE_V2: "base v2 speaker/reference",
            CHECKPOINT_FAMILY_VOICEDESIGN_V2: "VoiceDesign v2 caption",
            CHECKPOINT_FAMILY_V3: "v3 speaker/reference duration-predictor",
        }
        return labels[self.checkpoint_family]

    @property
    def checkpoint_capabilities(self) -> tuple[str, ...]:
        capabilities = ["text"]
        if self.use_caption_condition:
            capabilities.append("caption")
            capabilities.append("no-reference")
        else:
            capabilities.append("speaker-reference")
            capabilities.append("no-reference")
        if self.use_duration_predictor:
            capabilities.append("predicted-duration")
        else:
            capabilities.append("manual-or-fallback-duration")
        return tuple(capabilities)

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
