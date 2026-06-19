import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_architecture: str = field(default="qwen3vl")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    use_esm2: bool = field(default=True)
    esm_model_name: str = field(default="facebook/esm2_t33_650M_UR50D")
    esm_fusion_dim: int = field(default=512)
    esm_num_attention_heads: int = field(default=8)
    esm_fusion_num_layers: int = field(default=16)
    esm_fusion_ffn_dim: Optional[int] = field(default=None)
    esm_unfreeze_last_n_layers: int = field(default=8)
    esm_unfreeze_pooler: bool = field(default=False)
    esm_unfreeze_final_layer_norm: bool = field(default=True)
    esm_lr_multiplier: float = field(default=0.1)
    positive_loss_weight: float = field(default=50.0)
    protein_loss_type: str = field(default="bce")
    asl_gamma_pos: float = field(default=0.0)
    asl_gamma_neg: float = field(default=4.0)
    asl_clip: float = field(default=0.05)
    asl_eps: float = field(default=1e-8)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    annotation_path: Optional[str] = field(default=None)
    pooled_hidden_cache_path: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default=None)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
