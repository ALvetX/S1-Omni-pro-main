import os
import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen3VLForConditionalGeneration,
)
from transformers.modeling_outputs import ModelOutput
try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    safe_load_file = None


LINEAR_CLA_TOKEN = "<linear_cla>"
LINEAR_PRE_TOKEN = "<linear_pre>"
ROUTE_CLS = 0
ROUTE_REG = 1


def _find_special_token_id(tokenizer, token: str) -> Optional[int]:
    if tokenizer is None:
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        return None
    return token_id

@dataclass
class S1OmniOutput(ModelOutput):
    loss: torch.Tensor | None = None
    route_logits: torch.Tensor | None = None
    task_logits: torch.Tensor | None = None
    regression_value: torch.Tensor | None = None
    pred_route: torch.Tensor | None = None
    backbone_hidden_states: tuple[torch.Tensor, ...] | None = None
    backbone_attentions: tuple[torch.Tensor, ...] | None = None


class S1OmniConfigMixin:
    route_loss_weight: float = 1.0
    task_loss_weight: float = 1.0
    hidden_dropout_prob: float = 0.1


class S1Omni(nn.Module):
    def __init__(self, backbone, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.tokenizer = None
        self.backbone_frozen = False

        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )
        self.classification_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )
        self.regression_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )
        self._init_head_module(self.router)
        self._init_head_module(self.classification_decoder)
        self._init_head_module(self.regression_decoder)

        for p in self.backbone.parameters():
            p.requires_grad = False

    @property
    def supports_gradient_checkpointing(self):
        if self.backbone_frozen:
            return False
        return hasattr(self.backbone, "supports_gradient_checkpointing") or hasattr(
            self.backbone, "gradient_checkpointing_enable"
        )

    @property
    def is_gradient_checkpointing(self):
        if hasattr(self.backbone, "is_gradient_checkpointing"):
            return self.backbone.is_gradient_checkpointing
        return False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if self.backbone_frozen:
            return None
        if gradient_checkpointing_kwargs is None:
            # PyTorch reentrant checkpoint 在一些 FlashAttention + DeepSpeed 场景下更容易触发
            # recompute metadata mismatch，这里默认切到非 reentrant 模式。
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            return self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        raise AttributeError(
            f"{self.__class__.__name__} does not support gradient checkpointing"
        )

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            return self.backbone.gradient_checkpointing_disable()
        raise AttributeError(
            f"{self.__class__.__name__} does not support gradient checkpointing"
        )

    def enable_input_require_grads(self):
        if hasattr(self.backbone, "enable_input_require_grads"):
            return self.backbone.enable_input_require_grads()
        return None

    def disable_input_require_grads(self):
        if hasattr(self.backbone, "disable_input_require_grads"):
            return self.backbone.disable_input_require_grads()
        return None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        dtype=None,
        **kwargs,
    ):
        backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=dtype,
            **kwargs,
        )
        hidden_size = backbone.config.text_config.hidden_size
        model = cls(backbone=backbone, hidden_size=hidden_size)
        target_dtype = next(backbone.parameters()).dtype
        model.to(dtype=target_dtype)
        model._load_heads_from_pretrained(model_name_or_path)
        model.config = backbone.config
        model.processor_name_or_path = model_name_or_path
        return model

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "disable_input_require_grads"):
            self.backbone.disable_input_require_grads()
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        self.backbone_frozen = True

    @staticmethod
    def _init_head_module(module: nn.Module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    @staticmethod
    def _prior_bias(counts: Optional[list[int]], size: int, device, dtype):
        if counts is None or len(counts) != size:
            counts = [1] * size
        counts_tensor = torch.tensor(counts, device=device, dtype=torch.float32)
        probs = (counts_tensor + 1.0) / (counts_tensor.sum() + float(size))
        return torch.log(probs).to(dtype=dtype)

    def initialize_heads_from_priors(
        self,
        route_counts: Optional[list[int]] = None,
        class_counts: Optional[list[int]] = None,
        init_std: float = 0.02,
        output_std: float = 1e-3,
    ):
        del init_std, output_std
        self._init_head_module(self.router)
        self._init_head_module(self.classification_decoder)
        self._init_head_module(self.regression_decoder)

        with torch.no_grad():
            router_out = self.router[-1]
            cls_out = self.classification_decoder[-1]
            reg_out = self.regression_decoder[-1]

            router_out.bias.copy_(
                self._prior_bias(route_counts, 2, router_out.bias.device, router_out.bias.dtype)
            )
            cls_out.bias.copy_(
                self._prior_bias(class_counts, self.num_labels, cls_out.bias.device, cls_out.bias.dtype)
            )
            nn.init.zeros_(reg_out.bias)

    @staticmethod
    def _load_checkpoint_file(path: str):
        if path.endswith(".safetensors"):
            if safe_load_file is None:
                raise ImportError("safetensors is required to load safetensors checkpoints")
            return safe_load_file(path, device="cpu")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @classmethod
    def _load_state_dict_files(cls, model_name_or_path: str):
        if not os.path.isdir(model_name_or_path):
            return []

        files = []
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = os.path.join(model_name_or_path, index_name)
            if not os.path.exists(index_path):
                continue
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            head_files = {
                filename
                for key, filename in index.get("weight_map", {}).items()
                if key.startswith(("router.", "classification_decoder.", "regression_decoder."))
            }
            files.extend(os.path.join(model_name_or_path, filename) for filename in sorted(head_files))
            return files

        for filename in ("model.safetensors", "pytorch_model.bin"):
            path = os.path.join(model_name_or_path, filename)
            if os.path.exists(path):
                files.append(path)
        return files

    def _load_heads_from_pretrained(self, model_name_or_path: str):
        head_state_dict = {
            "router": {},
            "classification_decoder": {},
            "regression_decoder": {},
        }
        for path in self._load_state_dict_files(model_name_or_path):
            state_dict = self._load_checkpoint_file(path)
            for module_name in head_state_dict:
                prefix = f"{module_name}."
                for key, value in state_dict.items():
                    if key.startswith(prefix):
                        head_state_dict[module_name][key.removeprefix(prefix)] = value

        found_modules = [module_name for module_name, module_state_dict in head_state_dict.items() if module_state_dict]
        if found_modules and len(found_modules) != len(head_state_dict):
            missing_modules = sorted(set(head_state_dict) - set(found_modules))
            raise RuntimeError(f"incomplete S1-Omni head checkpoint, missing={missing_modules}")
        for module_name, module_state_dict in head_state_dict.items():
            if module_state_dict:
                getattr(self, module_name).load_state_dict(module_state_dict, strict=True)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_tokenizer(self, tokenizer):
        # 仅用于调试：将 tokenizer 挂到模型上，便于在 forward 中把最后一个有效 token 解码出来。
        self.tokenizer = tokenizer

    def _is_rank0(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _pool_hidden(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        # 将序列级的 hidden_states 池化成一个向量表示。
        if attention_mask is None:
            return hidden_states[:, -1]
        lengths = attention_mask.long().sum(dim=-1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def _token_ids_for_text(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _find_subsequence(row: list[int], pattern: list[int], start: int = 0) -> Optional[int]:
        if not pattern:
            return None
        max_start = len(row) - len(pattern)
        for idx in range(start, max_start + 1):
            if row[idx : idx + len(pattern)] == pattern:
                return idx
        return None

    def _find_chat_role_token_spans(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        role: str,
        allow_open_ended: bool = False,
    ):
        if self.tokenizer is None or input_ids is None:
            return None

        im_start_id = _find_special_token_id(self.tokenizer, "<|im_start|>")
        im_end_id = _find_special_token_id(self.tokenizer, "<|im_end|>")
        role_ids = self._token_ids_for_text(role)
        newline_ids = self._token_ids_for_text("\n")
        if im_start_id is None or im_end_id is None or not role_ids:
            return None

        spans = []
        lengths = (
            attention_mask.long().sum(dim=-1).clamp(min=1).tolist()
            if attention_mask is not None
            else [input_ids.size(1)] * input_ids.size(0)
        )
        header_pattern = [im_start_id] + role_ids
        for row, length in zip(input_ids, lengths):
            row_list = row[: int(length)].tolist()
            search_from = 0
            selected_span = None
            while True:
                header_pos = self._find_subsequence(row_list, header_pattern, search_from)
                if header_pos is None:
                    break
                content_start = header_pos + len(header_pattern)
                if newline_ids and row_list[content_start : content_start + len(newline_ids)] == newline_ids:
                    content_start += len(newline_ids)
                content_end = self._find_subsequence(row_list, [im_end_id], content_start)
                if content_end is None:
                    if allow_open_ended:
                        content_end = len(row_list)
                    else:
                        break
                if content_end > content_start:
                    selected_span = (content_start, content_end)
                if content_end >= len(row_list):
                    break
                search_from = content_end + 1
            spans.append(selected_span)
        return spans

    def _find_question_token_spans(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ):
        return self._find_chat_role_token_spans(input_ids, attention_mask, role="user")

    def _decode_spans(
        self,
        input_ids: Optional[torch.Tensor],
        spans,
    ):
        if self.tokenizer is None or input_ids is None or spans is None:
            return None

        decoded = []
        for row, span in zip(input_ids, spans):
            if span is None:
                decoded.append(None)
                continue
            start, end = span
            token_ids = row[int(start) : int(end)].tolist()
            try:
                text = self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            decoded.append(text)
        return decoded

    def _decode_question_tokens(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ):
        return self._decode_spans(
            input_ids,
            self._find_question_token_spans(input_ids, attention_mask),
        )

    def _pool_hidden_from_marker(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ):
        question_spans = self._find_question_token_spans(input_ids, attention_mask)
        if question_spans is None:
            return self._pool_hidden(hidden_states, attention_mask)

        pooled_states = []
        fallback_lengths = attention_mask.long().sum(dim=-1).clamp(min=1) - 1 if attention_mask is not None else None
        for i, span in enumerate(question_spans):
            if span is None:
                if fallback_lengths is None:
                    pooled_states.append(hidden_states[i, -1])
                else:
                    pooled_states.append(hidden_states[i, int(fallback_lengths[i].item())])
            else:
                start, end = span
                pooled_states.append(hidden_states[i, int(start) : int(end)].mean(dim=0))
        return torch.stack(pooled_states, dim=0)

    def compute_pooled_hidden(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        outputs = self.backbone.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        hidden_states = outputs[0]
        if self.backbone_frozen:
            hidden_states = hidden_states.detach()
        pooled = self._pool_hidden_from_marker(hidden_states, input_ids, attention_mask)
        return pooled, outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        logits_to_keep=0,
        route_id=None,
        task_type=None,
        label=None,
        pooled_hidden=None,
        cache_index=None,
        **kwargs,
    ):
        del cache_index
        kwargs.pop("output_assistant_text", None)
        if pooled_hidden is None:
            pooled, outputs = self.compute_pooled_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
            backbone_hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
            backbone_attentions = outputs.attentions if hasattr(outputs, "attentions") else None
        else:
            outputs = None
            backbone_hidden_states = None
            backbone_attentions = None
            head_param = next(self.router.parameters())
            pooled = pooled_hidden.to(device=head_param.device, dtype=head_param.dtype)

        if pooled_hidden is None and os.environ.get("S1OMNI_DEBUG_LAST_TOKEN") == "1" and self._is_rank0():
            decoded_questions = self._decode_question_tokens(input_ids, attention_mask)
            if decoded_questions is not None:
                print(
                    "S1Omni question decode:",
                    decoded_questions,
                    flush=True,
                )

        route_logits = self.router(pooled)
        pred_route = route_logits.argmax(dim=-1)

        task_logits = self.classification_decoder(pooled)
        regression_value = self.regression_decoder(pooled).squeeze(-1)

        loss = None
        if route_id is not None and label is not None:
            route_target = route_id.to(route_logits.device).long()
            route_loss = F.cross_entropy(route_logits, route_target)

            task_mask = route_target == ROUTE_CLS
            reg_mask = route_target == ROUTE_REG
            task_loss = route_logits.new_zeros(())
            zero = route_logits.new_zeros(())

            if task_mask.any():
                cls_target = label[task_mask].to(route_logits.device).long()
                task_loss = task_loss + F.cross_entropy(task_logits[task_mask], cls_target)
            else:
                task_loss = task_loss + task_logits.sum() * zero
            if reg_mask.any():
                reg_target = label[reg_mask].to(
                    device=regression_value.device,
                    dtype=regression_value.dtype,
                )
                task_loss = task_loss + F.smooth_l1_loss(regression_value[reg_mask], reg_target)
            else:
                task_loss = task_loss + regression_value.sum() * zero

            loss = route_loss + task_loss
            if os.environ.get("S1OMNI_DEBUG_DTYPE") == "1":
                print(
                    "S1Omni dtype debug:",
                    {
                        "pooled": pooled.dtype,
                        "route_logits": route_logits.dtype,
                        "task_logits": task_logits.dtype,
                        "regression_value": regression_value.dtype,
                        "label": label.dtype,
                        "loss": loss.dtype,
                    },
                    flush=True,
                )

        return S1OmniOutput(
            loss=loss,
            route_logits=route_logits,
            task_logits=task_logits,
            regression_value=regression_value,
            pred_route=pred_route,
            backbone_hidden_states=backbone_hidden_states,
            backbone_attentions=backbone_attentions,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        state_dict = kwargs.pop("state_dict", None)

        combined_state_dict = {}
        if state_dict:
            normalized_state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }
            combined_state_dict.update(
                {
                    key.removeprefix("backbone."): value
                    for key, value in normalized_state_dict.items()
                    if key.startswith("backbone.")
                }
            )
            for module_name in ("router", "classification_decoder", "regression_decoder"):
                prefix = f"{module_name}."
                combined_state_dict.update(
                    {
                        key: value
                        for key, value in normalized_state_dict.items()
                        if key.startswith(prefix)
                    }
                )
        else:
            combined_state_dict.update(self.backbone.state_dict())

        for module_name in ("router", "classification_decoder", "regression_decoder"):
            module = getattr(self, module_name)
            prefix = f"{module_name}."
            for key, value in module.state_dict().items():
                combined_state_dict.setdefault(f"{prefix}{key}", value)

        self.backbone.save_pretrained(save_directory, state_dict=combined_state_dict, **kwargs)

    @classmethod
    def from_checkpoint(cls, model_name_or_path: str, **kwargs):
        return cls.from_pretrained(model_name_or_path, **kwargs)
