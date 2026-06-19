# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import json
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from qwenvl.modeling_s1_omni import S1Omni
from qwenvl.modeling_s1_protein import S1Protein

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    if isinstance(trainer.model, (S1Omni, S1Protein)):
        if not trainer.args.should_save:
            return
        trainer.model.save_pretrained(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class S1OmniTrainer(Trainer):
    def create_optimizer(self):
        """Custom optimizer that splits ESM2 unfrozen params into a lower-LR group.

        HuggingFace ``Trainer.create_optimizer`` always groups trainable
        parameters with a single ``learning_rate`` (``self.args.learning_rate``)
        and a single ``weight_decay`` (``self.args.weight_decay``). For
        ``S1Protein`` with the ESM2 last-N-layers unfreezing strategy, we
        want the unfrozen ESM2 parameters to use a much smaller learning
        rate (``learning_rate * esm_lr_multiplier``) so that we don't damage
        the pretrained ESM2 representations.
        """
        model = self.model
        if not isinstance(model, S1Protein) or model.esm_model is None:
            return super().create_optimizer()

        esm_lr_multiplier = float(getattr(model, "esm_lr_multiplier", 0.0) or 0.0)
        n_esm_trainable = model.trainable_esm_param_count()
        if esm_lr_multiplier <= 0.0 or n_esm_trainable == 0:
            return super().create_optimizer()

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(model)
            esm_param_ids = {
                id(p) for p in model.esm_model.parameters() if p.requires_grad
            }
            base_lr = self.args.learning_rate
            esm_lr = base_lr * esm_lr_multiplier

            other_trainable = [
                (n, p)
                for n, p in model.named_parameters()
                if p.requires_grad and id(p) not in esm_param_ids
            ]
            esm_trainable = [
                (f"esm_model.{n}", p)
                for n, p in model.esm_model.named_parameters()
                if p.requires_grad
            ]

            def split_decay(items):
                decay, no_decay = [], []
                for n, p in items:
                    if n in decay_parameters:
                        decay.append(p)
                    else:
                        no_decay.append(p)
                return decay, no_decay

            other_decay, other_no_decay = split_decay(other_trainable)
            esm_decay, esm_no_decay = split_decay(esm_trainable)

            param_groups = []
            if other_decay:
                param_groups.append(
                    {"params": other_decay, "lr": base_lr, "weight_decay": self.args.weight_decay}
                )
            if other_no_decay:
                param_groups.append(
                    {"params": other_no_decay, "lr": base_lr, "weight_decay": 0.0}
                )
            if esm_decay:
                param_groups.append(
                    {"params": esm_decay, "lr": esm_lr, "weight_decay": self.args.weight_decay}
                )
            if esm_no_decay:
                param_groups.append(
                    {"params": esm_no_decay, "lr": esm_lr, "weight_decay": 0.0}
                )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, model
            )
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            print(
                f"S1OmniTrainer built custom optimizer: "
                f"{len(param_groups)} param groups; "
                f"ESM2 lr multiplier={esm_lr_multiplier}, "
                f"esm_lr={esm_lr:.3e}, base_lr={base_lr:.3e}"
            )
        return self.optimizer

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if not isinstance(model, (S1Omni, S1Protein)):
            return super()._save(output_dir=output_dir, state_dict=state_dict)

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir, state_dict=state_dict)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def save_regression_label_stats(train_dataset, output_dir: str):
    if not hasattr(train_dataset, "regression_label_mean"):
        return

    stats = {
        "mean": train_dataset.regression_label_mean,
        "std": train_dataset.regression_label_std,
        "count": train_dataset.regression_label_count,
        "route_counts": getattr(train_dataset, "route_counts", None),
        "class_counts": getattr(train_dataset, "class_counts", None),
    }
    stats_path = pathlib.Path(output_dir) / "s1_omni_regression_label_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def find_resumable_checkpoint(output_dir: str) -> str | None:
    """Return the latest checkpoint directory that contains trainer state.

    Interrupted saves can leave behind ``checkpoint-*`` directories with model
    files but without ``trainer_state.json``. Those are not resumable via
    ``Trainer.train(resume_from_checkpoint=...)`` and should be skipped.
    """
    checkpoint_dirs = sorted(
        pathlib.Path(output_dir).glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoint_dirs:
        return None

    incomplete = []
    for checkpoint_dir in reversed(checkpoint_dirs):
        trainer_state = checkpoint_dir / "trainer_state.json"
        if trainer_state.exists():
            if incomplete:
                logging.warning(
                    "ignoring incomplete checkpoints without trainer_state.json: %s",
                    ", ".join(path.name for path in reversed(incomplete)),
                )
            return str(checkpoint_dir)
        incomplete.append(checkpoint_dir)

    logging.warning(
        "found checkpoint directories in %s but none are resumable; missing trainer_state.json in: %s",
        output_dir,
        ", ".join(path.name for path in checkpoint_dirs),
    )
    return None


def set_model(model_args, model):
    if isinstance(model, S1Protein):
        model.unfreeze_llm_backbone()
        for p in model.protein_head.parameters():
            p.requires_grad = True
        # Re-apply the configured ESM2 partial-unfreeze policy. The call above
        # only re-enables Qwen language_model + lm_head and then re-runs the
        # unfreeze helper, which respects ``esm_unfreeze_last_n_layers`` etc.
        esm_unfreeze = getattr(model_args, "esm_unfreeze_last_n_layers", 0)
        if esm_unfreeze and esm_unfreeze > 0:
            n_trainable_esm = model.trainable_esm_param_count()
            print(
                f"S1Protein ESM2 unfreeze: last {esm_unfreeze} layers configured; "
                f"trainable ESM2 params={n_trainable_esm}; "
                f"esm_lr_multiplier={model.esm_lr_multiplier}"
            )
        return

    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        head_names = ("router", "classification_decoder", "regression_decoder", "protein_head")
        for head_name in head_names:
            head = getattr(model, head_name, None)
            if head is None:
                continue
            for p in head.parameters():
                p.requires_grad = True
        return

    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    data_args.model_architecture = model_args.model_architecture
    data_args.use_esm2 = model_args.use_esm2
    data_args.esm_model_name = model_args.esm_model_name

    if model_args.model_architecture.lower() == "s1-omni":
        model = S1Omni.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif model_args.model_architecture.lower() == "s1-protein":
        model = S1Protein.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
            use_esm2=model_args.use_esm2,
            esm_model_name=model_args.esm_model_name,
            esm_fusion_dim=model_args.esm_fusion_dim,
            esm_num_attention_heads=model_args.esm_num_attention_heads,
            esm_fusion_num_layers=model_args.esm_fusion_num_layers,
            esm_fusion_ffn_dim=model_args.esm_fusion_ffn_dim,
            esm_unfreeze_last_n_layers=model_args.esm_unfreeze_last_n_layers,
            esm_unfreeze_pooler=model_args.esm_unfreeze_pooler,
            esm_unfreeze_final_layer_norm=model_args.esm_unfreeze_final_layer_norm,
            esm_lr_multiplier=model_args.esm_lr_multiplier,
            positive_loss_weight=model_args.positive_loss_weight,
            protein_loss_type=model_args.protein_loss_type,
            asl_gamma_pos=model_args.asl_gamma_pos,
            asl_gamma_neg=model_args.asl_gamma_neg,
            asl_clip=model_args.asl_clip,
            asl_eps=model_args.asl_eps,
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if hasattr(model, "set_tokenizer"):
        model.set_tokenizer(tokenizer)

    if training_args.lora_enable and not hasattr(model, "freeze_backbone"):
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_rank0 = torch.distributed.get_rank() == 0
        else:
            is_rank0 = True
        if is_rank0 and hasattr(model, "backbone"):
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"{model.__class__.__name__} trainable params: {trainable}/{total}")
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    if training_args.should_save:
        save_regression_label_stats(data_module["train_dataset"], training_args.output_dir)
    train_dataset = data_module["train_dataset"]
    if hasattr(model, "initialize_heads_from_priors"):
        model.initialize_heads_from_priors(
            route_counts=getattr(train_dataset, "route_counts", None),
            class_counts=getattr(train_dataset, "class_counts", None),
        )
        if local_rank == 0:
            print(
                "S1-Omni heads initialized from priors:",
                {
                    "route_counts": getattr(train_dataset, "route_counts", None),
                    "class_counts": getattr(train_dataset, "class_counts", None),
                },
            )
    trainer = S1OmniTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    resume_checkpoint = find_resumable_checkpoint(training_args.output_dir)
    if resume_checkpoint is not None:
        logging.info("resuming training from checkpoint %s", resume_checkpoint)
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    trainer.save_state()

    if hasattr(model, "config"):
        model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
