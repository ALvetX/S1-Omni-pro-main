#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from qwenvl.modeling_s1_omni import ROUTE_CLS, ROUTE_REG, S1Omni


DEFAULT_CHECKPOINT = "/data/home/zdhs0092/Code/S1-Omni-pro/output/s1_omni_ep150/checkpoint-10000"
DEFAULT_STATS = "/data/home/zdhs0092/Code/S1-Omni-pro/output/s1_omni_ep150/s1_omni_regression_label_stats.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stats_path", default=DEFAULT_STATS)
    parser.add_argument(
        "--question",
        default=None,
        help="Run one question and exit. If omitted, enter interactive mode.",
    )
    parser.add_argument("--question_file", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--generate_text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def read_question(args) -> str:
    if args.question_file:
        return Path(args.question_file).read_text(encoding="utf-8").strip()
    if args.question is None:
        raise ValueError("missing question")
    return args.question


def build_messages(question: str):
    return [{"role": "user", "content": [{"type": "text", "text": question}]}]


def prepare_inputs(tokenizer, question: str, device: str, add_generation_prompt: bool = False):
    inputs = tokenizer.apply_chat_template(
        build_messages(question),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def prepare_batch_inputs(tokenizer, questions: list[str], device: str, add_generation_prompt: bool = False):
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(
            build_messages(question),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        for question in questions
    ]
    inputs = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def load_regression_stats(path: str | Path) -> dict[str, float]:
    stats_path = Path(path)
    if not stats_path.exists():
        return {"mean": 0.0, "std": 1.0}
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return {"mean": float(stats.get("mean", 0.0)), "std": float(stats.get("std", 1.0))}


def denormalize_regression(value: float, stats: dict[str, float]) -> float:
    return value * stats["std"] + stats["mean"]


def tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().float().cpu().tolist()]


def load_s1_omni(checkpoint_dir: str | Path, args):
    checkpoint_dir = Path(checkpoint_dir)

    dtype = resolve_dtype(args.dtype)
    model = S1Omni.from_pretrained(
        str(checkpoint_dir),
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir),
        padding_side="right",
        use_fast=False,
    )
    model.set_tokenizer(tokenizer)
    model.to(args.device)
    model.eval()
    return model, tokenizer


def infer_once(model: S1Omni, tokenizer, question: str, stats: dict[str, float], args):
    return infer_batch(model, tokenizer, [question], stats, args)[0]


def build_result(
    question: str,
    outputs,
    row_index: int,
    llm_output: str,
    stats: dict[str, float],
):
    route_prob = outputs.route_logits.softmax(dim=-1)[row_index]
    cls_prob = outputs.task_logits.softmax(dim=-1)[row_index]
    pred_route = int(outputs.pred_route[row_index].detach().cpu().item())
    pred_class = int(cls_prob.argmax().detach().cpu().item())
    regression_normalized = float(outputs.regression_value[row_index].detach().float().cpu().item())
    regression_value = denormalize_regression(regression_normalized, stats)
    final_prediction: int | float = pred_class if pred_route == ROUTE_CLS else regression_value

    return {
        "question": question,
        "route": {
            "pred_id": pred_route,
            "pred_name": "classification" if pred_route == ROUTE_CLS else "regression",
            "prob": tensor_to_float_list(route_prob),
        },
        "classification": {
            "pred_class": pred_class,
            "prob": tensor_to_float_list(cls_prob),
        },
        "regression": {
            "value": regression_value,
            "normalized_value": regression_normalized,
            "label_mean": stats["mean"],
            "label_std": stats["std"],
        },
        "final_prediction": final_prediction,
        "llm_output": llm_output,
    }


def infer_batch(model: S1Omni, tokenizer, questions: list[str], stats: dict[str, float], args):
    if not questions:
        return []

    # Training pools the hidden states from the user/question span. For inference we feed
    # the generated sequence back into forward; pooling still uses the user/question span.
    if len(questions) == 1:
        inputs = prepare_inputs(tokenizer, questions[0], args.device, add_generation_prompt=True)
    else:
        inputs = prepare_batch_inputs(tokenizer, questions, args.device, add_generation_prompt=True)

    with torch.inference_mode():
        head_inputs = dict(inputs)
        if getattr(args, "generate_text", True):
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "temperature": args.temperature if args.temperature > 0 else None,
                "top_p": args.top_p if args.temperature > 0 else None,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
            generated_ids = model.backbone.generate(**inputs, **generation_kwargs)
            prompt_len = inputs["input_ids"].shape[-1]
            new_ids = generated_ids[:, prompt_len:]
            llm_outputs = [
                output.strip()
                for output in tokenizer.batch_decode(new_ids, skip_special_tokens=False)
            ]

            head_inputs["input_ids"] = generated_ids
            head_inputs["attention_mask"] = torch.ones_like(generated_ids)
        else:
            llm_outputs = [""] * len(questions)

        outputs = model(**head_inputs, output_assistant_text=True)

    return [
        build_result(question, outputs, index, llm_outputs[index], stats)
        for index, question in enumerate(questions)
    ]


def main():
    args = parse_args()
    stats = load_regression_stats(args.stats_path)
    model, tokenizer = load_s1_omni(args.checkpoint_dir, args)

    if args.question is not None or args.question_file is not None:
        question = read_question(args)
        result = infer_once(model, tokenizer, question, stats, args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("S1Omni inference ready. 输入问题后回车，输入 q 退出。")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            break
        if not question or question.lower() == "q":
            break
        result = infer_once(model, tokenizer, question, stats, args)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
