#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor import (
    _extract_protein_sequence_and_qwen_text,
    _find_subsequence,
    _space_protein_sequence,
)
from qwenvl.modeling_s1_protein import S1Protein


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="/nfs-12/liujunyi/S1-Omni-pro-main/output/protein_RNA_ep6_4")
    parser.add_argument("--question", default=None)
    parser.add_argument("--question_file", default=None) 
    parser.add_argument("--batch_file", default="/nfs-12/liujunyi/S1-Omni-pro-main/data/protein_pre_data/RNA_binding_site/test/protein_site_prediction-interaction_site-RNA_binding_site.jsonl")
    parser.add_argument("--output_file", default="/nfs-12/liujunyi/S1-Omni-pro-main/output/predict_protein/predictions_RNA.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument(
        "--auto_threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search completed predictions for the best threshold before writing results.",
    )
    parser.add_argument(
        "--optimize_threshold_metric",
        default="f1_then_mcc",
        choices=["f1", "mcc", "f1_then_mcc"],
        help="Metric used to choose the final threshold when --auto_threshold is enabled.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def build_messages(question: str):
    return [{"role": "user", "content": [{"type": "text", "text": question}]}]


def prepare_batch_inputs(tokenizer, questions: list[str], device: str, esm_tokenizer=None, use_esm2: bool = True):
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    normalized_questions = []
    protein_sequences = []
    for question in questions:
        if use_esm2:
            normalized_question, protein_sequence = _extract_protein_sequence_and_qwen_text(question)
        else:
            normalized_question, protein_sequence = _space_protein_sequence(question)
        normalized_questions.append(normalized_question)
        protein_sequences.append(protein_sequence)

    texts = [
        tokenizer.apply_chat_template(
            build_messages(question),
            tokenize=False,
            add_generation_prompt=False,
        )
        for question in normalized_questions
    ]
    inputs = tokenizer(
        texts,
        padding=True,
        return_tensors="pt",
    )
    if use_esm2:
        if esm_tokenizer is None:
            raise ValueError("esm_tokenizer is required when use_esm2=True")
        esm_inputs = esm_tokenizer(
            protein_sequences,
            padding=True,
            return_tensors="pt",
        )
        inputs["esm_input_ids"] = esm_inputs["input_ids"]
        inputs["esm_attention_mask"] = esm_inputs["attention_mask"]
    else:
        protein_token_mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        for index, protein_sequence in enumerate(protein_sequences):
            residue_token_ids = tokenizer.encode(
                " " + " ".join(protein_sequence),
                add_special_tokens=False,
            )
            residue_start = _find_subsequence(inputs["input_ids"][index].tolist(), residue_token_ids)
            if residue_start < 0:
                raise ValueError("failed to locate spaced protein residue tokens in tokenized input")
            protein_token_mask[index, residue_start : residue_start + len(residue_token_ids)] = True
        inputs["protein_token_mask"] = protein_token_mask
    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    protein_lengths = [len(sequence) for sequence in protein_sequences]
    return model_inputs, protein_lengths


def read_question(args) -> str:
    if args.question_file:
        return Path(args.question_file).read_text(encoding="utf-8").strip()
    if args.question is None:
        raise ValueError("missing question")
    return args.question


def _answer_from_item(item: Any):
    if isinstance(item, dict):
        return item.get("answer", item.get("output"))
    return None


def _record_from_item(item: Any) -> dict[str, Any]:
    return {
        "question": _text_from_item(item),
        "answer": _answer_from_item(item),
    }


def _text_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    for key in ("input", "text", "question"):
        if key in item:
            return str(item[key])
    if "messages" in item:
        parts = []
        for msg in item["messages"]:
            if msg.get("role") == "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type", "text") == "text"
                )
        text = "".join(parts).strip()
        if text:
            return text
    raise KeyError("batch item must be a string or contain input/text/question/messages")


def read_batch(path: str | Path) -> list[dict[str, Any]]:
    batch_path = Path(path)
    if batch_path.suffix == ".json":
        data = json.loads(batch_path.read_text(encoding="utf-8"))
        return [_record_from_item(item) for item in data]

    records = []
    with batch_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_record_from_item(json.loads(line)))
            except json.JSONDecodeError:
                records.append(_record_from_item(line))
    return records


def load_s1_protein(checkpoint_dir: str | Path, args):
    dtype = resolve_dtype(args.dtype)
    model = S1Protein.from_pretrained(
        str(checkpoint_dir),
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir),
        padding_side="right",
        use_fast=False,
    )
    esm_tokenizer = None
    if getattr(model, "use_esm2", False):
        esm_source = S1Protein.resolve_esm_tokenizer_source(
            str(checkpoint_dir),
            model.esm_model_name,
        )
        esm_tokenizer = AutoTokenizer.from_pretrained(
            esm_source,
            trust_remote_code=True,
        )
    model.set_tokenizer(tokenizer)
    model.to(args.device)
    model.eval()
    return model, tokenizer, esm_tokenizer


def build_result(
    question: str,
    probabilities: torch.Tensor,
    protein_length: int,
    threshold: float,
    answer: Any,
):
    probabilities = probabilities.squeeze(-1)[:protein_length]
    probs = [float(x) for x in probabilities.detach().float().cpu().tolist()]
    bits = ["1" if value >= threshold else "0" for value in probs]
    positive_indices_zero_based = [idx for idx, bit in enumerate(bits) if bit == "1"]
    return {
        "positive_indices": [idx + 1 for idx in positive_indices_zero_based],
        # "positive_indices_zero_based": positive_indices_zero_based,
        "answer": answer,
        "bit_string": "".join(bits),
        "question": question,
        "threshold": threshold,
        "probabilities": probs
    }


def parse_answer(answer_raw) -> list[int]:
    if isinstance(answer_raw, list):
        return [int(index) for index in answer_raw]
    if isinstance(answer_raw, str):
        text = answer_raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [int(index) for index in parsed]
    return []


def _answer_indices_for_prediction(pred: dict[str, Any]) -> list[int]:
    probs = pred.get("probabilities", [])
    n = len(probs)
    raw_answer = pred.get("answer", pred.get("output"))
    return [index - 1 for index in parse_answer(raw_answer) if 1 <= index <= n]


def _set_prediction_threshold(pred: dict[str, Any], threshold: float) -> None:
    probs = [float(value) for value in pred.get("probabilities", [])]
    bits = ["1" if value >= threshold else "0" for value in probs]
    pred["positive_indices"] = [idx + 1 for idx, bit in enumerate(bits) if bit == "1"]
    pred["bit_string"] = "".join(bits)
    pred["threshold"] = float(threshold)


def apply_threshold(preds: list[dict[str, Any]], threshold: float) -> None:
    for pred in preds:
        _set_prediction_threshold(pred, threshold)


def _labels_and_scores(preds: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    y_true = []
    y_score = []
    for pred in preds:
        probs = [float(value) for value in pred.get("probabilities", [])]
        answer_set = set(_answer_indices_for_prediction(pred))
        if not answer_set:
            continue
        y_true.extend(1 if idx in answer_set else 0 for idx in range(len(probs)))
        y_score.extend(probs)
    return y_true, y_score


def _confusion_counts_from_labels(
    y_true: list[int],
    y_score: list[float],
    threshold: float,
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for label, score in zip(y_true, y_score):
        is_true = bool(label)
        is_pred = float(score) >= threshold
        if is_true and is_pred:
            tp += 1
        elif is_true:
            fn += 1
        elif is_pred:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _binary_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / (denom ** 0.5)) if denom else 0.0
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
    }


def _threshold_metrics(
    y_true: list[int],
    y_score: list[float],
    threshold: float,
) -> dict[str, float]:
    return _binary_metrics_from_counts(
        *_confusion_counts_from_labels(y_true, y_score, threshold)
    )


def find_best_thresholds(preds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    y_true, y_score = _labels_and_scores(preds)
    if not y_score:
        metrics = _binary_metrics_from_counts(0, 0, 0, 0)
        row = {"threshold": 0.5, **metrics}
        return {"best_f1": row, "best_mcc": row}

    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    tp = fp = 0
    fn = total_pos
    tn = total_neg
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    no_positive_threshold = float(pairs[0][0]) + 1e-12
    empty_row = {
        "threshold": no_positive_threshold,
        **_binary_metrics_from_counts(tp, fp, tn, fn),
    }
    best_f1 = empty_row
    best_mcc = empty_row

    i = 0
    while i < len(pairs):
        threshold = float(pairs[i][0])
        j = i
        while j < len(pairs) and pairs[j][0] == threshold:
            if pairs[j][1]:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
            j += 1

        row = {"threshold": threshold, **_binary_metrics_from_counts(tp, fp, tn, fn)}
        if (
            row["F1"] > best_f1["F1"]
            or (
                row["F1"] == best_f1["F1"]
                and (
                    row["MCC"] > best_f1["MCC"]
                    or (
                        row["MCC"] == best_f1["MCC"]
                        and row["threshold"] > best_f1["threshold"]
                    )
                )
            )
        ):
            best_f1 = row

        if (
            row["MCC"] > best_mcc["MCC"]
            or (
                row["MCC"] == best_mcc["MCC"]
                and (
                    row["F1"] > best_mcc["F1"]
                    or (
                        row["F1"] == best_mcc["F1"]
                        and row["threshold"] > best_mcc["threshold"]
                    )
                )
            )
        ):
            best_mcc = row
        i = j

    return {"best_f1": best_f1, "best_mcc": best_mcc}


def _choose_threshold(
    best_thresholds: dict[str, dict[str, float]],
    fallback_threshold: float,
    optimize_metric: str,
) -> float:
    if optimize_metric == "f1":
        return float(best_thresholds.get("best_f1", {}).get("threshold", fallback_threshold))
    if optimize_metric == "mcc":
        return float(best_thresholds.get("best_mcc", {}).get("threshold", fallback_threshold))
    return float(best_thresholds.get("best_f1", {}).get("threshold", fallback_threshold))


def _rank_auc(y_true: list[int], y_score: list[float]) -> float:
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _score, label in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _precision_recall_auc(y_true: list[int], y_score: list[float]) -> float:
    total_pos = sum(y_true)
    if total_pos == 0:
        return float("nan")
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    points = [(0.0, 1.0)]
    tp = fp = 0
    i = 0
    while i < len(pairs):
        threshold = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == threshold:
            if pairs[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_pos
        points.append((recall, precision))

    area = 0.0
    for (recall_a, precision_a), (recall_b, precision_b) in zip(points, points[1:]):
        area += (recall_b - recall_a) * (precision_a + precision_b) / 2.0
    return area


def compute_metrics(preds: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    y_true, y_score = _labels_and_scores(preds)
    metrics = _threshold_metrics(y_true, y_score, threshold)
    metrics.update(
        {
            "AUROC": _rank_auc(y_true, y_score),
            "AUPR": _precision_recall_auc(y_true, y_score),
        }
    )
    return metrics


def finalize_predictions(
    preds: list[dict[str, Any]],
    threshold: float,
    auto_threshold: bool = False,
    optimize_metric: str = "f1_then_mcc",
) -> dict[str, Any]:
    y_true, y_score = _labels_and_scores(preds)
    can_optimize = bool(y_score) and sum(y_true) > 0
    best_thresholds = find_best_thresholds(preds) if auto_threshold and can_optimize else {}
    selected_threshold = (
        _choose_threshold(best_thresholds, threshold, optimize_metric)
        if best_thresholds
        else float(threshold)
    )
    apply_threshold(preds, selected_threshold)
    return {
        "selected_threshold": selected_threshold,
        "best_f1": best_thresholds.get("best_f1"),
        "best_mcc": best_thresholds.get("best_mcc"),
        "metrics": compute_metrics(preds, selected_threshold),
    }


def print_metrics_summary(summary: dict[str, Any]) -> None:
    print(f"Selected threshold      {summary['selected_threshold']:.6f}")
    if summary.get("best_f1"):
        best_f1 = summary["best_f1"]
        print(
            "Best F1 threshold      "
            f"{best_f1['threshold']:.6f} "
            f"(F1={best_f1['F1']:.4f}, MCC={best_f1['MCC']:.4f})"
        )
    if summary.get("best_mcc"):
        best_mcc = summary["best_mcc"]
        print(
            "Best MCC threshold     "
            f"{best_mcc['threshold']:.6f} "
            f"(F1={best_mcc['F1']:.4f}, MCC={best_mcc['MCC']:.4f})"
        )
    for name in ("Precision", "Recall", "F1", "MCC", "AUROC", "AUPR"):
        print(f"{name:<22} {summary['metrics'][name]:.4f}")


def infer_batch(model: S1Protein, tokenizer, records: list[dict[str, Any]], args):
    if not records:
        return []

    questions = [record["question"] for record in records]
    esm_tokenizer = getattr(model, "esm_tokenizer", None)
    inputs, protein_lengths = prepare_batch_inputs(
        tokenizer,
        questions,
        args.device,
        esm_tokenizer=esm_tokenizer,
        use_esm2=getattr(model, "use_esm2", False),
    )
    with torch.inference_mode():
        outputs = model(**inputs, threshold=args.threshold)

    return [
        build_result(
            question,
            outputs.probabilities[index],
            protein_lengths[index],
            args.threshold,
            records[index].get("answer"),
        )
        for index, question in enumerate(questions)
    ]


def write_results(results: list[dict[str, Any]], output_file: str | None):
    if output_file is None:
        payload: dict[str, Any] | list[dict[str, Any]]
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    model, tokenizer, esm_tokenizer = load_s1_protein(args.checkpoint_dir, args)
    model.esm_tokenizer = esm_tokenizer

    if args.batch_file:
        records = read_batch(args.batch_file)
    else:
        records = [{"question": read_question(args), "answer": None}]
    batch = 10
    all_results = []
    # 添加tqdm进度条显示    
    for i in tqdm(range(0, len(records), batch), desc="Processing batches", total=(len(records) + batch - 1) // batch):
        batch_records = records[i:i + batch]
        results = infer_batch(model, tokenizer, batch_records, args)
        all_results.extend(results)

    summary = finalize_predictions(
        all_results,
        threshold=args.threshold,
        auto_threshold=args.auto_threshold,
        optimize_metric=args.optimize_threshold_metric,
    )
    print_metrics_summary(summary)
    if args.output_file is not None:
        output_path = Path(args.output_file)
        if output_path.exists():
            output_path.unlink()
    write_results(all_results, args.output_file)


if __name__ == "__main__":
    main()
