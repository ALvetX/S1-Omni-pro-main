import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

TOPK_VALUES = (5, 10, 20)


def safe_literal_eval(x):
    """把字符串形式的 list/number 安全转成 Python 对象。"""
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def parse_answer(answer_field) -> List[int]:
    """
    answer 为真实位点索引，按 1-based 编号。
    这里返回 1-based 的整数列表，后面再转 0-based。
    """
    answer = safe_literal_eval(answer_field)

    if answer is None:
        return []

    if isinstance(answer, (int, float)):
        return [int(answer)]

    if isinstance(answer, (list, tuple, set, np.ndarray)):
        return [int(x) for x in answer]

    raise ValueError(f"无法解析 answer 字段: {answer_field}")


def parse_bit_string(bit_string_field) -> List[int]:
    """
    bit_string 例如 '0010010' -> [0, 0, 1, 0, 0, 1, 0]
    """
    # 如果原始字段是字符串且看起来像纯 0/1 位串，直接按位解析，避免 safe_literal_eval
    if isinstance(bit_string_field, str):
        s_raw = bit_string_field.strip()
        if s_raw != "":
            # 只要里面包含的非空字符都是 0/1，就认为是位串
            chars = [ch for ch in s_raw if ch in {"0", "1"}]
            if len(chars) > 0 and all(ch in {"0", "1"} for ch in chars):
                return [int(ch) for ch in chars]

    bit_string = safe_literal_eval(bit_string_field)

    # 如果字段是字符串（经过 safe_literal_eval 返回的），保留其中的 0/1 字符
    if isinstance(bit_string, str):
        s = bit_string.strip()
        s = "".join(ch for ch in s if ch in {"0", "1"})
        return [int(ch) for ch in s]

    # 如果字段是数字（例如 JSON 写成了未加引号的长数字），先转成字符串再处理
    if isinstance(bit_string, (int, float)):
        s = str(bit_string)
        s = "".join(ch for ch in s if ch in {"0", "1"})
        return [int(ch) for ch in s]

    if isinstance(bit_string, (list, tuple, np.ndarray)):
        return [int(x) for x in bit_string]

    raise ValueError(f"无法解析 bit_string 字段: {bit_string_field}")


def parse_probabilities(prob_field) -> List[float]:
    """
    probabilities 应该是每个位点的概率分数列表。
    """
    probs = safe_literal_eval(prob_field)

    if probs is None:
        return []

    if isinstance(probs, (list, tuple, np.ndarray)):
        return [float(x) for x in probs]

    raise ValueError(f"无法解析 probabilities 字段: {prob_field}")


def parse_true_length_from_question(question_field) -> Optional[int]:
    """
    从 question 中提取 <PROT>...</PROT> 之间的序列长度。
    如果无法提取，返回 None。
    """
    question = safe_literal_eval(question_field)
    if not isinstance(question, str):
        return None

    start_tag = "<PROT>"
    end_tag = "</PROT>"
    start_idx = question.find(start_tag)
    end_idx = question.find(end_tag, start_idx + len(start_tag))
    if start_idx == -1 or end_idx == -1:
        return None

    seq = "".join(question[start_idx + len(start_tag) : end_idx].split())
    return len(seq)


def build_true_labels(answer_1based: List[int], length: int) -> np.ndarray:
    """
    根据 answer 构造真实标签向量。
    answer 是 1-based，内部转成 0-based。
    """
    y_true = np.zeros(length, dtype=np.int32)

    for idx1 in answer_1based:
        idx0 = idx1 - 1
        if 0 <= idx0 < length:
            y_true[idx0] = 1
        else:
            print(
                f"Warning: answer 中存在越界索引: {idx1}，但当前样本长度为 {length}，已忽略该索引",
                file=sys.stderr,
            )
    return y_true


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算单组二分类指标。
    """
    result: Dict[str, float] = {}

    result["precision"] = precision_score(y_true, y_pred, zero_division=0)
    result["recall"] = recall_score(y_true, y_pred, zero_division=0)
    result["f1"] = f1_score(y_true, y_pred, zero_division=0)
    result["mcc"] = matthews_corrcoef(y_true, y_pred)

    if y_score is not None and len(np.unique(y_true)) == 2:
        result["auroc"] = roc_auc_score(y_true, y_score)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)
        result["aupr"] = auc(pr_recall, pr_precision)
    else:
        result["auroc"] = float("nan")
        result["aupr"] = float("nan")

    return result


def compute_topk_metrics(
    y_true: np.ndarray,
    y_score: Optional[np.ndarray],
    topk_values: tuple[int, ...] = TOPK_VALUES,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    positive_count = int(y_true.sum())

    def fill_nan(name: str):
        result[f"precision@{name}"] = float("nan")
        result[f"recall@{name}"] = float("nan")

    if y_score is None or len(y_true) == 0 or positive_count == 0:
        fill_nan("num_true")
        for k in topk_values:
            fill_nan(str(k))
        return result

    order = np.argsort(-y_score)

    def add_at_k(name: str, k: int):
        k = min(max(int(k), 1), len(y_true))
        selected = order[:k]
        hits = int(y_true[selected].sum())
        result[f"precision@{name}"] = float(hits / k)
        result[f"recall@{name}"] = float(hits / positive_count)

    add_at_k("num_true", positive_count)
    for k in topk_values:
        add_at_k(str(k), k)
    return result


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_score: Optional[np.ndarray],
) -> Dict[str, float]:
    if y_score is None or len(np.unique(y_true)) != 2:
        return {
            "best_f1_threshold": float("nan"),
            "best_f1": float("nan"),
            "best_f1_precision": float("nan"),
            "best_f1_recall": float("nan"),
        }

    pr_precision, pr_recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_values = (2 * pr_precision * pr_recall) / (pr_precision + pr_recall + 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    if len(thresholds) == 0:
        best_threshold = float("nan")
    elif best_idx >= len(thresholds):
        best_threshold = float(np.min(y_score))
    else:
        best_threshold = float(thresholds[best_idx])

    return {
        "best_f1_threshold": best_threshold,
        "best_f1": float(f1_values[best_idx]),
        "best_f1_precision": float(pr_precision[best_idx]),
        "best_f1_recall": float(pr_recall[best_idx]),
    }


def mean_ignore_nan(values: List[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def evaluate_jsonl(
    jsonl_path: str,
    pred_source: str = "bit_string",
    threshold: float = 0.5,
    use_record_threshold: bool = False,
) -> Dict[str, Any]:
    """
    读取 jsonl 并评估：
    - global/micro: 所有样本所有位点拼接后整体计算
    - macro: 每条样本先算，再求平均

    参数：
    - pred_source:
        "bit_string"    -> 用 bit_string 作为预测标签
        "probabilities" -> 用 probabilities + threshold 生成预测标签
    - threshold:
        pred_source="probabilities" 时使用的默认阈值
    - use_record_threshold:
        若为 True，则优先使用每条记录里的 threshold 字段
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {jsonl_path}")

    all_y_true = []
    all_y_pred = []
    all_y_score = []

    per_sample_metrics = []
    per_sample_topk_metrics = []
    num_samples = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            num_samples += 1

            answer_1based = parse_answer(record.get("answer", []))
            bit_pred = parse_bit_string(record.get("bit_string", ""))
            probs = parse_probabilities(record.get("probabilities", []))

            real_length = parse_true_length_from_question(record.get("question", ""))
            max_length = 99999999999
            if real_length is not None:
                if real_length > max_length:
                    target_length = max_length
                    print(
                        f"Warning: 第 {line_no} 行真实序列长度 {real_length} 超出 {max_length}，评估时截断为前 {max_length} 位。",
                        file=sys.stderr,
                    )
                else:
                    target_length = real_length
            else:
                if len(probs) > 0:
                    target_length = len(probs)
                elif len(bit_pred) > 0:
                    target_length = len(bit_pred)
                else:
                    raise ValueError(
                        f"第 {line_no} 行没有可用的 probabilities 或 bit_string"
                    )
                if target_length > max_length:
                    print(
                        f"Warning: 第 {line_no} 行预测长度({target_length}) 超出 {max_length}，已截断为前 {max_length} 位。",
                        file=sys.stderr,
                    )
                    target_length = max_length

            if len(probs) > target_length:
                probs = probs[:target_length]
            if len(bit_pred) > target_length:
                bit_pred = bit_pred[:target_length]

            length = target_length

            # 如果 bit_pred 只有 1 个元素（例如 JSON 中被写成了数字 0/1），
            # 且存在 probabilities 给出的期望长度，则将该值扩展为全长填充。
            if len(bit_pred) == 1 and length > 1:
                print(
                    f"Warning: 第 {line_no} 行 bit_string 只有 1 位，推测为单值（0/1），已扩展为长度 {length}。",
                    file=sys.stderr,
                )
                bit_pred = [bit_pred[0]] * length

            if len(bit_pred) > 0 and len(bit_pred) != length:
                # 如果同时存在 probabilities，优先使用 probabilities（可能 bit_string 被解析成单个数字）
                if len(probs) > 0:
                    print(
                        f"Warning: 第 {line_no} 行 bit_string 长度({len(bit_pred)}) 与 probabilities 长度({length}) 不一致，使用 probabilities 生成预测。",
                        file=sys.stderr,
                    )
                    bit_pred = []
                else:
                    raise ValueError(
                        f"第 {line_no} 行 bit_string 长度({len(bit_pred)}) "
                        f"与 probabilities 长度({length}) 不一致"
                    )

            y_true = build_true_labels(answer_1based, length)

            if pred_source == "bit_string":
                if len(bit_pred) == 0:
                    raise ValueError(
                        f"第 {line_no} 行没有 bit_string，无法用 bit_string 评估"
                    )
                y_pred = np.array(bit_pred, dtype=np.int32)

            elif pred_source == "probabilities":
                if len(probs) == 0:
                    raise ValueError(
                        f"第 {line_no} 行没有 probabilities，无法用概率阈值评估"
                    )

                th = threshold
                if use_record_threshold and ("threshold" in record):
                    th = float(record["threshold"])
                y_pred = (np.array(probs) >= th).astype(np.int32)

            else:
                raise ValueError("pred_source 只能是 'bit_string' 或 'probabilities'")

            y_score = np.array(probs, dtype=np.float64) if len(probs) > 0 else None

            sample_metrics = compute_binary_metrics(y_true, y_pred, y_score)
            per_sample_metrics.append(sample_metrics)
            per_sample_topk_metrics.append(compute_topk_metrics(y_true, y_score))

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            if y_score is not None:
                all_y_score.append(y_score)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_y_score = np.concatenate(all_y_score) if len(all_y_score) > 0 else None

    global_metrics = compute_binary_metrics(all_y_true, all_y_pred, all_y_score)
    threshold_sweep = compute_threshold_sweep(all_y_true, all_y_score)

    macro_metrics = {
        "precision": mean_ignore_nan([m["precision"] for m in per_sample_metrics]),
        "recall": mean_ignore_nan([m["recall"] for m in per_sample_metrics]),
        "f1": mean_ignore_nan([m["f1"] for m in per_sample_metrics]),
        "mcc": mean_ignore_nan([m["mcc"] for m in per_sample_metrics]),
        "auroc": mean_ignore_nan([m["auroc"] for m in per_sample_metrics]),
        "aupr": mean_ignore_nan([m["aupr"] for m in per_sample_metrics]),
    }
    topk_metrics: Dict[str, float] = {}
    if per_sample_topk_metrics:
        for key in per_sample_topk_metrics[0].keys():
            topk_metrics[key] = mean_ignore_nan([m[key] for m in per_sample_topk_metrics])

    summary = {
        "num_samples": num_samples,
        "total_positions": int(len(all_y_true)),
        "total_positive_labels": int(all_y_true.sum()),
        "total_predicted_positive": int(all_y_pred.sum()),
        "global_metrics": global_metrics,
        "macro_metrics": macro_metrics,
        "topk_metrics": topk_metrics,
        "threshold_sweep": threshold_sweep,
    }

    return summary


def print_metrics(result: Dict[str, Any]):
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"num_samples             : {result['num_samples']}")
    print(f"total_positions         : {result['total_positions']}")
    print(f"total_positive_labels   : {result['total_positive_labels']}")
    print(f"total_predicted_positive: {result['total_predicted_positive']}")

    print("\n" + "=" * 60)
    print("Global / Micro Metrics")
    print("=" * 60)
    for k, v in result["global_metrics"].items():
        print(f"{k:10s}: {v:.6f}" if not math.isnan(v) else f"{k:10s}: nan")

    # print("\n" + "=" * 60)
    # print("Macro Metrics")
    # print("=" * 60)
    # for k, v in result["macro_metrics"].items():
    #     print(f"{k:10s}: {v:.6f}" if not math.isnan(v) else f"{k:10s}: nan")

    # print("\n" + "=" * 60)
    # print("Top-K Metrics / Macro Per Sample")
    # print("=" * 60)
    # for k, v in result.get("topk_metrics", {}).items():
    #     print(f"{k:18s}: {v:.6f}" if not math.isnan(v) else f"{k:18s}: nan")

    # print("\n" + "=" * 60)
    # print("Threshold Sweep / Global")
    # print("=" * 60)
    # for k, v in result.get("threshold_sweep", {}).items():
    #     print(f"{k:20s}: {v:.6f}" if not math.isnan(v) else f"{k:20s}: nan")


if __name__ == "__main__":
    jsonl_file = "/data/home/zdhs0092/Code/S1-Omni-pro/output_protein_PPI_esm2-3b-weight10-crossattention8-esm8-epoch10.jsonl"
    # 默认使用 probabilities + threshold 进行评估（更稳健）
    result = evaluate_jsonl(
        jsonl_path=jsonl_file,
        pred_source="probabilities",
        threshold=0.9,
        use_record_threshold=False,
    )

    # 如果你想改成用 probabilities + threshold：
    # result = evaluate_jsonl(
    #     jsonl_path=jsonl_file,
    #     pred_source="probabilities",
    #     threshold=0.5,
    #     use_record_threshold=False,
    # )

    print_metrics(result)
