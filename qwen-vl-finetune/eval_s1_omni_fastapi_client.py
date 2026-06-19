#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_DATASET = "/data/home/zdhs0092/Code/S1-Omni-pro/rl_test_data/chem_pre_lipo.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--output_file", default="/data/home/zdhs0092/Code/S1-Omni-pro/rl_test_data/chem_pre_lipo_llmpredict.jsonl")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()


def extract_question(item: dict[str, Any]) -> str:
    messages = item.get("messages") or []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [part.get("text", "") for part in content if part.get("type") == "text"]
                return "\n".join(texts).strip()
    raise ValueError("missing user question in messages")


def extract_label(item: dict[str, Any]) -> float:
    if "solution" not in item:
        raise ValueError("missing solution")
    return float(item["solution"])


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_jsonl(path: str | Path, limit: int | None = None):
    with Path(path).open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if line:
                yield index, json.loads(line)


def rmse(predictions: list[float], labels: list[float]) -> float:
    if not predictions:
        return float("nan")
    mse = sum((pred - label) ** 2 for pred, label in zip(predictions, labels)) / len(predictions)
    return math.sqrt(mse)


def evaluate_one(index: int, item: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], float | None, float | None]:
    try:
        question = extract_question(item)
        label = extract_label(item)
        response = post_json(args.url, {"question": question}, args.timeout)
        prediction = float(response["final_prediction"])
    except (ValueError, KeyError, urllib.error.URLError, TimeoutError) as exc:
        record = {
            "index": index,
            "ok": False,
            "error": str(exc),
            "raw_item": item,
        }
        return record, None, None

    error = prediction - label
    record = {
        "index": index,
        "ok": True,
        "question": question,
        "label": label,
        "prediction": prediction,
        "rmse": abs(error),
        "llm_output": response.get("llm_output", ""),
    }
    return record, prediction, label


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_file) if args.output_file else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle = output_path.open("w", encoding="utf-8") if output_path else None

    predictions: list[float] = []
    labels: list[float] = []
    failures = 0
    start_time = time.time()
    samples = list(iter_jsonl(args.dataset, args.limit))

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for index, item in samples:
                futures.append(executor.submit(evaluate_one, index, item, args))
                if args.sleep > 0:
                    time.sleep(args.sleep)

            completed_futures = as_completed(futures)
            if tqdm is not None:
                completed_futures = tqdm(
                    completed_futures,
                    total=len(futures),
                    desc="Evaluating",
                    unit="sample",
                )

            for future in completed_futures:
                record, prediction, label = future.result()
                index = record["index"]

                if not record["ok"]:
                    failures += 1
                    print(f"[{index}] failed: {record['error']}", file=sys.stderr)
                else:
                    assert prediction is not None
                    assert label is not None
                    predictions.append(prediction)
                    labels.append(label)
                    print(
                        f"[{index}] pred={prediction:.6f} label={label:.6f} "
                        f"sample_rmse={record['rmse']:.6f} "
                        f"running_rmse={rmse(predictions, labels):.6f}",
                        flush=True,
                    )

                if output_handle is not None:
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()

    elapsed = time.time() - start_time
    metrics = {
        "count": len(predictions),
        "failures": failures,
        "rmse": rmse(predictions, labels),
        "elapsed_seconds": elapsed,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if failures == 0 and predictions else 1


if __name__ == "__main__":
    raise SystemExit(main())
