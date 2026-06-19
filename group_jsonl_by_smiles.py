#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "/data/home/zdhs0092/Code/S1-Omni-pro/"
    "output_protein_iron_binding_site_mlp_test_esm2-3b-weight20-ep6.jsonl"
)

SMILES_PATTERN = re.compile(r"<SMILES>(.*?)</SMILES>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统计 JSONL question 字段中 <SMILES>...</SMILES> 的内容，"
            "并把相同 SMILES 内容对应的数据保存到同一个 JSONL 文件。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入 JSONL 文件，默认: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认: 输入文件名去掉 .jsonl 后加 _by_smiles",
    )
    parser.add_argument(
        "--field",
        default="question",
        help="需要提取 SMILES 标签的字段名，默认: question",
    )
    parser.add_argument(
        "--include-no-smiles",
        action="store_true",
        help="是否把没有 <SMILES> 标签的数据保存到 __NO_SMILES__.jsonl",
    )
    return parser.parse_args()


def extract_smiles(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    return [match.strip() for match in SMILES_PATTERN.findall(text)]


def safe_group_filename(smiles: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "_", smiles).strip("._-+")
    if not slug:
        slug = "empty"
    slug = slug[:80]
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:10]
    return f"smiles_{slug}_{digest}.jsonl"


def write_jsonl_line(handle, item: dict[str, Any]) -> None:
    json.dump(item, handle, ensure_ascii=False)
    handle.write("\n")


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir or input_path.with_suffix("").with_name(
        f"{input_path.with_suffix('').name}_by_smiles"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    file_map: dict[str, str] = {}
    handles = {}
    total_lines = 0
    invalid_json_lines = 0
    no_smiles_lines = 0

    try:
        with input_path.open("r", encoding="utf-8") as reader:
            for line_no, line in enumerate(reader, start=1):
                line = line.strip()
                if not line:
                    continue

                total_lines += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json_lines += 1
                    print(f"跳过非法 JSON 行: {line_no}")
                    continue

                smiles_values = extract_smiles(item.get(args.field, ""))
                if not smiles_values:
                    no_smiles_lines += 1
                    if args.include_no_smiles:
                        no_smiles_path = output_dir / "__NO_SMILES__.jsonl"
                        handle = handles.get("__NO_SMILES__")
                        if handle is None:
                            handle = no_smiles_path.open("w", encoding="utf-8")
                            handles["__NO_SMILES__"] = handle
                        write_jsonl_line(handle, item)
                    continue

                for smiles in sorted(set(smiles_values)):
                    counts[smiles] += smiles_values.count(smiles)
                    filename = file_map.get(smiles)
                    if filename is None:
                        filename = safe_group_filename(smiles)
                        file_map[smiles] = filename

                    handle = handles.get(smiles)
                    if handle is None:
                        handle = (output_dir / filename).open("w", encoding="utf-8")
                        handles[smiles] = handle
                    write_jsonl_line(handle, item)
    finally:
        for handle in handles.values():
            handle.close()

    sorted_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    summary = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "field": args.field,
        "total_json_lines": total_lines,
        "invalid_json_lines": invalid_json_lines,
        "no_smiles_lines": no_smiles_lines,
        "unique_smiles_count": len(counts),
        "smiles_counts": [
            {
                "smiles": smiles,
                "count": count,
                "file": file_map[smiles],
            }
            for smiles, count in sorted_items
        ],
    }

    with (output_dir / "counts.json").open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, ensure_ascii=False, indent=2)
        writer.write("\n")

    with (output_dir / "counts.tsv").open("w", encoding="utf-8") as writer:
        writer.write("smiles\tcount\tfile\n")
        for smiles, count in sorted_items:
            writer.write(f"{smiles}\t{count}\t{file_map[smiles]}\n")

    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}")
    print(f"总 JSON 行数: {total_lines}")
    print(f"非法 JSON 行数: {invalid_json_lines}")
    print(f"无 SMILES 标签行数: {no_smiles_lines}")
    print(f"SMILES 内容种类数: {len(counts)}")
    print("SMILES 统计:")
    for smiles, count in sorted_items:
        print(f"  {smiles}: {count} -> {file_map[smiles]}")


if __name__ == "__main__":
    main()
