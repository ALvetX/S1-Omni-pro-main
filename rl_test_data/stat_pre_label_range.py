#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def to_number(value, file_path, line_no):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{file_path}:{line_no} label is not a number: {value!r}')
    return float(value)


def scan_file(file_path):
    count = 0
    min_label = None
    max_label = None

    with file_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if 'label' not in item:
                raise KeyError(f'{file_path}:{line_no} missing label')

            label = to_number(item['label'], file_path, line_no)
            count += 1
            min_label = label if min_label is None else min(min_label, label)
            max_label = label if max_label is None else max(max_label, label)

    return count, min_label, max_label


def format_number(value):
    if value is None:
        return 'NA'
    return f'{value:g}'


def main():
    parser = argparse.ArgumentParser(
        description='统计目录下文件名包含 pre 的 JSONL 文件中 label 的范围。')
    parser.add_argument(
        '--dir',
        type=Path,
        default=Path(__file__).resolve().parent,
        help='待统计目录，默认是脚本所在目录。')
    parser.add_argument(
        '--pattern',
        default='*pre*.jsonl',
        help='文件匹配模式，默认是 *pre*.jsonl。')
    args = parser.parse_args()

    data_dir = args.dir.resolve()
    files = sorted(path for path in data_dir.iterdir()
                   if path.is_file() and path.match(args.pattern))

    if not files:
        raise SystemExit(f'No files matched {args.pattern!r} in {data_dir}')

    total_count = 0
    overall_min = None
    overall_max = None

    print(f'Directory: {data_dir}')
    print(f'Pattern: {args.pattern}')
    print()
    print(f'{"file":<32} {"count":>8} {"min_label":>14} {"max_label":>14}')
    print('-' * 72)

    for file_path in files:
        count, min_label, max_label = scan_file(file_path)
        total_count += count
        if min_label is not None:
            overall_min = min_label if overall_min is None else min(overall_min, min_label)
            overall_max = max_label if overall_max is None else max(overall_max, max_label)

        print(f'{file_path.name:<32} {count:>8} '
              f'{format_number(min_label):>14} {format_number(max_label):>14}')

    print('-' * 72)
    print(f'{"OVERALL":<32} {total_count:>8} '
          f'{format_number(overall_min):>14} {format_number(overall_max):>14}')


if __name__ == '__main__':
    main()
