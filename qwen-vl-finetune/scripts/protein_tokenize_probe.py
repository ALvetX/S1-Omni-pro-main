#!/usr/bin/env python3
import argparse

from transformers import AutoTokenizer


DEFAULT_TEXT = (
    "T I K D V A K R A N V S T T T V S H V I N K T R F V A E E T R N A V W A A I K E L H Y S P S A V A R S L A V N H T K S I G L L A T S S E A A Y F A E I I E A V E K N C F Q K G Y T L I L G N A W N N L E K Q R A Y L S M M A Q K R V D G L L V M C S E Y P E P L L A M L E E Y R H I P M V V M D W G E A K A D F T D A V I D N A F E G G Y M A G R Y L I E R G H R E I G V I P G P L E R N T G A G R L A G F M K A M E E A M I K V P E S W I V Q G D F E P E S G Y R A M Q Q I L S Q P H R P T A V F C G G D I M A M G A L C A A D E M G L R V P Q D V S L I G Y D N V R N A R Y F T P A L T T I H Q P K D S L G E T A F N M L L D R I V N K R E E P Q S I E V H P R L I E R R S V A D G P F R D Y R"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        default="/data/group/wenge/xlf_model/S1-VL-32B-RL",
        help="Qwen3-VL model/tokenizer path.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast,
        trust_remote_code=True,
    )

    encoded = tokenizer(
        args.text,
        add_special_tokens=False,
        return_offsets_mapping=args.use_fast,
    )
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offsets = encoded.get("offset_mapping")

    prot_seq = args.text
    if args.text.startswith("<PROT>") and args.text.endswith("</PROT>"):
        prot_seq = args.text[len("<PROT>") : -len("</PROT>")]

    print(f"tokenizer class: {tokenizer.__class__.__name__}")
    print(f"use_fast: {args.use_fast}")
    print(f"input chars: {len(args.text)}")
    print(f"protein residue chars: {len(prot_seq)}")
    print(f"token count: {len(input_ids)}")
    print(f"one residue per token possible: {len(input_ids) == len(prot_seq)}")
    print()

    rows = []
    for idx, (token_id, token) in enumerate(zip(input_ids, tokens)):
        if offsets is None:
            rows.append((idx, token_id, token, ""))
        else:
            start, end = offsets[idx]
            rows.append((idx, token_id, token, repr(args.text[start:end])))

    if not args.show_all and len(rows) > 80:
        display_rows = rows[:40] + [("...", "...", "...", "...")] + rows[-40:]
    else:
        display_rows = rows

    if offsets is None:
        print("idx\ttoken_id\ttoken")
        for idx, token_id, token, _ in display_rows:
            print(f"{idx}\t{token_id}\t{token}")
    else:
        print("idx\ttoken_id\ttoken\toffset_text")
        for idx, token_id, token, offset_text in display_rows:
            print(f"{idx}\t{token_id}\t{token}\t{offset_text}")


if __name__ == "__main__":
    main()
