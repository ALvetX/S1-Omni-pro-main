import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.modeling_s1_omni import S1Omni
from qwenvl.train.argument import DataArguments, ModelArguments


@dataclass
class PrecomputeArguments:
    output_path: str = field(metadata={"help": "Path to save pooled hidden cache."})
    per_device_batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    bf16: bool = field(default=True)
    cache_dtype: str = field(default="bfloat16")
    overwrite: bool = field(default=False)
    log_steps: int = field(default=20)


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.distributed.get_rank(), world_size, torch.device(f"cuda:{local_rank}")


def _cache_dtype(name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported cache_dtype: {name}")


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if key in {"route_id", "label", "task_type", "cache_index"}:
            moved[key] = value
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _save_cache(path: str, pooled_hidden: torch.Tensor, metadata: dict):
    torch.save(
        {
            "pooled_hidden": pooled_hidden.contiguous(),
            "metadata": metadata,
        },
        path,
    )


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, PrecomputeArguments)
    )
    model_args, data_args, precompute_args = parser.parse_args_into_dataclasses()

    rank, world_size, device = _init_distributed()
    output_path = Path(precompute_args.output_path)
    shard_path = output_path.with_suffix(output_path.suffix + f".rank{rank}")

    if output_path.exists() and not precompute_args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite True to replace it")
    if shard_path.exists() and not precompute_args.overwrite:
        raise FileExistsError(f"{shard_path} exists; pass --overwrite True to replace it")

    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    processor = transformers.AutoProcessor.from_pretrained(model_args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )
    data_args.pooled_hidden_cache_path = None
    data_args.include_cache_index = True
    data_args.model_type = "qwen3vl"
    data_module = make_supervised_data_module(processor, data_args=data_args)
    dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]

    model = S1Omni.from_pretrained(
        model_args.model_name_or_path,
        dtype=(torch.bfloat16 if precompute_args.bf16 else None),
    )
    model.set_tokenizer(tokenizer)
    model.freeze_backbone()
    model.eval()
    model.to(device)

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else SequentialSampler(dataset)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=precompute_args.per_device_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=precompute_args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    pooled_chunks = []
    index_chunks = []
    target_dtype = _cache_dtype(precompute_args.cache_dtype)

    with torch.inference_mode():
        for step, batch in enumerate(dataloader, start=1):
            cache_index = batch.pop("cache_index")
            batch.pop("route_id", None)
            batch.pop("label", None)
            batch.pop("task_type", None)
            batch.pop("labels", None)
            batch = _move_batch_to_device(batch, device)

            pooled, _ = model.compute_pooled_hidden(**batch)
            pooled_chunks.append(pooled.detach().to(device="cpu", dtype=target_dtype))
            index_chunks.append(cache_index.cpu())

            if rank == 0 and precompute_args.log_steps > 0 and step % precompute_args.log_steps == 0:
                done = step * precompute_args.per_device_batch_size * world_size
                print(f"precomputed approximately {min(done, len(dataset))}/{len(dataset)} samples", flush=True)

    shard_pooled = torch.cat(pooled_chunks, dim=0) if pooled_chunks else torch.empty(0)
    shard_indices = torch.cat(index_chunks, dim=0) if index_chunks else torch.empty(0, dtype=torch.long)
    torch.save(
        {
            "pooled_hidden": shard_pooled,
            "indices": shard_indices,
            "metadata": {
                "model_name_or_path": model_args.model_name_or_path,
                "annotation_path": data_args.annotation_path,
                "data_path": data_args.data_path,
                "num_samples": len(dataset),
                "world_size": world_size,
                "rank": rank,
                "dtype": str(shard_pooled.dtype),
            },
        },
        shard_path,
    )

    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0:
        shards = []
        for shard_rank in range(world_size):
            current_shard_path = output_path.with_suffix(output_path.suffix + f".rank{shard_rank}")
            try:
                shard = torch.load(current_shard_path, map_location="cpu", weights_only=True)
            except TypeError:
                shard = torch.load(current_shard_path, map_location="cpu")
            shards.append(shard)

        nonempty_pooled = next(shard["pooled_hidden"] for shard in shards if shard["pooled_hidden"].numel() > 0)
        hidden_size = nonempty_pooled.size(1)
        pooled_hidden = torch.empty(
            len(dataset),
            hidden_size,
            dtype=nonempty_pooled.dtype,
        )
        seen = torch.zeros(len(dataset), dtype=torch.bool)
        for shard in shards:
            indices = shard["indices"].long()
            pooled_hidden[indices] = shard["pooled_hidden"]
            seen[indices] = True

        if not bool(seen.all()):
            missing = (~seen).nonzero(as_tuple=False).flatten()[:20].tolist()
            raise RuntimeError(f"missing pooled hidden rows, first missing indices: {missing}")

        _save_cache(
            str(output_path),
            pooled_hidden,
            {
                "model_name_or_path": model_args.model_name_or_path,
                "annotation_path": data_args.annotation_path,
                "data_path": data_args.data_path,
                "num_samples": len(dataset),
                "hidden_size": hidden_size,
                "dtype": str(pooled_hidden.dtype),
            },
        )
        print(f"saved pooled hidden cache to {output_path}", flush=True)

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
