#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_CHECKPOINT = "/data/home/zdhs0092/Code/S1-Omni-pro/output/s1_omni_ep150"
DEFAULT_STATS = "/data/home/zdhs0092/Code/S1-Omni-pro/output/s1_omni_ep150/s1_omni_regression_label_stats.json"


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    llm_output: str
    final_prediction: int | float


@dataclass
class ModelReplica:
    index: int
    device: str
    model: Any
    tokenizer: Any
    args: SimpleNamespace


@dataclass
class PendingRequest:
    question: str
    future: asyncio.Future


def build_args(cli_args: argparse.Namespace | None = None) -> SimpleNamespace:
    def value(name: str, env_name: str, default: Any, cast=str) -> Any:
        cli_value = getattr(cli_args, name, None)
        if cli_value is not None:
            return cli_value
        env_value = os.getenv(env_name)
        if env_value is not None:
            return cast(env_value)
        return default

    return SimpleNamespace(
        checkpoint_dir=value("checkpoint_dir", "S1_OMNI_CHECKPOINT", DEFAULT_CHECKPOINT),
        stats_path=value("stats_path", "S1_OMNI_STATS", DEFAULT_STATS),
        device=value("device", "S1_OMNI_DEVICE", "cuda:0"),
        devices=value("devices", "S1_OMNI_DEVICES", None),
        dtype=value("dtype", "S1_OMNI_DTYPE", "bf16"),
        attn_implementation=value(
            "attn_implementation",
            "S1_OMNI_ATTN_IMPLEMENTATION",
            "flash_attention_2",
        ),
        max_new_tokens=value("max_new_tokens", "S1_OMNI_MAX_NEW_TOKENS", 4096, int),
        temperature=value("temperature", "S1_OMNI_TEMPERATURE", 0.2, float),
        top_p=value("top_p", "S1_OMNI_TOP_P", 0.95, float),
        generate_text=value("generate_text", "S1_OMNI_GENERATE_TEXT", True, lambda x: x.lower() in {"1", "true", "yes"}),
        batch_size=value("batch_size", "S1_OMNI_BATCH_SIZE", 1, int),
        batch_wait_ms=value("batch_wait_ms", "S1_OMNI_BATCH_WAIT_MS", 10.0, float),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for replica parallelism, e.g. cuda:0,cuda:1.",
    )
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--generate_text", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--batch_wait_ms", type=float, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


settings = build_args()
replicas: list[ModelReplica] = []
request_queue: asyncio.Queue[PendingRequest] | None = None
worker_tasks: list[asyncio.Task] = []
stats: dict[str, float] | None = None
infer_once_fn: Any | None = None
infer_batch_fn: Any | None = None


def resolve_devices() -> list[str]:
    if settings.devices:
        devices = [device.strip() for device in settings.devices.split(",") if device.strip()]
        if not devices:
            raise ValueError("--devices is set but no valid devices were found")
        return devices
    return [settings.device]


def load_model_once() -> None:
    global stats, infer_once_fn, infer_batch_fn
    if replicas:
        return
    if settings.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if settings.batch_wait_ms < 0:
        raise ValueError("--batch_wait_ms must be >= 0")
    from infer_s1_omni_checkpoint import infer_batch, infer_once, load_regression_stats, load_s1_omni

    stats = load_regression_stats(settings.stats_path)
    infer_once_fn = infer_once
    infer_batch_fn = infer_batch
    devices = resolve_devices()

    for index, device in enumerate(devices):
        replica_args = SimpleNamespace(**vars(settings))
        replica_args.device = device
        model, tokenizer = load_s1_omni(settings.checkpoint_dir, replica_args)
        replicas.append(
            ModelReplica(
                index=index,
                device=device,
                model=model,
                tokenizer=tokenizer,
                args=replica_args,
            )
        )

async def replica_worker(replica: ModelReplica) -> None:
    assert request_queue is not None
    assert stats is not None
    assert infer_batch_fn is not None

    while True:
        first = await request_queue.get()
        batch = [first]
        deadline = asyncio.get_running_loop().time() + (settings.batch_wait_ms / 1000.0)

        while len(batch) < settings.batch_size:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(request_queue.get(), timeout=timeout))
            except asyncio.TimeoutError:
                break

        active_batch = [item for item in batch if not item.future.cancelled()]
        if not active_batch:
            for _ in batch:
                request_queue.task_done()
            continue

        try:
            results = await asyncio.to_thread(
                infer_batch_fn,
                replica.model,
                replica.tokenizer,
                [item.question for item in active_batch],
                stats,
                replica.args,
            )
        except Exception as exc:
            for item in active_batch:
                if not item.future.cancelled():
                    item.future.set_exception(RuntimeError(f"{replica.device}: {exc}"))
        else:
            for item, result in zip(active_batch, results):
                if not item.future.cancelled():
                    item.future.set_result(result)
        finally:
            for _ in batch:
                request_queue.task_done()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global request_queue, worker_tasks
    load_model_once()
    request_queue = asyncio.Queue()
    worker_tasks = [
        asyncio.create_task(replica_worker(replica), name=f"s1-omni-replica-{replica.index}")
        for replica in replicas
    ]
    try:
        yield
    finally:
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)


app = FastAPI(title="S1-Omni Inference Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ready": bool(replicas) and stats is not None,
        "checkpoint_dir": settings.checkpoint_dir,
        "devices": [replica.device for replica in replicas],
        "replicas": len(replicas),
        "queue_size": request_queue.qsize() if request_queue is not None else 0,
        "generate_text": settings.generate_text,
        "max_new_tokens": settings.max_new_tokens,
        "batch_size": settings.batch_size,
        "batch_wait_ms": settings.batch_wait_ms,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    if not replicas or request_queue is None or stats is None or infer_batch_fn is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    try:
        future = asyncio.get_running_loop().create_future()
        await request_queue.put(PendingRequest(question=request.question.strip(), future=future))
        result = await future
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        llm_output=result["llm_output"],
        final_prediction=result["final_prediction"],
    )


def main() -> None:
    global settings
    args = parse_args()
    settings = build_args(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
