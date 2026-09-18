from __future__ import annotations

import contextlib
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _m2_utils import NvidiaSmiSampler, timed_cuda_loop, write_json
from game2048.m2_models import ResidualMLP2048, Transformer2048, trainable_parameter_count

OUTPUT = ROOT / "m2_gpu_benchmark.json"
BATCHES = [256, 512, 1024, 2048, 4096, 8192]
DEVICE = torch.device("cuda")
SEED = 20260918
MODELS = {"Transformer2048": Transformer2048, "ResidualMLP2048": ResidualMLP2048}


def autocast_for(precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_model(model_cls, execution: str):
    torch.manual_seed(SEED)
    model = model_cls().to(DEVICE)
    if execution == "compile":
        model = torch.compile(model)
    return model


def make_boards(batch: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    cpu = torch.randint(0, 22, (batch, 16), dtype=torch.uint8, generator=g)
    return cpu.to(DEVICE)


def inference_point(model_cls, batch: int, precision: str, execution: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = make_model(model_cls, execution).eval()
        boards = make_boards(batch, SEED + batch)
        with torch.inference_mode():
            def body():
                with autocast_for(precision):
                    out = model(boards)
                if not torch.isfinite(out).all():
                    raise RuntimeError("non-finite inference output")

            with NvidiaSmiSampler() as sampler:
                iterations, elapsed = timed_cuda_loop(torch, body)
        return {
            "status": "PASS",
            "batch": batch,
            "precision": precision,
            "execution": execution,
            "iterations": iterations,
            "elapsed_s": elapsed,
            "latency_ms": elapsed / iterations * 1000.0,
            "states_per_s": batch * iterations / elapsed,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "gpu": sampler.summary(),
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"status": "OOM", "batch": batch, "precision": precision, "execution": execution, "error": str(exc)}
    except Exception as exc:
        torch.cuda.empty_cache()
        return {
            "status": "FAILED",
            "batch": batch,
            "precision": precision,
            "execution": execution,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
        }
def training_point(model_cls, batch: int, precision: str, execution: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = make_model(model_cls, execution).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        states = make_boards(batch, SEED + batch + 1)
        afterstates = make_boards(batch, SEED + batch + 2)
        g = torch.Generator(device=DEVICE).manual_seed(SEED + batch + 3)
        q_target = torch.randn((batch, 4), device=DEVICE, generator=g)
        v_target = torch.randn((batch,), device=DEVICE, generator=g)
        a_target = torch.randn((batch,), device=DEVICE, generator=g)

        def body():
            optimizer.zero_grad(set_to_none=True)
            with autocast_for(precision):
                q, v = model.forward_state(states)
                a = model.forward_afterstate(afterstates)
                loss = (
                    F.mse_loss(q, q_target)
                    + 0.5 * F.mse_loss(v, v_target)
                    + 0.5 * F.mse_loss(a, a_target)
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with NvidiaSmiSampler() as sampler:
            iterations, elapsed = timed_cuda_loop(torch, body)
        gradients_finite = all(
            p.grad is None or torch.isfinite(p.grad).all().item()
            for p in model.parameters()
        )
        if not gradients_finite:
            raise RuntimeError("non-finite gradients")
        return {
            "status": "PASS",
            "batch": batch,
            "precision": precision,
            "execution": execution,
            "iterations": iterations,
            "elapsed_s": elapsed,
            "step_ms": elapsed / iterations * 1000.0,
            "samples_per_s": batch * iterations / elapsed,
            "board_forwards_per_s": 2 * batch * iterations / elapsed,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "gpu": sampler.summary(),
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"status": "OOM", "batch": batch, "precision": precision, "execution": execution, "error": str(exc)}
    except Exception as exc:
        torch.cuda.empty_cache()
        return {
            "status": "FAILED",
            "batch": batch,
            "precision": precision,
            "execution": execution,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def best_point(points: list[dict], metric: str) -> dict | None:
    valid = [p for p in points if p.get("status") == "PASS" and metric in p]
    return max(valid, key=lambda p: p[metric]) if valid else None


def main() -> int:
    if not torch.cuda.is_available():
        write_json(OUTPUT, {"pass": False, "reason": "CUDA unavailable"})
        return 2
    bf16 = bool(torch.cuda.is_bf16_supported())
    precisions = ["fp32", "bf16"] if bf16 else ["fp32", "fp16"]
    payload = {
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16_supported": bf16,
            "fp16_required": not bf16,
        },
        "models": {},
    }
    overall_ok = True
    for name, model_cls in MODELS.items():
        model_data = {
            "parameter_count": trainable_parameter_count(model_cls()),
            "inference_eager": [],
            "training_eager": [],
            "compile_comparisons": [],
        }
        for precision in precisions:
            for batch in BATCHES:
                inf = inference_point(model_cls, batch, precision, "eager")
                train = training_point(model_cls, batch, precision, "eager")
                model_data["inference_eager"].append(inf)
                model_data["training_eager"].append(train)
                print(name, "inference", inf)
                print(name, "training", train)

        best_inf = best_point(model_data["inference_eager"], "states_per_s")
        best_train = best_point(model_data["training_eager"], "samples_per_s")
        model_data["best_inference_eager"] = best_inf
        model_data["best_training_eager"] = best_train
        if best_inf is None or best_train is None:
            overall_ok = False
        else:
            compile_batches = sorted({1024, int(best_inf["batch"])})
            precision = str(best_inf["precision"])
            for batch in compile_batches:
                inf_compile = inference_point(model_cls, batch, precision, "compile")
                train_compile = training_point(model_cls, batch, precision, "compile")
                model_data["compile_comparisons"].append(
                    {"batch": batch, "precision": precision, "inference": inf_compile, "training": train_compile}
                )
                print(name, "compile", batch, inf_compile, train_compile)

            candidates = [best_inf]
            for item in model_data["compile_comparisons"]:
                point = item["inference"]
                if point.get("status") == "PASS":
                    candidates.append(point)
            selected = max(candidates, key=lambda p: p["states_per_s"])
            model_data["selected_inference"] = selected
        payload["models"][name] = model_data

    payload["pass"] = overall_ok
    write_json(OUTPUT, payload)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
