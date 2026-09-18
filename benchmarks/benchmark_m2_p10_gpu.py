from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _m2_utils import NvidiaSmiSampler, timed_cuda_loop, write_json
from benchmark_m2_gpu import inference_point
from game2048.m2_models import ResidualMLP2048

OUT = ROOT / "m2_gpu_benchmark.json"
DEVICE = torch.device("cuda")
SEED = 20260918


def autocast_bf16():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class CompiledTrainingWrapper(nn.Module):
    """Single forward(states, afterstates) -> Q, V, A for valid compile evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.model = ResidualMLP2048()

    def forward(
        self, states: torch.Tensor, afterstates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_features = self.model.encode(states)
        after_features = self.model.encode(afterstates)
        q = self.model.q_head(state_features)
        v = self.model.value_head(state_features).squeeze(-1)
        a = self.model.afterstate_head(after_features).squeeze(-1)
        return q, v, a


def make_boards(batch: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    cpu = torch.randint(0, 22, (batch, 16), dtype=torch.uint8, generator=g)
    return cpu.to(DEVICE)


def corrected_compile_training_point(batch: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        torch.manual_seed(SEED)
        wrapper = CompiledTrainingWrapper().to(DEVICE).train()
        compiled = torch.compile(wrapper)
        optimizer = torch.optim.AdamW(
            compiled.parameters(), lr=3e-4, weight_decay=1e-4
        )
        states = make_boards(batch, SEED + batch + 101)
        afterstates = make_boards(batch, SEED + batch + 102)
        g = torch.Generator(device=DEVICE).manual_seed(SEED + batch + 103)
        q_target = torch.randn((batch, 4), device=DEVICE, generator=g)
        v_target = torch.randn((batch,), device=DEVICE, generator=g)
        a_target = torch.randn((batch,), device=DEVICE, generator=g)

        def body():
            optimizer.zero_grad(set_to_none=True)
            with autocast_bf16():
                q, v, a = compiled(states, afterstates)
                loss = (
                    F.mse_loss(q, q_target)
                    + 0.5 * F.mse_loss(v, v_target)
                    + 0.5 * F.mse_loss(a, a_target)
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(compiled.parameters(), 1.0)
            optimizer.step()

        with NvidiaSmiSampler() as sampler:
            iterations, elapsed = timed_cuda_loop(torch, body)

        gradients_finite = all(
            p.grad is None or torch.isfinite(p.grad).all().item()
            for p in compiled.parameters()
        )
        if not gradients_finite:
            raise RuntimeError("non-finite gradients")

        return {
            "status": "PASS",
            "evidence": "single compiled wrapper forward returns Q,V,A",
            "batch": batch,
            "precision": "bf16",
            "execution": "compile",
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
        return {
            "status": "OOM",
            "evidence": "single compiled wrapper forward returns Q,V,A",
            "batch": batch,
            "precision": "bf16",
            "execution": "compile",
            "error": str(exc),
        }
    except Exception as exc:
        torch.cuda.empty_cache()
        unsupported = type(exc).__name__ == "TritonMissing" or "TritonMissing" in repr(exc)
        return {
            "status": "UNSUPPORTED" if unsupported else "FAILED",
            "evidence": "single compiled wrapper forward returns Q,V,A",
            "batch": batch,
            "precision": "bf16",
            "execution": "compile",
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=10),
            "extra_triton_installed": False,
        }


def main() -> int:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    model = data["models"]["ResidualMLP2048"]
    existing_8192 = next(
        p for p in model["inference_eager"]
        if p.get("status") == "PASS"
        and p.get("batch") == 8192
        and p.get("precision") == "bf16"
        and p.get("execution") == "eager"
    )

    p16384 = inference_point(
        ResidualMLP2048, 16384, "bf16", "eager"
    )
    extension = {
        "baseline_8192": existing_8192,
        "batch_16384": p16384,
        "batch_32768": None,
        "rule": "measure 32768 only if 16384 vs 8192 gain >= 5% and resources permit",
    }

    gain = None
    if p16384.get("status") == "PASS":
        gain = (
            p16384["states_per_s"] / existing_8192["states_per_s"] - 1.0
        ) * 100.0
    extension["gain_16384_vs_8192_percent"] = gain

    total_vram = torch.cuda.get_device_properties(0).total_memory
    resource_ok = (
        p16384.get("status") == "PASS"
        and p16384.get("peak_reserved", total_vram) < 0.70 * total_vram
    )
    extension["resource_ok_for_32768"] = bool(resource_ok)

    if gain is not None and gain >= 5.0 and resource_ok:
        extension["batch_32768"] = inference_point(
            ResidualMLP2048, 32768, "bf16", "eager"
        )

    new_points = [p16384]
    if extension["batch_32768"] is not None:
        new_points.append(extension["batch_32768"])
    existing_keys = {
        (p.get("batch"), p.get("precision"), p.get("execution"))
        for p in model["inference_eager"]
    }
    for point in new_points:
        key = (point.get("batch"), point.get("precision"), point.get("execution"))
        if key not in existing_keys:
            model["inference_eager"].append(point)

    valid_eager = [
        p for p in model["inference_eager"]
        if p.get("status") == "PASS" and "states_per_s" in p
    ]
    model["best_inference_eager"] = max(
        valid_eager, key=lambda p: p["states_per_s"]
    )

    # compile inference remains unsupported on this Windows setup; eager remains
    # the official inference execution unless an existing valid compiled point wins.
    candidates = [model["best_inference_eager"]]
    for item in model.get("compile_comparisons", []):
        point = item.get("inference", {})
        if point.get("status") == "PASS" and "states_per_s" in point:
            candidates.append(point)
    model["selected_inference"] = max(
        candidates, key=lambda p: p["states_per_s"]
    )

    corrected = [
        corrected_compile_training_point(1024),
        corrected_compile_training_point(8192),
    ]
    model["p10_gpu_only_extension"] = extension
    model["legacy_training_compile_evidence_valid"] = False
    model["legacy_training_compile_reason"] = (
        "legacy benchmark called forward_state/forward_afterstate on the compiled "
        "wrapper object; those bound methods remained original-model methods and "
        "do not establish compiled training evidence"
    )
    model["corrected_training_compile"] = corrected

    write_json(OUT, data)
    print(json.dumps({
        "p10_gpu_only_extension": extension,
        "selected_inference": model["selected_inference"],
        "corrected_training_compile": corrected,
        "legacy_training_compile_evidence_valid": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
