from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _m2_utils import write_json
from game2048.m2_models import (
    ResidualMLP2048,
    Transformer2048,
    trainable_parameter_count,
)

SEED = 20260918
DEVICE = torch.device("cuda")
OUTPUT = ROOT / "reports/m2/m2_correctness.json"


def make_boards(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    states = torch.randint(0, 18, (128, 16), dtype=torch.uint8, generator=g)
    after = torch.randint(0, 18, (128, 16), dtype=torch.uint8, generator=g)
    # Explicit overflow-vocabulary coverage.
    states[:8, :4] = torch.randint(18, 256, (8, 4), dtype=torch.uint8, generator=g)
    after[:8, :4] = torch.randint(18, 256, (8, 4), dtype=torch.uint8, generator=g)
    return states.to(DEVICE), after.to(DEVICE)


def make_targets(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn((128, 4), device=DEVICE, generator=g)
    v = torch.randn((128,), device=DEVICE, generator=g)
    a = torch.randn((128,), device=DEVICE, generator=g)
    return q, v, a


def losses(model, states, afterstates, targets):
    q_target, v_target, a_target = targets
    q, v = model.forward_state(states)
    a = model.forward_afterstate(afterstates)
    q_loss = F.mse_loss(q, q_target)
    v_loss = F.mse_loss(v, v_target)
    a_loss = F.mse_loss(a, a_target)
    total = q_loss + 0.5 * v_loss + 0.5 * a_loss
    return total, q_loss, v_loss, a_loss


def validate_one(name: str, model_cls) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = model_cls().to(DEVICE)
    params = trainable_parameter_count(model)
    states, afterstates = make_boards(SEED + 1)
    targets = make_targets(SEED + 2)
    torch.cuda.reset_peak_memory_stats()

    model.train()
    initial = [item.detach().item() for item in losses(model, states, afterstates, targets)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = 0
    finite = True
    final = initial
    for step in range(1, 1001):
        optimizer.zero_grad(set_to_none=True)
        total, q_loss, v_loss, a_loss = losses(model, states, afterstates, targets)
        if not torch.isfinite(total):
            finite = False
            break
        total.backward()
        gradients_finite = all(
            p.grad is None or torch.isfinite(p.grad).all().item()
            for p in model.parameters()
        )
        if not gradients_finite:
            finite = False
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps = step
        final = [
            total.detach().item(),
            q_loss.detach().item(),
            v_loss.detach().item(),
            a_loss.detach().item(),
        ]
        if (
            final[0] <= initial[0] * 0.20
            and final[1] < initial[1]
            and final[2] < initial[2]
            and final[3] < initial[3]
        ):
            break

    # Separate real forward/backward smoke on the actual baseline.
    optimizer.zero_grad(set_to_none=True)
    total, *_ = losses(model, states, afterstates, targets)
    total.backward()
    backward_ok = all(
        p.grad is None or torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )
    torch.cuda.synchronize()
    passed = bool(
        finite
        and backward_ok
        and final[0] <= initial[0] * 0.20
        and final[1] < initial[1]
        and final[2] < initial[2]
        and final[3] < initial[3]
    )
    return {
        "model": name,
        "parameter_count": params,
        "initial_total_loss": initial[0],
        "final_total_loss": final[0],
        "q_loss_before": initial[1],
        "q_loss_after": final[1],
        "v_loss_before": initial[2],
        "v_loss_after": final[2],
        "a_loss_before": initial[3],
        "a_loss_after": final[3],
        "steps": steps,
        "peak_vram_allocated": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved": torch.cuda.max_memory_reserved(),
        "finite": finite,
        "forward_backward_ok": bool(backward_ok),
        "pass": passed,
    }


def main() -> int:
    if not torch.cuda.is_available():
        payload = {"pass": False, "reason": "CUDA unavailable"}
        write_json(OUTPUT, payload)
        print(payload)
        return 2
    device_name = torch.cuda.get_device_name(0)
    if "RTX 5060" not in device_name:
        payload = {"pass": False, "reason": f"unexpected GPU: {device_name}"}
        write_json(OUTPUT, payload)
        print(payload)
        return 2

    results = [
        validate_one("Transformer2048", Transformer2048),
        validate_one("ResidualMLP2048", ResidualMLP2048),
    ]
    payload = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": device_name,
        "synthetic_targets_have_chess_strength_semantics": False,
        "models": results,
        "pass": all(result["pass"] for result in results),
    }
    write_json(OUTPUT, payload)
    print(payload)
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
