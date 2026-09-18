# M2 Formal Network Implementation Report

## 1. RESULT

**M2 INCOMPLETE**

Formal M2 cannot form a candidate commit because `ResidualMLP2048` still triggers the operational GPU starvation gate after the allowed safe pipeline tuning. Per the M2 work order, execution stops at M2: no C++ rewrite, no M2 tag, no master-plan PASS update, and no M3.

Secondary validation limitations are also recorded below: the existing completed GPU artifact has no 16384 MLP GPU-only point even though 8192 still improved throughput materially, and its training `torch.compile` rows do not prove that `forward_state` / `forward_afterstate` were compiled.

## 2. STARTING STATE

- Repository: `D:\CodexTasks\NN-2048-ai`
- Initial/current HEAD and main: `ea3c2167fb3d8b7cb0a205007ea0b7ca49d5afc3`
- Formal-plan expected base `827f30c8ccd635420c8ef1c2dfea00a9e54197b2` is an ancestor of this HEAD.
- The committed delta from that base is prompt synchronization only: `prompts/M2_IMPLEMENTATION_PROMPT.md` plus `prompts/M2_PRE_IMPLEMENTATION_PROMPT.md`.
- M0 tag: `m0-reference-pass` -> `3f2def1d95f56eff776e671143188947bf64485b`
- M1 tag: `m1-fastenv-audited-pass` -> `e5486017a90eeec4fb9814de7880b3c413dc06dd`
- M2 Preflight audited implementation: `356cedd443d6f56ac513f97e723e9addb78a28a9`
- Authoritative master plan was read and kept read-only.

## 3. ENVIRONMENT

- OS: Microsoft Windows 11, build 10.0.26200
- Python used for Torch/GPU runs: 3.11.9
- CPU: AMD Ryzen 7 H 260, 8 cores / 16 logical processors
- RAM: 16,422,010,880 bytes (~15.3 GiB)
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB
- Driver: 596.13
- PyTorch: 2.9.1+cu130
- CUDA runtime: 13.0
- BF16 support: True

## 4. FILES CHANGED / CREATED

- `.github/workflows/ci.yml` — CPU PyTorch dependency/pin for expanded full pytest.
- `src/game2048/m2_models.py` — Transformer2048, ResidualMLP2048, strict input/tokenization, independent Q/V/A heads.
- `src/game2048/m2_symmetry.py` — batched Torch D4 mappings derived from the frozen M0 oracle.
- `src/game2048/m2_policy.py` — legal masking and random tie-safe greedy action selection.
- `tests/test_m2_models.py` — model/input/gradient/parameter tests.
- `tests/test_m2_symmetry.py` — D4 differential tests.
- `tests/test_m2_policy.py` — mask/tie/action-selection tests.
- `benchmarks/_m2_utils.py` — benchmark timing/GPU sampling helper.
- `benchmarks/validate_m2_models.py` — GPU forward/backward and tiny-overfit gate.
- `benchmarks/benchmark_m2_gpu.py` — precision/batch/eager/compile GPU benchmark.
- `benchmarks/benchmark_m2_closed_loop.py` — M1+M2 closed-loop, profile, starvation gate.
- `benchmarks/benchmark_m2_pipeline_tuning.py` — evidence-driven safe-pipeline A/B after starvation.
- `m2_correctness.json` — correctness artifact.
- `m2_gpu_benchmark.json` — completed GPU benchmark artifact; not rerun in this continuation.
- `m2_closed_loop_benchmark.json` — closed-loop/profile and tuning evidence.
- `M2_REPORT.md` — this report.

No frozen M0/M1 source, authoritative master plan, Teacher, tuple checkpoint, Replay, Self-play, C++, pybind11, or custom CUDA environment was modified or used.

## 5. MODELS

### Transformer2048

- Vocabulary 22; 16 tokens; hidden 256; 6 TransformerEncoder blocks; 8 attention heads; FFN 1024; GELU; dropout 0; norm-first; mean pooling.
- Learned absolute position embedding shape `(1,16,256)`, trainable.
- Independent Q `256->4`, V `256->1`, A `256->1` heads.
- Trainable parameters: 4,750,342.

### ResidualMLP2048

- Vocabulary 22; embedding dim 32; fixed-order 16-cell flatten; hidden 640; 6 residual MLP blocks; GELU/LayerNorm.
- Independent Q/V/A heads.
- Trainable parameters: 5,264,710.
- Parameter-count gap vs Transformer: ~10.8%, within the <=15% gate.

## 6. INPUT CONTRACT

- NumPy transfer boundary: `(N,16)`, `uint8`, `N>=1`, C-contiguous via `prepare_board_batch_for_transfer`.
- Torch models accept only `torch.Tensor`, shape `(B,16)`, `B>=1`, dtype `torch.uint8`.
- NN vocabulary mapping only: exponent `0..20` unchanged; `21..255 -> token 21`.
- Environment boards are not clamped to 21.

## 7. CORRECTNESS

- Frozen M0/M1/M2-Preflight regression before new Torch tests: 465/465 passed.
- New M2 CPU tests: 22/22 passed.
- CPU forward/backward, Q/V/A shapes, D4 differential, legal mask, tie handling, and Transformer position-embedding gradient: PASS.
- GPU Transformer tiny-overfit: PASS, 40 steps, total loss 2.740334 -> 0.517512, finite forward/backward.
- GPU Residual MLP tiny-overfit: PASS, 37 steps, total loss 2.854209 -> 0.550019, finite forward/backward.
- Synthetic tiny-overfit targets are correctness/optimization checks only; they are not chess-strength targets.

## 8. PYTEST

Final local run using the Torch environment:

- Collected: 487
- Passed: 487
- Failed: 0
- Skipped: 0
- Xfailed: 0
- Warnings: 6 identical PyTorch nested-tensor warnings caused by `norm_first=True`; no failure.

## 9. GPU PRECISION

BF16 is natively supported on the RTX 5060 and was tested together with FP32. FP16 + GradScaler was not required by the work order because BF16 support is available.

## 10. BATCH SWEEP — TRANSFORMER INFERENCE

| Precision | Batch | Status | states/s | GPU mean % | reserved MiB |
|---|---:|---|---:|---:|---:|
| FP32 | 256 | PASS | 29,861.01 | 68.33 | 92 |
| FP32 | 512 | PASS | 32,369.93 | 94.53 | 170 |
| FP32 | 1024 | PASS | 31,813.09 | 97.88 | 302 |
| FP32 | 2048 | PASS | 26,303.17 | 98.72 | 540 |
| FP32 | 4096 | PASS | 28,127.15 | 99.18 | 1022 |
| FP32 | 8192 | PASS | 29,904.82 | 99.18 | 1982 |
| BF16 | 256 | PASS | 49,857.01 | 58.67 | 90 |
| BF16 | 512 | PASS | 63,485.88 | 85.71 | 134 |
| BF16 | 1024 | PASS | 64,168.50 | 88.87 | 216 |
| BF16 | 2048 | PASS | 54,901.15 | 96.26 | 374 |
| BF16 | 4096 | PASS | 58,358.75 | 98.83 | 678 |
| BF16 | 8192 | PASS | 59,487.71 | 97.40 | 1286 |

### Transformer training

| Precision | Batch | Status | samples/s |
|---|---:|---|---:|
| FP32 | 256 | PASS | 4,612.89 |
| FP32 | 512 | PASS | 4,597.17 |
| FP32 | 1024 | PASS | 4,383.27 |
| FP32 | 2048 | PASS | 1,091.53 |
| FP32 | 4096 | PASS | 94.35 |
| FP32 | 8192 | OOM | - |
| BF16 | 256 | PASS | 6,772.67 |
| BF16 | 512 | PASS | 10,639.67 |
| BF16 | 1024 | PASS | 9,672.39 |
| BF16 | 2048 | PASS | 9,785.51 |
| BF16 | 4096 | PASS | 2,078.94 |
| BF16 | 8192 | OOM | - |

Transformer batch-8192 training OOMs were explicitly recorded, not skipped.

## 10. BATCH SWEEP — RESIDUAL MLP INFERENCE

| Precision | Batch | Status | states/s | GPU mean % | reserved MiB |
|---|---:|---|---:|---:|---:|
| FP32 | 256 | PASS | 108,555.16 | 52.46 | 44 |
| FP32 | 512 | PASS | 191,061.07 | 71.77 | 44 |
| FP32 | 1024 | PASS | 462,561.28 | 77.77 | 62 |
| FP32 | 2048 | PASS | 568,151.69 | 82.00 | 62 |
| FP32 | 4096 | PASS | 554,829.71 | 92.29 | 72 |
| FP32 | 8192 | PASS | 564,103.69 | 96.87 | 120 |
| BF16 | 256 | PASS | 83,679.44 | 19.62 | 56 |
| BF16 | 512 | PASS | 151,429.61 | 20.15 | 58 |
| BF16 | 1024 | PASS | 348,747.28 | 23.23 | 74 |
| BF16 | 2048 | PASS | 685,690.53 | 39.77 | 76 |
| BF16 | 4096 | PASS | 1,355,184.19 | 74.08 | 96 |
| BF16 | 8192 | PASS | 1,509,056.29 | 88.08 | 142 |

### Residual MLP training

| Precision | Batch | Status | samples/s |
|---|---:|---|---:|
| FP32 | 256 | PASS | 14,442.70 |
| FP32 | 512 | PASS | 30,450.69 |
| FP32 | 1024 | PASS | 62,456.57 |
| FP32 | 2048 | PASS | 92,125.19 |
| FP32 | 4096 | PASS | 103,121.83 |
| FP32 | 8192 | PASS | 104,657.91 |
| BF16 | 256 | PASS | 11,150.80 |
| BF16 | 512 | PASS | 25,488.85 |
| BF16 | 1024 | PASS | 50,833.25 |
| BF16 | 2048 | PASS | 101,932.85 |
| BF16 | 4096 | PASS | 208,100.53 |
| BF16 | 8192 | PASS | 232,290.78 |

The JSON artifact contains exact latency/step-time, utilization min/max/sample count, allocated VRAM, and reserved VRAM for every point.

## 11. EAGER VS COMPILE

- Transformer batch 1024 BF16 compile inference: FAILED.
- Residual MLP batch 1024 and 8192 BF16 compile inference: FAILED.
- Exact failure: `TritonMissing: Cannot find a working triton installation`.
- Per the work order, no third-party compiler stack was installed as a workaround; eager remains allowed.
- Important validation finding: raw training compile rows are **not accepted as compiled-training evidence**. Introspection showed `torch.compile(model).forward_state.__self__ is original_model` is True, so the custom Q/V/A training methods bypass the OptimizedModule compiled `forward` path.

## 12. SELECTED M2 SYSTEMS CONFIG FROM EXISTING GPU ARTIFACT

- Transformer inference: BF16 / eager / batch 1024 = 64,168.50 states/s.
- Transformer best eager training: BF16 / batch 512 = 10,639.67 samples/s.
- Residual MLP inference: BF16 / eager / batch 8192 = 1,509,056.29 states/s.
- Residual MLP best eager training: BF16 / batch 8192 = 232,290.78 samples/s.

Validation limitation: MLP BF16 inference rose from 1,355,184.19 states/s at 4096 to 1,509,056.29 at 8192 (>10%) while reserved VRAM remained low. The existing completed GPU artifact has no 16384 GPU-only point. The continuation explicitly prohibited rerunning the completed GPU benchmark, so this gap is documented rather than rerun.

## 13. CLOSED LOOP

### Transformer2048

| envs | decisions/s | GPU mean % | CPU cores | reserved MiB |
|---:|---:|---:|---:|---:|
| 1024 | 33,373.63 | 55.84 | 1.09 | 196 |
| 4096 | 35,719.40 | 63.40 | 1.06 | 658 |
| 8192 | 31,464.12 | 63.98 | 1.04 | 1266 |
| 16384 | 32,249.45 | 49.47 | 1.06 | 2482 |

### ResidualMLP2048 baseline

| envs | decisions/s | GPU mean % | CPU cores | reserved MiB |
|---:|---:|---:|---:|---:|
| 1024 | 59,252.28 | 20.00 | 1.05 | 54 |
| 4096 | 91,638.80 | 10.04 | 1.07 | 86 |
| 8192 | 70,059.08 | 8.35 | 1.06 | 122 |
| 16384 | 73,522.62 | 10.05 | 1.07 | 224 |

## 14. WALL-CLOCK PROFILE

### Transformer, best closed-loop 4096 envs

- Environment stepping: 18.96%
- Legal mask: 13.16%
- CPU batch preparation: 0.01%
- H2D: 0.31%
- NN inference: 66.65%
- Action selection: 0.79%
- D2H: 0.11%
- reset_where: 0.01%
- Other: 0.00%
- CPU/env/legal/prep/H2D feed share: 32.44%
- Pipeline stall: 0.43%

### Residual MLP baseline, best closed-loop 4096 envs

- Environment stepping: 51.84%
- Legal mask: 36.11%
- CPU batch preparation: 0.04%
- H2D: 1.99%
- NN inference: 7.76%
- Action selection: 1.91%
- D2H: 0.31%
- reset_where: 0.04%
- Other: 0.00%
- CPU/env/legal/prep/H2D feed share: 89.98%
- Pipeline stall: 2.34%

## 15. GPU STARVATION GATE

### Transformer2048 — PASS

- Model-only inference: 64,168.50 states/s.
- Best closed-loop: 35,719.40 decisions/s.
- Closed-loop/model-only ratio: 55.67%.
- GPU mean: 63.40%.
- CPU/H2D feed share: 32.44%.
- Pipeline stall: 0.43%.
- Operational tests: A=False, B=False, C=False.

### ResidualMLP2048 baseline — FAIL

- Model-only inference: 1,509,056.29 states/s.
- Best closed-loop: 91,638.80 decisions/s.
- Closed-loop/model-only ratio: 6.07%.
- GPU mean: 10.04%.
- CPU/H2D feed share: 89.98%.
- Pipeline stall: 2.34%.
- Operational tests: A=True, B=False, C=True.

### ResidualMLP2048 after permitted safe tuning — STILL FAIL

- Best tuned closed-loop: 101,193.45 decisions/s.
- Improvement vs baseline: +10.43%.
- Closed-loop/model-only ratio: 6.71%.
- GPU mean: 11.25%.
- CPU/H2D feed share: 87.15%.
- Pipeline stall: 1.23%.
- Operational tests: A=True, B=False, C=True.
- Main bottleneck remains CPU environment stepping + legal-mask production, not H2D.

Because safe tuning did not clear starvation, the work order requires `M2 RESULT = INCOMPLETE` and STOP. C++ or custom CUDA is not authorized in this M2 task.

## 16. PERFORMANCE OPTIMIZATIONS

Evidence before: MLP baseline 91,638.80 decisions/s, ~10.04% mean GPU, 89.98% CPU/H2D feed share.

Allowed safe change tested without modifying frozen M1 source:

- worker-sharded independent `Fast2048BatchEnv` instances;
- reusable pinned CPU board/legal/action staging buffers;
- reusable GPU buffers;
- non-blocking H2D/D2H copies.

A/B results at 4096 total envs:

- 2 workers: 101,193.45 decisions/s, CPU 1.66 cores, GPU mean 11.25%.
- 4 workers: 72,792.25 decisions/s, CPU 2.01 cores, GPU mean 9.06%.
- 8 workers: 42,500.90 decisions/s, CPU 1.72 cores, GPU mean 7.00%.

2 workers was the only improvement (+10.43%); 4/8 workers regressed. Starvation remained, so no stronger rewrite was attempted.

## 17. CI

- `.github/workflows/ci.yml` now installs CPU `torch==2.9.1` and runs full pytest.
- Candidate commit: NOT CREATED because M2 Exit Criteria are not met.
- Candidate GitHub Actions run: NOT RUN because no incomplete candidate was pushed.
- Local CI-equivalent full pytest: 487 collected / 487 passed / 0 failed / 0 skipped / 0 xfailed.

## 18. FROZEN VERIFICATION

- M0 tag unchanged: YES.
- M1 tag unchanged: YES.
- Frozen tests deleted: NONE.
- Skip added: NONE.
- Xfail added: NONE.
- Frozen M1 source modified: NONE.

## 19. OUT OF SCOPE

- Teacher: NO
- tuple checkpoint: NO
- Expectimax: NO
- Replay: NO
- Self-play learner: NO
- Double-Q: NO
- Target network: NO
- C++: NO
- pybind11: NO
- custom CUDA env: NO
- M3 started: NO

## 20. ARTIFACTS

- `M2_REPORT.md`
- `m2_correctness.json`
- `m2_gpu_benchmark.json`
- `m2_closed_loop_benchmark.json`

## 21. FINAL GIT STATE

- HEAD remains `ea3c2167fb3d8b7cb0a205007ea0b7ca49d5afc3`.
- M2 source/tests/benchmark scripts/results/report remain intentionally uncommitted because the Exit Criteria are not met.
- No M2 tag was created.
- No M2 candidate push was made.

## 22. REMAINING BLOCKERS

1. **Primary blocker:** ResidualMLP2048 still has obvious GPU starvation after permitted safe tuning; gates A and C remain true.
2. Existing GPU-only artifact omitted a 16384 MLP point even though 8192 still materially improved BF16 inference with ample VRAM. It was not rerun because the user explicitly prohibited rerunning the completed GPU benchmark.
3. Training `torch.compile` rows in the raw artifact do not validate compiled Q/V/A training methods; inference compile is explicitly FAILED due missing Triton. Eager is the only validated execution mode.
4. No candidate commit / remote CI run exists because blocker 1 requires `M2 INCOMPLETE + STOP` before candidate closeout.

**STOP: do not update the master plan to M2 PASS, do not create an M2 tag, and do not enter M3.**
