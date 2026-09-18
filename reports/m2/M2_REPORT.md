# M2 Formal Network Implementation Report

## 1. RESULT

**M2 PERFORMANCE UNBLOCK PASS — CANDIDATE READY, REMOTE CI PENDING**

The M2 pipeline performance blocker is cleared without changing frozen M0/M1 semantics. The final rollout backend is scalar C++ plus a 4-cell row LUT common path with exact scalar fallback for high exponents. SIMD was not implemented because the LUT path already cleared all starvation gates.

No M2 tag is created here. M2 remains a candidate until the pushed candidate commit passes actual GitHub Actions. M3 is not started.

## 2. STARTING STATE AND BASELINE CHECKPOINT

- Formal working directory: D:\CodexTasks\NN-2048-ai
- Performance-unblock starting HEAD: b645ab6c12b7a8295ae6b31ac7a489a4dcfb09bd
- Baseline checkpoint message: m2: checkpoint correctness-pass performance-blocked baseline
- M0 tag unchanged: 3f2def1d95f56eff776e671143188947bf64485b
- M1 tag unchanged: e5486017a90eeec4fb9814de7880b3c413dc06dd
- Authoritative master plan remained read-only.

## 3. PERFORMANCE UNBLOCK — FINE-GRAINED HOTSPOTS

P1 measured 4096/8192/16384 boards with at least 20 warmups and at least 100 measured iterations / 2 seconds per component.

At 4096 boards the largest CPU components were:

- post-spawn terminal: 17.600 ms
- current legal mask: 12.911 ms
- selected movement-only: 4.598 ms
- full Fast2048BatchEnv.step: 24.255 ms
- reward aggregation: 0.785 ms
- spawn placement: 1.096 ms
- output-copy allocation: about 0.011 ms, only about 0.028% of the complete CPU decision

cProfile independently identified _move_groups, legal_mask_batch, is_terminal_batch and row packing/merging as the dominant path.

## 4. REPEATED LEGAL / TERMINAL ANALYSIS

Measured invariant:

terminal(s') == ~next_legal(s').any(axis=1)

Steady-state movement work before reuse was 13 directional evaluations per decision:

- current legal: 4
- selected action: 1
- post-spawn terminal: 4
- next-iteration legal: 4

The post-spawn terminal set and next-iteration policy legal mask are the same next_legal result. Reuse removes 4/13 directional evaluations, or 30.77%.

## 5. TOOLCHAIN RESULT

PASS.

- MSVC compiler: 19.51.36248 x64
- Windows SDK: 10.0.26100.0
- CMake: available
- Python used for M2 GPU/local C++ testing: 3.11.9
- pybind11: 3.1.0
- RTX 5060 Laptop GPU / PyTorch 2.9.1+cu130 / CUDA 13.0

The required pybind11 smoke extension compiled, linked and imported successfully; add(2,3) returned 5 and the smoke emitted M2_CPP_SMOKE_OK.

## 6. PYTHON SAFE A/B

The cached-next-legal Python prototype passed a 256-env x 200-step trajectory differential check.

- historical tuned baseline: 101,193.45 decisions/s
- cached-next-legal: 112,912.16 decisions/s
- improvement vs historical tuned baseline: +11.58%
- same-run improvement: +4.93%
- starvation after Python-only optimization: A=True, B=False, C=True

Because the safe Python route improved less than 20% and did not clear starvation, the fixed P5 decision was C++ = YES.

## 7. C++ ARCHITECTURE

The formal M2 C++ path migrates only the measured hotspot primitives:

- selected 2048 movement
- all-actions legal mask / next-legal

NumPy PCG64 remains the sole spawn RNG. Spawn placement, scores and reset semantics remain Python/NumPy. Frozen M1 source is not modified.

The final backend metadata is:

- kind = scalar+row-lut
- simd_used = False
- lut_used = True

The LUT is only an internal common-path key for four cells whose exponents are all <=15. Official board representation stays uint8[16]. Any row containing exponent 16+ takes the exact generic scalar path; high-tile semantics are never clamped for environment movement.

## 8. DIFFERENTIAL CORRECTNESS

Final LUT differential result: 18/18 passed, 0 failed.

Coverage includes:

- 10,000 ordinary boards
- all four actions
- direct M0 oracle anchors
- required high-tile exponents including 15,16,17,20,21,31,61,62,63,126,127,128,254,255
- all 8 D4 transforms
- legal and terminal
- exact rewards
- seeded spawn sequence
- illegal-action RNG non-consumption
- reset and reset_where
- reward overflow
- score overflow
- tile exponent overflow
- atomicity
- live-view contract

No test was skipped or xfailed.

## 9. PRIMITIVE A/B — NUMPY VS SCALAR C++ VS ROW LUT

| Batch | NumPy legal/s | Scalar legal/s | LUT legal/s | NumPy move/s | Scalar move/s | LUT move/s | NumPy fused/s | Scalar fused/s | LUT fused/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 264,937 | 2,639,668 | 6,628,842 | 632,664 | 9,008,402 | 15,060,757 | 88,431 | 686,044 | 595,957 |
| 4096 | 259,079 | 2,472,299 | 6,409,648 | 870,892 | 9,296,524 | 17,641,859 | 111,706 | 999,773 | 1,072,110 |
| 8192 | 179,447 | 2,477,544 | 6,493,452 | 803,366 | 8,498,874 | 17,439,171 | 81,054 | 1,018,460 | 1,163,987 |
| 16384 | 178,941 | 2,245,072 | 6,331,926 | 753,151 | 8,752,172 | 18,385,119 | 76,094 | 1,074,385 | 1,325,743 |

Local kernel speed is diagnostic only; the retention decision is based on closed-loop throughput.

## 10. CLOSED-LOOP A/B

| Envs | Scalar C++ decisions/s | LUT decisions/s | Scalar->LUT |
|---:|---:|---:|---:|
| 1024 | 128,948.80 | 98,499.61 | -23.61% |
| 4096 | 405,833.15 | 364,287.81 | -10.24% |
| 8192 | 558,786.10 | 524,129.56 | -6.20% |
| 16384 | 591,443.46 | 638,949.53 | +8.03% |

Although smaller batches regressed, the primary metric is the best ResidualMLP closed-loop decisions/s. The row-LUT path raises the 16,384-env result from 591,443.46 to 638,949.53 decisions/s (+8.03%), exceeding the 5% retention threshold.

Post-candidate §27.3 adaptive-scale validation then measured the production LUT backend at 32,768 envs without rerunning old scalar baselines:

- 32,768 envs: 704,818.03 decisions/s
- gain vs 16,384 LUT: +10.31%
- run status: PASS, 100 measured iterations, 4.649 s
- 65,536 envs: NOT RUN because the 32,768 run hit the host-RAM pressure stop condition

## 11. STARVATION GATE BEFORE PERFORMANCE UNBLOCK

ResidualMLP after safe Python tuning:

- best closed-loop: 101,193.45 decisions/s
- GPU mean: 11.25%
- CPU/H2D feed: 87.15%
- closed-loop/model-only: 6.71%
- pipeline stall: 1.23%
- A=True, B=False, C=True

Transformer was already PASS with A=False, B=False, C=False.

## 12. STARVATION GATE AFTER FINAL LUT BACKEND

After the §27.3 adaptive-scale extension, the selected ResidualMLP systems configuration is 32,768 envs:

- closed-loop: 704,818.03 decisions/s
- GPU mean/min/max: 60.71% / 56.0% / 67.0%
- CPU utilization: 1.281 logical-core equivalents
- CPU environment + batch/H2D feed: 31.06%
- closed-loop/model-only: 46.71%
- pipeline stall: 1.21%
- A=False
- B=False
- C=False

The 32,768 run completed stably and is the highest measured end-to-end throughput. It also showed host-RAM pressure (maximum system memory load 96%, minimum available physical RAM 648,470,528 bytes), so §27.3 stopped expansion before 65,536. Therefore the ResidualMLP performance-unblock gate remains PASS, but no larger scale headroom is claimed on this measured 16 GiB host. Transformer remains PASS.

## 13. REMAINING BOTTLENECK

At the final selected 32,768-env LUT operating point:

- NN inference: 63.14% of measured decision wall time
- environment step: 29.83%
- reset_where: 3.09%
- action selection: 2.39%
- legal cache copy: 0.33%
- CPU batch preparation: 0.43%
- H2D: 0.48%
- D2H: 0.30%

The largest component is NN inference, while CPU/H2D feed is 31.06%, below the 40% starvation threshold. There is still no formal evidence justifying SIMD. The adaptive-scale stop is host RAM pressure, not movement/legal throughput.

## 14. MLP 16384 GPU-ONLY RESULT

ResidualMLP BF16 eager GPU-only:

- batch 8192: 1,509,056.29 states/s
- batch 16384: 1,171,351.16 states/s
- 16384 vs 8192: -22.38%
- 16384 status: PASS

Because the 16,384 GPU-only point did not improve by at least 5%, GPU-only batch 32,768 was not measured. This is separate from §27.3 closed-loop adaptive scaling, where 32,768 envs was measured. The selected model-only inference point remains BF16 eager batch 8192.

## 15. COMPILE-TRAINING VALIDATION CORRECTION

The old training compile rows are explicitly marked invalid as compile evidence because custom forward_state / forward_afterstate calls did not establish use of the compiled forward path.

A benchmark-only CompiledTrainingWrapper with one forward(states, afterstates) returning Q, V and A was then wrapped by torch.compile.

Corrected results:

- BF16 batch 1024: UNSUPPORTED — TritonMissing
- BF16 batch 8192: UNSUPPORTED — TritonMissing

No extra Triton stack was installed. Eager training remains the validated formal configuration.

## 16. FULL REGRESSION

Final local pytest after LUT and P10 changes:

- collected: 505
- passed: 505
- failed: 0
- skipped: 0
- xfailed: 0
- warnings: 6 PyTorch Transformer nested-tensor warnings

All original 487 tests remain; total test count is greater than 487.

## 17. CI / BUILD REPRODUCIBILITY

.github/workflows/ci.yml now builds the pybind11 C++ extension on ubuntu-latest before running the complete CPU suite. Backend tests are not skipped when the extension is absent; the CI must build it.

docs/M2_FAST_BACKEND_BUILD.md documents the MSVC requirement, dependencies, CMake configure/build, extension output and pytest command.

The provisional candidate commit 52c40dad5908fd4dce46bb15ebde581f1bc6a8d5 was pushed and GitHub Actions run #8 completed successfully, including the Ubuntu C++ backend build and full pytest. The supplemental adaptive-scale follow-up requires its own post-push Actions run before the supplemental validation is handed back to audit.

## 18. FINAL ARTIFACTS

- reports/m2/M2_REPORT.md
- reports/m2/m2_correctness.json
- reports/m2/m2_gpu_benchmark.json
- reports/m2/m2_closed_loop_benchmark.json
- m2_cpu_hotspots.json
- reports/m2/m2_fast_backend_benchmark.json
- m2_fast_backend_scalar_benchmark.json
- m2_post_cpp_hotspots.json
- m2_python_final_ab.json

reports/m2/m2_closed_loop_benchmark.json now preserves historical pre-unblock evidence while exposing the final LUT backend as the current ResidualMLP result.

## 19. FROZEN / OUT-OF-SCOPE VERIFICATION

- frozen M0 source modified: NO
- frozen M1 source modified: NO
- m0-reference-pass moved: NO
- m1-fastenv-audited-pass moved: NO
- M2 tag created: NO
- Teacher: NO
- tuple checkpoint: NO
- Replay: NO
- Self-play: NO
- custom CUDA FastEnv: NO
- M3 started: NO

## 20. CANDIDATE STATE

The provisional candidate commit is:

52c40dad5908fd4dce46bb15ebde581f1bc6a8d5
m2: unblock pipeline performance and complete candidate

Its GitHub Actions run #8 completed successfully. This post-candidate supplemental validation does not rewrite that history; it will be delivered as a normal follow-up commit.

This report does not claim M2 AUDITED PASS and does not authorize M3.

## 21. POST-CANDIDATE SUPPLEMENTAL VALIDATION — §27.3 CLOSED-LOOP ADAPTIVE SCALE

The latest implementation prompt added §27.3 after the provisional candidate. The existing production LUT closed-loop result at 16,384 envs was 638,949.53 decisions/s versus 524,129.56 at 8,192 envs, a gain of more than 5%, so 32,768 envs was mandatory.

The supplemental run reused the exact formal P8/P9 path:

- ResidualMLP2048
- BF16 eager inference
- scalar+row-lut backend
- 2-worker rollout runner
- existing pinned CPU / GPU staging path
- 20 warmup iterations
- at least 100 measured iterations and at least 2 seconds
- the same 30-iteration stage profile
- unchanged starvation gate definitions

### 32,768-env result

- env_count: 32,768
- decisions/s: 704,818.03
- relative gain vs 16,384: +10.31%
- iterations: 100
- measured elapsed: 4.649 s
- GPU mean/min/max: 60.71% / 56.0% / 67.0%
- CPU utilization: 1.281 logical-core equivalents
- NVIDIA memory-used max: 4,524 MiB
- Torch peak VRAM allocated: 255,961,600 bytes
- Torch peak VRAM reserved: 411,041,792 bytes
- total VRAM: 8,546,484,224 bytes
- peak reserved / total VRAM: 4.81%
- host RAM total: 16,422,010,880 bytes
- maximum host memory load during measurement: 96%
- minimum available physical RAM: 648,470,528 bytes
- minimum available pagefile: 1,277,259,776 bytes
- CPU/H2D feed: 31.06%
- pipeline stall: 1.21%
- closed-loop/model-only: 46.71%
- Gate A: False
- Gate B: False
- Gate C: False

### 65,536 decision

65,536 envs was not run. Although 32,768 improved throughput by more than 5% and completed without OOM, the host-RAM pressure stop condition fired: memory load reached 96% and available physical memory dropped below 1 GiB. VRAM was not the limiting resource.

### Final selected systems configuration

The highest stable measured end-to-end throughput is the 32,768-env point at 704,818.03 decisions/s. It is selected as the final measured systems configuration, with an explicit operational caveat that the measured 16 GiB host has no demonstrated headroom for further scale expansion.

Final P9 at the selected scale remains:

A=False
B=False
C=False

No SIMD work was performed. No M2 tag was created. The master plan was not marked frozen. M3 and tuple checkpoint work were not started.
