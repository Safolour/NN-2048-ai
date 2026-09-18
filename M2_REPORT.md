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

Although smaller batches regressed, the primary metric is the best ResidualMLP closed-loop decisions/s. The row-LUT path raises the best closed-loop result from 591,443.46 to 638,949.53 decisions/s (+8.03%), exceeding the 5% retention threshold.

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

ResidualMLP final best at 16,384 envs:

- closed-loop: 638,949.53 decisions/s
- GPU mean: 56.94%
- CPU environment + batch/H2D feed: 34.31%
- closed-loop/model-only: 42.34%
- pipeline stall: 2.13%
- A=False
- B=False
- C=False

Therefore the ResidualMLP performance-unblock gate is PASS. Transformer remains PASS.

## 13. REMAINING BOTTLENECK

At the final 16,384-env LUT operating point:

- NN inference: 56.33% of measured decision wall time
- environment step: 32.13%
- reset_where: 4.69%
- action selection: 4.17%
- legal cache copy: 0.56%
- CPU batch preparation: 0.82%
- H2D: 0.80%
- D2H: 0.50%

The largest component is now NN inference, and CPU/H2D feed is below the 40% starvation threshold. No post-LUT profile is required by the gate logic and there is no formal evidence justifying SIMD. SIMD remains unimplemented.

## 14. MLP 16384 GPU-ONLY RESULT

ResidualMLP BF16 eager GPU-only:

- batch 8192: 1,509,056.29 states/s
- batch 16384: 1,171,351.16 states/s
- 16384 vs 8192: -22.38%
- 16384 status: PASS

Because the 16384 point did not improve by at least 5%, batch 32768 was not measured. The selected model-only inference point remains BF16 eager batch 8192.

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

Remote GitHub Actions status is intentionally not claimed in this pre-push report. The candidate is only complete after the pushed candidate commit receives an actual passing Actions run.

## 18. FINAL ARTIFACTS

- M2_REPORT.md
- m2_correctness.json
- m2_gpu_benchmark.json
- m2_closed_loop_benchmark.json
- m2_cpu_hotspots.json
- m2_fast_backend_benchmark.json
- m2_fast_backend_scalar_benchmark.json
- m2_post_cpp_hotspots.json
- m2_python_final_ab.json

m2_closed_loop_benchmark.json now preserves historical pre-unblock evidence while exposing the final LUT backend as the current ResidualMLP result.

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

Local candidate criteria are satisfied. The intended candidate commit message is:

m2: unblock pipeline performance and complete candidate

The commit must be pushed and actual GitHub Actions must pass before reporting M2 CANDIDATE COMPLETE. This report does not claim M2 AUDITED PASS and does not authorize M3.
