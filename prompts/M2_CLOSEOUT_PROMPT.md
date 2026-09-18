# M2 Closeout — 严格施工提示词

> 项目：Safolour/NN-2048-ai
> 正式工作区：D:\CodexTasks\NN-2048-ai
> 本任务只做 M2 独立审计后的 closeout / 状态固化。
> 不做任何 M3 实现，不读取 tuple checkpoint，不修改 M2 算法或性能实现。

## 0. 权威依据

必须同时读取并遵守：

1. prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md
2. prompts/M2_IMPLEMENTATION_PROMPT.md
3. prompts/M2_IMPLEMENTATION_PROMPT_FIX_1.md

权威层级：
- 唯一权威总计划最高
- 两份 M2 文件是详细施工规范
- 若真实冲突：STOP，不自行裁决

## 1. 已完成独立审计的最终 M2 实现

最终 audited implementation commit 固定为：

```text
6a9da5b1bf7c72207acd6e89bc667d439d4823a8
```

其父 provisional candidate：

```text
52c40dad5908fd4dce46bb15ebde581f1bc6a8d5
```

不得把 provisional candidate 当最终 audited commit。

独立审计已确认：
- HEAD/origin-main = 6a9da5b1...
- M0 tag -> 3f2def1d95f56eff776e671143188947bf64485b
- M1 tag -> e5486017a90eeec4fb9814de7880b3c413dc06dd
- GitHub Actions run #9 id 35374650846 = success
- Ubuntu C++ backend build success
- full pytest = 505 collected / 505 passed / 0 failed / 0 skipped / 0 xfailed
- production backend = scalar+row-lut
- simd_used = False
- lut_used = True
- selected closed-loop scale = 32768
- selected throughput = 704,818.03 decisions/s
- starvation A=False / B=False / C=False
- 65536 not run because 32768 reached 96% host RAM load and ~648 MB min free physical RAM
- M2 Exit Criteria 已全部满足

## 2. 本任务唯一目标

把“已经独立审计通过”的事实正式固化为：
- authoritative M2 tag
- 唯一权威 master plan 的 M2 FINAL AUDITED PASS / FROZEN 状态
- 当前执行阶段切换为 M3
- docs closeout commit
- remote CI green

不得做 M3 实现。

## 3. 开工检查

必须先执行：

```powershell
Set-Location 'D:\CodexTasks\NN-2048-ai'
git status --short
git rev-parse HEAD
git rev-parse origin/main
git rev-parse 'm0-reference-pass^{commit}'
git rev-parse 'm1-fastenv-audited-pass^{commit}'
git tag --list 'm2*'
```

要求：
- HEAD = origin/main = 6a9da5b1bf7c72207acd6e89bc667d439d4823a8
- M0/M1 tag 精确匹配上述 frozen commit
- 当前不存在任何 M2 tag
- 用户 prompt 文件若未跟踪，保持未跟踪，不得 stage/commit/restore/delete/rename

若 HEAD/main 不匹配：STOP。

## 4. M2 authoritative tag

唯一正式 tag 名固定为：

```text
m2-network-audited-pass
```

tag 必须精确指向：

```text
6a9da5b1bf7c72207acd6e89bc667d439d4823a8
```

建议 annotated tag message：

```text
M2 network and performance audited pass
```

创建后立即验证：

```powershell
git rev-parse 'm2-network-audited-pass^{commit}'
```

必须等于 6a9da5b1...

绝对禁止移动/重建：
- m0-reference-pass
- m1-fastenv-audited-pass

## 5. 唯一允许修改的 tracked 文件

本 closeout 默认只允许修改：

```text
prompts/2048_AI_正式执行计划_M0_M1冻结_CI增强版_2026-09-18.md
```

不得修改：
- src/
- cpp/
- tests/
- benchmarks/
- M2_REPORT.md
- benchmark JSON
- .github/workflows/ci.yml
- docs/M2_FAST_BACKEND_BUILD.md
- M0/M1 frozen 文件

## 6. 主计划必须更新的状态

### 6.1 顶部状态块

把当前状态更新为：
- M0 completed / frozen
- M1 completed / audited / frozen
- M2 Preflight audited pass
- M2 Formal Network + Performance = FINAL AUDITED PASS / FROZEN
- M2 authoritative tag = m2-network-audited-pass -> 6a9da5b1...
- M2 audited implementation commit = 6a9da5b1...
- final M2 CI = run 35374650846 PASS
- final M2 pytest = 505 passed / 0 failed / 0 skipped / 0 xfailed
- current allowed stage = M3 Teacher 小规模验证

必须明确：
M2 audited implementation commit 与随后 docs closeout commit 是两个不同 commit，不得混同。

### 6.2 第 99 节 M2

在 #99 M2 开头增加 final closeout 状态，至少记录：
- M2 FINAL AUDITED PASS / FROZEN
- authoritative tag / audited commit
- Transformer + ResidualMLP + Q/V/A 已完成
- production backend scalar+row-lut
- SIMD=False / LUT=True
- selected scale 32768
- 704,818.03 decisions/s
- A=False / B=False / C=False
- RAM caveat：32768 时 96% host RAM load，因此未运行 65536
- local / CI full pytest 505/505
- GitHub Actions run #9 / 35374650846 PASS
- M0/M1 tags unchanged
- tuple checkpoint M2 未使用

### 6.3 修正过时状态文字

不得留下任何仍声称：
- “M2 formal network implementation = CURRENT EXECUTION STAGE”
- “M2 尚未完成”
- “M2 Exit Criteria 其余条目目前一条都没有完成”
- “当前允许执行 M2 正式网络实现”

这些是 Preflight closeout 时的历史状态，必须改成明确的历史说明或最终状态。

PF.1～PF.6 等历史正式规格继续保留，不删除。

### 6.4 M2 Exit Criteria

保留 criteria 本身。

状态说明改为：
- 全部 Exit Criteria 已由 audited commit 6a9da5b1... 满足
- mandatory CI green
- M2 正式通过
- 当前允许进入 M3

不得删除 criteria 或弱化标准。

## 7. 主计划禁止改动

除状态 closeout 文字外，不得修改：
- Q/V/A 语义
- 网络结构
- Teacher 语义
- Search 定义
- Replay/Double-Q
- M3～M12 技术路线
- 训练预算规则
- M0/M1 frozen semantics

## 8. 本地复验

修改 master plan 后：
- git diff 必须只显示 master plan tracked diff
- prompt 文件仍保持用户未提交状态
- python -m pytest 必须完整通过
- 不得 skip / xfail

## 9. Closeout commit

tracked closeout commit message 固定：

```text
docs: close M2 and advance plan to M3
```

该 commit 只包含唯一权威 master plan 状态更新。

不要把：
- M2_IMPLEMENTATION_PROMPT_FIX_1.md
- M2_CLOSEOUT_PROMPT.md
等用户 prompt 文件夹带进 commit。

## 10. Push / tag / CI

push main 与 authoritative tag。

最终要求：
- origin/main = closeout docs commit
- m2-network-audited-pass^{commit} = 6a9da5b1...
- M0/M1 tags unchanged

等待 closeout commit 对应的实际 GitHub Actions。

必须读取真实日志并记录：
- run id
- conclusion
- Build M2 C++ fast backend = success
- Run full pytest suite = success
- collected N
- N passed
- 0 failed / 0 skipped / 0 xfailed

CI 失败则：
- M2 implementation audit 事实不撤销
- 但 closeout 未完成
- STOP，不开始 M3

## 11. 本任务禁止事项

绝对禁止：
- 开始 M3 代码
- 读取/加载/转换 tuple 8x6 checkpoint
- Teacher
- Expectimax
- Replay
- Self-play
- M4+
- 修改 M2 backend
- 新一轮性能优化
- SIMD
- CUDA FastEnv
- 移动 M0/M1 tags

## 12. 最终报告格式

必须报告：

1. RESULT
   - M2 CLOSEOUT COMPLETE 或 M2 CLOSEOUT INCOMPLETE

2. STARTING STATE
   - audited M2 implementation commit
   - initial HEAD/main
   - M0/M1 tags

3. AUTHORITATIVE M2 TAG
   - tag name
   - resolved commit

4. MASTER PLAN UPDATE
   - exact status now
   - current stage now

5. FILES CHANGED
   - tracked files
   - user-owned prompts excluded

6. PYTEST
   - collected/passed/failed/skipped/xfailed

7. CLOSEOUT COMMIT
   - SHA
   - message

8. GITHUB ACTIONS
   - run id
   - actual steps
   - collected/passed

9. FINAL GIT STATE
   - HEAD
   - origin/main
   - git status --short
   - M0/M1/M2 tag resolutions

10. OUT OF SCOPE
   - M3 implementation: NO
   - tuple checkpoint used: NO
   - Teacher: NO

11. NEXT STAGE
   - M3 Teacher 小规模验证 is now eligible
   - but not started

完成后 STOP。
