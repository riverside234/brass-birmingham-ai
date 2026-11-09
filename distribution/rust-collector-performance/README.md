# PI00–PI05 ordinary collector performance supplement

Current g1284 update: Stage C adds eight-game buffered loss retention. Machine transfer ships the fixed IL probe and pinned runtime/checkpoint state, omits replay/IL caches, and rebuilds fresh buffers while auxiliary counters pause. Use [the transfer guide](../../docs/STEP5_MACHINE_TRANSFER.md) and [revision note](../../ai-research/RUST_STAGE_C_TRANSFER_REVISION.md); no new ZIP or unrelated qualification rerun is needed.


## PPG / 辅助 value phase 性能优化

新优化保留 A10/B20/C10、固定 teacher beta、PPG beta floor .25 和 KL cap .003。
无 retained PPG update 时不再全量扫描 train rows；有更新时仍精确测量原始/最终
weights 的完整 calibration。value-only / policy-only forward 去掉未使用分支；
critic mmap 分块写入保持逐字节相同；辅助 target/input 减少复制。30 个受影响 debug
回归通过；GPU 实际加速仍待 lab 测量。新报告分别记录 cache、calibration、fit 和 head。

若当前仍为 g1280，可只做一次正常续训记录 timing：

```bash
python scripts/continue-rust-teacher.py --expected-generation 1280 --iterations 1 --share-target 0.25
```

若已推进则填写实际 generation；下方 A10 的十轮命令仍只执行剩余轮次。
无须 rebuild Rust、完整 bundle 或重跑已通过 qualification。详见
[分析与 timing 字段](../../ai-research/RUST_VALUE_PHASE_PERFORMANCE.md) 和
[机器凭据](teacher-value-performance.json)。

## 当前 g1280 辅助 head 修订（2026-11-08）

已拉取 `results/` 到 `ebe6084`。policy teacher 已运行至 g1280，固定 beta 保持
0.16110139609854138。辅助 head 的全部 value ages 在 PPO 后被旧 policy-staleness
规则过滤，导致一直没有更新。新路径直接使用通过独立校验的 return/value buffer，
不再重复保存 head 不需要的 policy-label history；policy teacher 自身的 staleness 保留。

按用户修订：**A 10 轮 head-only → B 20 轮无条件 blend warm-up (.05→1) →
C 10 轮无条件 trunk warm-up (.1→1)**。移除辅助 validation gates、validation rollback
和 256-game test；只保留硬 joint KL cap **.003**，仅撤销超 cap 的 interval。
PPG cloning beta floor **.25**，不重置高于 floor 的已保存 beta。
4-game held-out diagnostics 每 10 轮一次，其他轮不运行 held-out games。
`teacher_phase.search_value.schedule.validation_during_beta_qualification=true`
使未完成 beta qualification 的 shadow rounds 仍每轮运行 4 games。

同步代码、退出旧 trainer 后在 lab 运行：

```bash
python scripts/continue-rust-teacher.py --expected-generation 1280 --iterations 10 --share-target 0.25
```

从 g1280 开始新的 A10，g1290 进入 B；保留现有 head、完整 checkpoint/Adam/RNG/PPG/
population、value shards、search_v1 和原 collector 性能机制。无须重编译 Rust、完整 bundle
或重做已通过资格。首次 revised lab training 仍 Pending。详见
[审查、配置与监控](../../ai-research/RUST_TEACHER_STAGE_SCHEDULE_REVISION.md) 和
[机器凭据](teacher-stage-schedule.json)。下方为历史记录。

## 历史 teacher on / Stage A 续训（2026-11-08）

已在 `results/` 拉取用户给出的 `92dbef3` 和 `a966c75`。四轮完整 28+4 shadow
已通过，当前 committed checkpoint 为 **g1261**。184 个有效梯度比例的 pooled
median 为 1.551817712660179，target 0.25 对应固定 **beta=0.16110139609854138**。
按用户要求启用 teacher **on**，辅助 search head 从 **Stage A** 开始训练。

同步 Git 代码并退出旧 trainer 后，在 lab 原训练环境运行：

```bash
python scripts/continue-rust-teacher.py --expected-generation 1261 --iterations 1 --share-target 0.25
```

继续同一 run **g1261→g1262**；保留完整 checkpoint/Adam/RNG/PPG/population、
branching 0%、search_v1、28+4 tables 和原 collector 性能优化。policy teacher 使用固定
beta；辅助 head 每轮在 PPG 后做 **32 steps head-only Adam**，**w=0、trunk scale=0**，
后续阶段仍须通过原 gates。首次 head 安装推迟至该边界，避免 collection model ID
变化；annealing 按已提交 progress 判断，避免静态 on 标志永久阻止训练。

需要真实四轮资格、readiness、source/checkpoint/native bindings；结果自动导出到
`results/rust-lab-qualification/`。重复同一命令只完成剩余工作。两个 Rust adapter
均未改变，**无须 rebuild、完整 bundle 或重做通过的 qualification**。本地 38 个相关
debug 回归通过，无 benchmark、游戏或真实训练。首次 on-mode lab iteration 仍 Pending。
详见 [审查与监控说明](../../ai-research/RUST_TEACHER_ON_STAGE_A_REVIEW.md) 和
[机器凭据](teacher-on-stage-a.json)。下方为历史修复/测试记录。

## 历史续训修复（2026-11-08，已完成）

已在 `results/` 拉取到 `5e03429`，确认 g1258 的 teacher/ordinary adapter 准备均通过。
后续 `Uncommitted iteration ... iteration-1258` 来自真实 checkpoint loader 的第二道 guard。
修复只允许已审查的 2→4 梯度测量窗口扩展复用相同 collection signature；其他配置、源、目标、
未提交 checkpoint、冷读校验仍受原约束。12 个受影响 debug 检查通过，包括真实 strict checkpoint、
Adam/RNG 恢复及 transition commit 后中断重试。本次不改 Rust binary，无须重复已通过的准备或资格。

同步代码并退出失败 trainer 后，只运行：

```bash
python scripts/continue-rust-shadow.py --expected-generation 1257 --iterations 4 --share-target 0.25
```

保留 `iteration-1258` 中的全部文件和首轮测量；只继续剩余三轮至 g1261。
详见 [loader 修复说明](../../ai-research/RUST_SHADOW_BONUS_AND_FOUR_ITERATION_FIX.md#2026-11-08-checkpoint-loader-pending-work-repair)
和 [源码/测试凭据](shadow-resume-fix.json)。下方三条命令是首次准备流程，已准备的 lab 不需重做。

本目录只包含说明与短测原始报告。源码和测试从 Git checkout 取得，不重复打完整 bundle，
不要把旧 ZIP 提取到新 checkout 上。

兼容当前 teacher shadow 命令，保留 28+4 tables、zero branching 和 share target 0.25。
普通传输升级不重建或替换已通过资格的 teacher/search_v1 binary，不启用 teacher on-mode。
本次另有用户要求的 [teacher 验证修订](../../ai-research/RUST_SHADOW_VERIFICATION_REVISION.md)：
两档搜索均取消独立验证硬限时，改为稀疏计时；先准备新的 teacher ACK adapter，复用原 search core。
当前 adapter policy 为 `/3`：包含 `/2` 的 handoff 修复，并修正根节点无缓存时
补算训练 value 误用已到期搜索 deadline 的错误；补算结果及根特征复用，不增加搜索轨迹。
同一准备命令会检测并重建旧版，保留 checkpoint、缓存和原资格，无需手动删除 runtime。
实现和证据限制见 [implementation notes](../../ai-research/RUST_COLLECTOR_PERFORMANCE_PI00_PI03.md)。
后半部分与当前检查清单见 [PI04–PI05 completion](../../ai-research/RUST_COLLECTOR_PERFORMANCE_PI04_PI05.md)。

## Lab 准备与正常续训

先同步代码，退出旧 trainer；在项目根目录、同一 Python/Rust/mono 环境运行：

```bash
python scripts/prepare-rust-teacher-verification.py --expected-generation 1258 --offline && \
python scripts/prepare-rust-collector-performance.py --expected-generation 1258 --offline && \
python scripts/continue-rust-shadow.py --expected-generation 1257 --iterations 4 --share-target 0.25
```

第一条准备或复用取消验证限时的 teacher adapter，保留原资格与 sealed search core。
第二条只构建普通 adapter 并跑受影响的短 wire 回归；复用已缓存的 Rust 依赖。
若 Cargo cache 不完整，可省略 `--offline`。它校验真实 checkpoint/generation、原 core/bootstrap/source
与构建证据，再发布新的 ordinary runtime selector。同源码重复执行复用该 runtime。
成功或失败都打印 Git-visible results 与 original evidence 路径。

第三条是用户要求的四轮 shadow 资格测量续训（g1258 已完成，只运行剩余三轮至 g1261），不是短 benchmark，也不重做旧资格验证。
准备命令使用**实际 committed generation**。续训命令保留 starting generation 1257 与 iterations 4，自动只运行剩余轮数，
不要倒退到 1257，也不要删除 pending collection 来迁就命令。
新代码保留旧完整 family 的原身份和冷验证恢复路径。

这两个选项已保存在训练配置
`configs/step5/joint7-v11-2-g1290-trained-restart-rust-shadow32-no-branch-share025.json`
的 `execution` 中，无需 export：

```json
{
  "collector_validation": "production_fast",
  "replay_bindings": true
}
```

需要完整调试复核时，退出 trainer 后把该配置中的 `collector_validation` 改为 `strict`，
再运行同一续训命令；启动器保留配置文件中的选项。正常训练以配置为准，忽略这两个旧环境变量。
没有这两个字段的历史配置按上述默认值运行；启动器会补齐字段。它们作为运行设置单独传给
actor、publisher 和 reader，不改变已签名的学习配方或 frozen job，不为此归档 pending collection。
独立诊断工具仍可使用旧环境变量。相同轨迹在 strict/fast 中使用相同策略/RNG。
fast receipt 明确区分 structural_admitted 与 semantic_audited，抽检不冒充全量语义验证。
`profile` 与 fast 使用相同审核选择，但 IPC 精细计时完整开启；正常训练为每 64 调用采样。
`replay_bindings=true` 复用准确 owner 下的 row/array/target；退出后改为 `false` 可回到重复检查路径。
未知数据、可变对象、真正冷读、动作/IPC/取消/提交边界仍保留必要检查。
配置接线的 19 项短调试回归通过；没有重跑 benchmark。扩展检查另有 5 项旧 fixture/source-review/
mock 初始化错误，详细结果见 [配置修订检查记录](runtime-config-verification.json)，未记为通过。

## 可选的短组件验证

在**干净的 committed boundary**，可选运行真实同一 checkpoint、同一 frozen jobs 的三桌对照：

```bash
python scripts/profile-rust-collector-performance.py --expected-generation 1257 --device cuda
```

每臂最多三个实际 `(kind,V4 presence)` 分层首桌，每臂子进程上限 120s，不改 seed/对手/分支概率，
比较 C# reference 与当前 NativePool，并核对完整轨迹、feature 和实际 training targets。
只测普通 collection，不运行 teacher search 或更新模型；当前 committed shadow/off 配置明确记录。
结果自动导出至新的 results 目录。Rust 数据用 diagnostic admission，不能被普通训练读取器误接纳。
如果有 pending work，先用正常 continuation 恢复；此工具不会删除或绕过 pending 检查。
若已经推进到 g1259 等后续 generation，修改 guard 为实际 generation。
reference 的 process-launch 计时与异步导入无法在现有 API 完全拆开，scope 明确写在报告中；
总 accepted-work 包含全部启动、读取、准备和关闭，不用两种不同 scope 的 startup 数字推断收益。

只测 authored 固定数据，不加载真实训练 checkpoint、运行 NN 或更新训练：

```bash
python tests/RustBackendLab/profile_collector_performance.py --out artifacts/rust-backend/pi00-pi03-smoke-lab --families 4 --decisions 64 --workers 2 --export-results
```

只隔离 PI05-B 的默认复用开关（同一 fast coverage、同一正式模块/冻结数据），每次使用新输出目录：

```bash
python tests/RustBackendLab/profile_collector_performance.py --out artifacts/rust-backend/pi04-bindings-lab --families 2 --decisions 64 --workers 2 --bindings-compare --export-results
```

可加 `--reverse` 做第二次反序短测。工具在成功前要求 runtime/helper source 前后完全一致；
代码变化、输出不等价或超时都不能写成速度结论。每臂原始 phase report 单独保留。

可选的 24-publisher 压力只使用已完成 authored 数据，不模拟 320 桌或运行 NN：

```bash
python tests/RustBackendLab/profile_collector_performance.py --out artifacts/rust-backend/pi04-publish24-lab --families 24 --decisions 8 --workers 24 --bindings-compare --export-results
```

此项在 lab 执行；本机没有为覆盖规模额外启动 24 个 Torch publisher。CPU admission 仍共享、
总 authored decisions 最多 1024，各 IPC/启动等待有界；小批次默认两个 publisher。

使用新输出目录。三臂是 legacy strict / shared strict / shared fast，同一数据和进程数，
记录 startup、handoff、publication、read、training preparation 和 shutdown，验证训练输出等价。
失败也保留/导出 report。每臂 publisher 等待有 30s 上限，不自动扩成长会话。
真实 checkpoint/GPU/full-volume 的收益仍待 lab 正常报告；不能将此组件结果当成 NN throughput。

prepare 已自动运行 native 短回归。只复核受影响的 CPU 测试时：

```bash
PYTHONPATH=src:scripts:tests/R2:tests/Recovery:tests/FeatureEngineering:tests/RustBackendLab python -B -m unittest test_collector_performance.Performance test_b07_wire.Wire -q
```

后半部分的 CPU 控制/不可变边界检查为 `test_collector_performance_tail.Bindings`；prepare 同时运行
原 `NativeWire` 和新 `NativeTail`，核对合并页、stale token/cursor、取消、特征篡改和下一局重置。

导出后按工具打印的新目录，在 results 仓库提交结果；只提交本次目录，不连带无关修改：

```bash
git -C results add -- rust-lab-qualification/<printed-folder-name>
git -C results commit -m "Report collector performance smoke"
git -C results push
```

## 原始本地证据

- `cpu-components-forward.json`、`cpu-components-reverse.json`：两次反向顺序的短 CPU 发布/准备对照。
- `native-wire.json`：固定复杂/late roots 的普通 wire 对照；包含接收字节、特征复用字节与原始分项。
- `verification.json`：受影响检查与 source/runtime 绑定范围，lab Pending 项。
- `pi04-components-forward.json`、`pi04-components-reverse.json`：最终相同源码/冻结 jobs，复用 off/on 对照。
- `pi04-native-wire.json`：相同 native 状态/概率/RNG 的双页导出前后对照。
- `completion-verification.json`：最终代码、回归、前驱/feature cache 兼容与外部 Pending 项。
- `pi04-initial-eager-components.json`：早期整树 eager-freeze 的负面结果，保留原源码身份，已被替换。
- `pi04-invalid-source-run.json`：开发中源码发生变化的那次 probe 的无效标记；其原始耗时不用于结论。

这些报告包含启动噪声和较慢的分项，没有剔除负面结果。最终 compatible replay 仍写完整历史，
当前 publisher 仍需一次记录 JSON 解码、最终压缩和持久化；真正冷读仍检查内容 SHA。
PI04/PI05-B 的剩余实施已完成。最终当前代码通过 118 项受影响 Python 检查及 2 项 Rust unit tests；
前一阶段的 94 项证据保留原身份，不重复计数。真实 checkpoint/GPU/shadow 整轮性能留给 lab。

## g1284 Stage C and machine transfer follow-up

Current code adds an eight-game buffered Stage-C loss holdout and a compact,
replay-free server transfer. Only the existing fixed 2048-position IL probe and
its required provenance travel; original IL data/cache and replay buffers do not.
Stage A and its counter pause until eight fresh new-reward return ages exist.

The destination-only config uses **placement=.05**, win=1, VP=.001/cap180 and zero
insolvency charge. VP pays once after canal scoring, then at game end by the
capped cumulative difference (`brass-combined-return/3.0.0`). Source training is unchanged. Prepare/reuse the separate
zero-charge Rust teacher before destination preflight; the original ordinary
worker and installed game remain unchanged. Preserve all Adam/RNG state, stage
progress and the qualified fixed beta; retire old-reward targets/readiness.

Follow [the transfer guide](../../docs/STEP5_MACHINE_TRANSFER.md) and the generated
restore JSON's `prepare_reward`, `preflight` and `resume` commands. Focused debug
evidence is recorded in [stage-c-machine-transfer.json](stage-c-machine-transfer.json).
Actual destination transfer/continuation remains Pending lab; no local benchmark
or full qualification is requested.

The transfer exporter also supports the current critic-upgraded IL origin
(`upgrade.json` + `candidate.pt`, without a local BC state/report). It retains
the exact upgrade receipt and recursively required parent BC provenance while
continuing to omit original IL replay/cache. See
[transfer-upgraded-il-fix.json](transfer-upgraded-il-fix.json). This is a
transport-only repair: reuse the already prepared reward teacher; no training
qualification, native build or benchmark needs repeating.

The rating-transfer follow-up distinguishes committed compact PL outcomes from
full action replays. Exact state-pinned `ratings/slots/.../games.jsonl` summaries
are checked against their slot/spec/lineups and copied unchanged for rating
validation, future refits and compaction. Full replay, IL data/cache and training
buffers remain excluded. See
[transfer-rating-summary-fix.json](transfer-rating-summary-fix.json). Sync both
transfer scripts; training/native fingerprints and the prepared adapter stay
valid.

Clone restoration now reuses identical files and backs up differing Git-tracked
repository text/config/source files before installing exact archive bytes. It
recovers omitted dependencies only from the exact archive Git commit/hash,
preserves newer transfer tools with verified archived snapshots, and keeps
private state/checkpoint/binary and untracked conflicts blocking. All conflicts
are reported before writes; backup and fresh-buffer publication retries retain
their original destination identity. See
[transfer-clone-restore-fix.json](transfer-clone-restore-fix.json) and the transfer
guide. Update all three transfer modules on the destination and reuse the
existing archive and prepared adapter; no training qualification or native
rebuild is needed.

The follow-up also repairs legacy archives that omitted mixed-line-ending
manifests because Git reported them unchanged. Recovery checks a bounded set of
LF/CRLF forms against the full archived SHA-256; both reported Unity/Persistence
manifest byte sequences are reproducible from the recorded commit. New exports
prove Git blob/record byte identity before omitting code and automatically bundle
unproved working-tree bytes. Verify now explains an incomplete relocation after
a failed restore; the guide runs it only after restore succeeds. Keep the new
destination `ROOT` and reuse the existing archive and adapter. See
[transfer-git-newlines-fix.json](transfer-git-newlines-fix.json) for focused debug
evidence; actual lab restore remains Pending.

Frozen rating-runtime transfer follow-up: explicitly export live typed shared
search references from rating agents and pending evaluations, with all exact
receipts/manifests/binaries. Older archives can use `step5-transfer.py
repair-runtimes` and the small Git-delivered authored supplement for
`search-76a2f9ad7529c0e5e2117a36`. It reuses restored private DLLs by SHA-256,
keeps the rated identity and training state unchanged, preflights conflicts and
publishes the runtime receipt last. No old-machine access, rebuild, full archive
re-export or repeated preflight/qualification is required. Eight authored focused
regressions await server execution; no local tests or benchmarks were run.
See [transfer-frozen-runtime-fix.json](transfer-frozen-runtime-fix.json) and
the transfer guide for commands.

Old-machine full revision: `scripts/continue-rust-revised.py` keeps the original
project ROOT, preserves the newer g1286 run under ignored `.training-rollback/`,
restores the chosen g1284 export at its original run path and explicitly adopts
the complete revised rewards/fresh-buffer instruction. It journals the move and
supports interruption/retry without a second rollback or buffer reset. The
current teacher helper then enforces all A10/B20/C10, Stage-C eight-game held-out
loss stopping, joint KL .003, PPG floor .25, validation cadence, fixed beta,
branch-off/search_v1 and zero-charge/canal-delta reward settings. Seven authored
regressions await server execution; only static checks were performed locally.
See [old-machine-full-revision.json](old-machine-full-revision.json).
