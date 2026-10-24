# Training Side Revision 1

整理日期：2026-10-23。

范围：本侧聊近期完成的训练报告导出、PPG 缓存与性能优化、reward、PPO KL/LR 控制，以及从 g660 恢复 g210 风格 PPG 的修改。本文依据当前代码、配置和保存的测试记录整理；最后一版设置见下一节。服务器实际是否已经采用，应核对运行目录中的 `state.json` 和新 iteration 报告；本文不代表已经执行续训命令。

当前运行：`artifacts/step5/joint7-v11-2-grouped-lr-critic`。

已完成的 IL：`artifacts/step5/features4-v11-2-batch7-il`。

## 1. 最终方案：从 g660 采用 g210 风格 PPG

根据 g210 的保存配置及当时的 `value_phase.fit` / `actor_value` 路径恢复辅助阶段，并应用后续确认的 beta、LR、epoch 和 KL 检查频率修改。

| 项目 | 最终设置 |
| --- | --- |
| 配置 contract | `ppg-value-phase/14.0.0` |
| PPG 频率 | 每个 iteration 的 PPO 完成后运行一次 |
| Critic buffer | 当前 iteration + 上一个 committed iteration，共 2 轮 |
| 年龄权重 | 均匀权重；每个 epoch 重新 shuffle |
| Held-out | 仅当前 iteration；沿用 g210 分层 whole-game 划分，比例 5%，各存在的桌型在支持足够时至少 2 局 |
| 最大 epoch | 固定 3 |
| Value batch | 4,096 行 |
| Cloning 样本 | 当前 learner 最多 8,192 个非 forced 决策，每步抽取 512 行 |
| Cloning / KL reference | 冻结本次 PPO 完成后的 raw temperature-1 完整合法动作分布，不含 exploration mixture |
| KL probe | 使用当前 PPO 选择的 probe，目标 2,048 行；不足时使用实际可用行，不复制补足 |
| PPG hard KL cap | `0.002` |
| 定期 KL 检查 | 每 32 个 attempted PPG optimizer steps；另检查短 epoch 尾部及尚未验证的最终保留检查点 |
| Held-out loss 检查 | 每 16 步，另检查不足 16 步的 epoch 尾部 |
| Loss 接受规则 | 相对上一次接受的状态，composite row-mean held-out loss 必须不增加；不使用 1.5 SE 容忍度 |
| PPG LR | 当前 iteration 最后一个实际执行的 PPO step 所用 LR × 0.65 |
| 阶段内 LR | 固定，无 warmup / decay；下个 iteration 跟随新的实际 PPO LR |
| Value Adam | 跨 PPG 阶段保留 moments，不在阶段开始时清空 |
| Beta 初始值 | 从旧方案采用本方案时一次性设为 4 |
| Beta 范围 | 1–32 |
| Critic 缓存 | Dense FP32 mmap，不在 minibatch 中在线解包 |
| CPU 准备 | 复用现有 4-worker PPO preparation pool |

Cloning / probe 使用当前 live learner 数据，不再建立多个历史年龄的 policy cache。Snapshot / bot 决策不进入 cloning。Critic 训练保留 recorded MC targets 和原始 seat descriptors；whole-game / branch-family holdout 排除 fresh-target critic fitting。沿用 g210 当前 learner cloning 的取样语义，不把它混同于此前 8/16-age 方案的 policy holdout 协议。

### Beta 更新

- 保存最近 5 个 PPG 阶段的 KL-trip 标志。
- 本阶段新触发 hard KL cap，且最近 5 阶段至少 3 次触发：下一阶段 beta × 2，最多 32。
- 本阶段没有 KL trip、有实际保留的训练 steps，且最终保留 probe KL **严格小于 0.001**：下一阶段 beta ÷ 2，最少 1。
- 完全未接受训练的阶段，不利用恢复到起点后的零 KL 降低 beta。
- 阶段内不进行提高 beta 后的重试。重启恢复 checkpoint 中的 beta 和 trip window；不会每次启动都重设为 4。

### 分开的 loss / KL 检查与回退

第 16 步通过 loss 检查的状态，在完成 KL 检查之前只属于暂时接受。第 32 步 KL 超限时，按从新到旧的顺序验证暂存的 loss 检查点，返回最近一个同时符合 loss 与 KL 条件的状态。

Loss 拒绝后恢复的较早检查点也要验证 KL。较晚权重的 KL 合格，不能证明较早的不同权重也合格。最终保留状态必须有安全的 KL 记录。

回退同时恢复主模型、value Adam、ownership/local-scoring 辅助头及其 PPG Adam；实际保留 steps、distinct rows 和 presentations 随最终保留的更新前缀重算。间隔内的中间 step 可能暂时超过 0.002；每 32 步检查不构成逐步 KL 保证。

### 与当前 PPO 的关系

此次恢复的是 PPG 辅助阶段。当前 PPO 的 DCPG value-output preservation 继续使用：PPO 不重新拟合 fresh critic targets，fresh-target value learning 由 PPG 负责。

g660 配置转换只改变 `value_phase`。模型、optimizer/RNG、普通游戏进度、人口池与 checkpoint lineage、reward、对手、探索、branch、GAE/snapshot evidence、economic baseline 和 PPO LR-controller 状态均沿用所恢复 checkpoint 的设置。

## 2. 同期已完成并沿用的修改

### PPO 精确 KL 与 LR 控制

- PPO stopping 使用固定的 **2,048 行非 forced probe 的精确 categorical KL**，完整求和每个位置的全部合法动作。
- Raw temperature-1 movement 和实际 behavior-policy movement 分开测量、分开保护；保留严格的 pre-update behavior-probability reconstruction。
- 移除新 contract 下 sampled chosen-action KL 对 stopping 的控制，保留为诊断指标。这里的“精确”是 probe 中每个状态的动作分布求和，不是扫描全部训练状态。
- LR controller 的保护性下调由除以 1.5 改为 **除以 1.25**。现有触发条件仍适用：早于一半 planned steps 停止、近 3 轮至少 2 次 early stop，或 post-PPO KL 超过 1.5 × target。
- 单次较晚的 KL stop 仍按已有窗口规则处理，不额外引入“任何 stop 都立刻降 LR”的新规则。
- 保留 multiplier、window、cooldown、warmup、clamp 和健康证据；旧显式配置仍按其历史 contract 执行。

### Reward 与 branching

当前配置的 insolvency 为：

```text
event_cost = 0.04
max_cost   = 0.10
```

每 seat/game 的前两次事件各扣 0.04，第三次只扣剩余 0.02，之后不再增加。Ordinary / search reward specifications 同步更新；旧对手仍按其自身 checkpoint 的 reward contract 校验。

当前 branching 为 **每个 eligible decision 0.5%**，即 `rate_per_decision=0.005`。保留已有 opening temperature、fork 后计数和 branch 限制。

PPO preparation 的现有设置为 `microbatch_size=512`、`max_candidate_cells=131072`、`preparation_workers=4`；这些不是本次 g660 PPG 转换额外调整的项目。

## 3. PPG 缓存与 ownership 性能优化

最终使用 dense FP32 `critic.npy` / `actions.npy` mmap 布局。当前 g210 路径只需要 dense critic 数据和 local-scoring sidecar，policy reference 使用当前 live 数据。

- 不在每个 minibatch 中恢复 bit-packed 特征、填常量列或按字典重建 descriptor。
- 复用持久的 PPO CPU preparation pool；worker 读取经过校验的 sealed dense arrays，不复制整份 replay 到临时 PPO staging 文件。
- 采用有界预取，并按原采样顺序消费结果；RNG、shuffle 和索引生成仍在主进程。
- Dense PPG preparation 使用有界 shared-memory 输出 slot；消费端先取得私有 tensor，再复用 slot。正常结束、early stop 和异常均释放 slot 与待处理任务。
- Ownership sidecar 每个 age 打开时只解码一次；metadata / label 用向量化 gather，按唯一 game/era/seat mapping 复用不可变 targets。
- CPU ownership target cache 上限提高到 65,536，减少多年龄阶段内的 cache churn。没有缓存模型输出或 autograd graph。
- Tensor、target、descriptor、seat rotation、mask、VP scale 和梯度一致性有回归验证。

此前 dense-eight 小型 CPU benchmark：同一批 authored tensors、相同索引顺序，32 个 512-row batches，warm workers，三次计时取中位数：

| 输入准备路径 | 最终一次 benchmark，秒 |
| --- | ---: |
| Packed，同步 | 0.51422 |
| Dense mmap，同步 | 0.44527 |
| Dense mmap，4 workers + shared memory | 0.36402 |

该试验中准备耗时较 packed 同步路径减少约 29%；ownership metadata/label assembly 从 0.01803 秒降到 0.00502 秒。此处是局部 CPU 测量，不能据此推算服务器完整 PPG 阶段或 GPU 训练的加速比。

## 4. 报告导出修改

`scripts/export-training-reports.py` 的导出 manifest 为 `selfplay-report-export/1.2.0`。

- 导出的 `retention-history.json` 删除字符串外的 JSON 排版空白，保留字符串、转义和数字 token；不通过浮点 parse/dump 改写内容。
- 原训练目录的 history 保持原样，其他报告保持原始字节。
- Manifest 的 `bytes` / `sha256` 描述导出文件；`source_bytes` / `source_sha256` 描述原文件，并标记 `json-whitespace-only/1.0.0` transform。
- 增量导出复用依据导出文件的真实字节与校验和，源文件核验按 source hash 处理，避免压缩空白后出现 checksum 不匹配。
- 保留旧 manifest 的读取兼容和已有 previous-bundle 依赖检查。

该修改减少导出文件体积；没有执行 Git 历史重写，也不保证任意后续 history 都低于 GitHub 的 50 MB 建议值。

## 5. 历史方案及被替代设置

以下用于解释 g620–660 日志，不应与当前最终方案混用。

| 阶段 | 当时实现的方案 | 当前状态 |
| --- | --- | --- |
| g620：16-age PPG | Buffer / held-out 均覆盖 16 轮，每 16 轮运行；均匀权重；loss 16 步、KL 32 步；关闭 hard KL cap，只保留 penalty；beta 固定为保存值 | 已被后续方案替代；历史 contract 保留 |
| g620：无损打包 | 按特征结构存储常量列、二值列及 selector，并去重游戏级 targets / descriptors；保留原始 FP32 位模式和随机 gather 顺序 | 历史读取保留；当前新缓存不用在线解包 |
| g620：minimum-loss | 每次出现更低 held-out loss 保存匹配的模型/Adam，阶段结束返回最低 loss 的状态，包含 step 0 | 被后续 last-accepted 规则替代 |
| g650：dense-eight | Buffer / held-out 均覆盖 8 轮，每 8 轮运行；相对上次接受 loss 使用 1.5 paired-SE；beta 固定；LR=0.35×实际 PPO LR；阶段开始 reset value Adam；当时 max epochs=5 | 被 g660 的 g210 风格 PPG 替代 |
| g660：最终方案 | Buffer 2、每轮 PPG、只用当前 held-out、严格 loss 不增加、hard KL=0.002、KL 每32步、LR=0.65×实际 PPO LR、beta 从4起自适应、保留 Adam、最多3 epochs | 当前应生成并采用的配置 |

曾提出的 `max_cost=0.16` 和 `max_epochs=6` 已被后续确认覆盖；最终分别是 **0.10** 和 **3**。g210 原来的逐步 KL 检查也已被最后的 **32-step** 请求覆盖。

g640 查到的 `.ppg-eight-cache/generation-000640*.json` 属于训练数据/标签缓存，不是含模型、Adam、RNG 和控制器状态的完整 checkpoint。缺少完整备份时不能用它们准确回滚。用户随后选择从 g650 继续，最新提供的运行报告已到 g660。

## 6. 续训状态与缓存迁移

配置 helper 检查 committed generation、完整 checkpoint 文件及 hash、允许的 source/config transition，然后只生成新配置；它不直接修改训练运行。

实际 trainer 在 run lock 下恢复 checkpoint，持久化经过 review 的配置转换后，才清理属于本 run 的旧 PPG 缓存。Holdout 协议改变，因此不能把旧缓存直接当成新两轮数据使用。未知文件、owner 不匹配或损坏文件会阻止清理。

- 采用 g660 新方案时 beta 一次性设为 4，清空旧 beta-trip 证据；GAE / snapshot 等独立状态保留。
- 首个新阶段为 **g661**，只使用当前一轮新数据。
- 从 **g662** 起具备当前 + 上一轮新数据。
- 新缓存过期文件只在新 checkpoint/state commit 后删除。
- 采用配置后、首个新 checkpoint 前发生中断，可以幂等重试清理；正常新 checkpoint 重启后恢复保存的 beta/window/cache，不重置迁移起点。
- Source review 保留已有受限迁移路径，并增加 g660 的明确 PPG-only 转换；不会以任意 source 改变为理由绕过配置兼容检查。

## 7. 验证记录与边界

本次文档整理没有重新运行训练或测试；下列为实现阶段已经保存的验证结果。

| 修改阶段 | 已保存的验证结果 |
| --- | --- |
| Reward 0.04 / cap 0.10 | 13 项 focused tests 通过，覆盖部分扣费、配置、缓存清理和续训状态保留 |
| Dense-eight / PPO exact KL | 最终组合回归 25/25 通过；小型 PPO/PPG + ownership smoke 通过，并验证 g7→g8 精确重启 |
| g660 KL32 回退/频率 | 8/8 tests 通过，109.21 秒 |
| g660 配置/续训/门控 | 11/11 tests 通过，40.98 秒 |
| g660 独立 smoke | 三轮小型 PPO/PPG，39.60 秒；第2轮 checkpoint 重启后第3轮结果一致 |

g660 smoke 保留 PPG steps 为 **3、6、6**，训练行数为 **108、217、218**，beta 为 **4→2→1→1**。模型、PPO/value/auxiliary Adam、RNG、beta/trips 和 gate state 在重启后匹配；缓存按两轮窗口在 commit 后过期。输入 checkpoint checksum 未改变。

该 smoke 仅有 121 个 eligible probe rows，阶段在 32 步前结束；真正的第32步检查、跨 epoch 计数及未验证检查点回退由 focused regressions 覆盖。没有在本地完成生产规模 2,048-row CUDA benchmark 或长程强度评估。

更广的历史测试探索中，还存在旧默认值断言及未批准的古早 source 直达路径问题；本报告的 focused pass 数不代表整个历史测试仓库全部通过。所有 CPU 小型 smoke/benchmark 仅验证执行、正确性和续训，不证明 EV、稳定性或棋力提升。

主要验证记录：

- `artifacts/insolvency040-cap100-g620-tests.log`
- `artifacts/ppg-dense-eight-g650-exact-kl-tests.log`
- `artifacts/ppg-dense-eight-g650-exact-smoke/report.json`
- `artifacts/ppg-g210-kl32-guards.log`
- `artifacts/ppg-g210-kl32-continuation.log`
- `artifacts/ppg-g210-g660-kl32-smoke/report.json`

## 8. 主要实现入口与说明

| 文件 | 作用 |
| --- | --- |
| `src/brass_ai/recovery/ppg_g210.py` | 最终两轮 PPG 配置、LR/beta、采用与 checkpoint 生命周期 |
| `src/brass_ai/recovery/value_phase.py` | 严格 loss 接受、32-step KL、待验证检查点和模型/Adam 联合回退 |
| `src/brass_ai/recovery/actor_value.py` | 接入 PPO→PPG、状态保存、门控证据 |
| `src/brass_ai/recovery/ppg_replay.py` | Dense critic 缓存、两轮窗口及 commit 后 retention |
| `src/brass_ai/recovery/ppg_packing.py` | 历史16轮方案的无损打包支持 |
| `src/brass_ai/recovery/ppo_preparation.py` | 多进程准备、顺序消费和 dense PPG transport |
| `src/brass_ai/recovery/local_scoring_auxiliary.py` | Ownership/local-scoring 标签与训练 |
| `src/brass_ai/recovery/ppo_kl_control.py` | PPO 精确 probe-only 控制 contract |
| `src/brass_ai/recovery/lr_controller.py` | LR cut divisor 1.25 与现有 controller 状态 |
| `src/brass_ai/recovery/config.py` / `resume.py` / `value_buffer.py` | 严格配置校验、reviewed migration、受控缓存清理 |
| `configs/learning/selfplay-resume-sources.json` | 明确 source/config 迁移许可 |
| `scripts/configure-ppg-g210.py` | 当前 g660 新配置生成入口 |
| `scripts/export-training-reports.py` | History 紧凑导出和 manifest / 增量校验 |
| `tests/Recovery/test_ppg_g210.py` / `smoke_ppg_g210.py` | 当前方案 focused tests / 独立 smoke |

详细阶段报告：

- `ai-research/PPG_SIXTEEN_G620.md`：历史16轮、打包、minimum-loss 与 reward 修改。
- `ai-research/PPG_DENSE_EIGHT_G650.md`：历史8轮、性能、PPO KL/LR 和 branch 修改。
- `ai-research/PPG_G210_G660.md`：当前最终两轮 PPG 方案。
- `docs/EXPORT_TRAINING_REPORTS.md`：导出 manifest 与增量复用。
- `docs/TRAINING_RECOVERY_LAB.md`：当前续训入口和历史说明。

## 9. 当前续训命令：g661–680

在 Linux 训练服务器的仓库根目录，同步包含上述修改的代码后执行：

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 660 --apply && \
python scripts/configure-ppg-g210.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 660 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppg-g210-g660.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-g210-g660.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 660 --iterations 20
```

Recovery 处理未提交的中断 iteration；不会把已经提交的更新倒退。Generation 不匹配会停止。正常完成后继续复用同一配置，只执行训练命令并使用新的 committed generation，不要借重新执行迁移来重设 beta 或缓存起点。
