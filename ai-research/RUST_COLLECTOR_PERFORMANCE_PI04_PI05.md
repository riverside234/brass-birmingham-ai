# Rust collector performance: PI04–PI05 completion

2026-11-07。PI04、PI05-B 和检查矩阵在正式 collector/reader 路径的实施已完成；本地只运行受影响的
调试回归、冻结 authored 数据与短 native wire 对照。当前 g1257 的 Linux/CUDA 普通采集加完整 shadow
整轮收益仍须在 lab 实测，不能把组件结果写成训练吞吐结论。

## 实际运行路径

仍是 `league → RoutedPool → NativePool → collect_lane → Publisher → ingress → Dataset/PPG`。
没有另一套只供 qualification 使用的快速 collector。保留 28+4 teacher shadow tables、零普通分支、
share target 0.25 和原 fixed-beta 资格流程；search_v1、teacher Rust 算法、模型/Adam/RNG、reward、
feature dtype/布局、采样概率和数据配方不变。

### PI04

- 新普通 binary 的 `ordinary-export-pages/1` 在同一个答复中给出完成 receipt、record page 和
  feature page。记录与特征使用独立 cursor，保留各自原有 64 行/字节分页边界；最后一页退休 lane，
  省去 finish、分别拉取页和 release 的往返。旧 binary/诊断接口保留兼容路径。
- 每个子页保留 16MiB 上限；仅此双页信封允许最多 36MiB（两个旧页加控制信息），旧数组操作仍为
  16MiB、JSON 控制仍为 4MiB。不会为合并请求而截断合法候选或缩短历史。
- Pipe 使用有界 `readinto`，不再每次 OS partial read 都创建 bytes 再扩展 bytearray。
  Rust 分别写两个 Vec，不再为了信封拼接一次大 payload；仍有必要的 pipe/CPU 拷贝。
- fast admitted frame 的 canonical view/choices bytes 同时用于原 input digest 和原 Request digest；
  Request 原 JSON 排序、转义和 SHA 公式完全保持，避免再次编码整段历史。
- 不跨非强制决策预测下一步 context，不移动 Python 采样 RNG。原 16-stage forced batching、
  turn/era/debt 停止条件、逐行记录、branch 机会和 NN admission/deadline/cancellation 保持。

### PI05-B

- publisher 在同步、私有、已独立验证的数据作用域内复用 row/seal/feature binding。它不会把
  `validated=true` 写到文件后自行相信，也不为只使用一次的最终写入深拷贝整棵回放树。
- family 只做一次 canonical JSON 编码；相同 bytes 用于原 family SHA 以及最终 gzip，SHA 字段
  在被哈希对象之外追加，保持原 digest 公式和 replay 格式。避免为了 hash 后再写一遍而重复编码历史。
- reader 验证准确 replay/feature bytes 后，使用自己独占 JSON 的只读视图；按访问需要包装子树。
  通用借用输入仍使用完整 detached freeze。只读视图的 get/items/values、copy、切片、拼接和
  pickle 都有回归；正常 API 不会暴露可写子树。跨进程不保留本地能力。
- ingress 返回前只分离 compact choices/labels/headers，共享 envelope 只分离一次；单 reader
  不经过 pickle 时也不会让这些引用间接保留整局 JSON。board history 不进入 learner。
- CheckedGame 对不可变输入复用完整 history/input/label seal；metadata 用准确根身份、keys 和
  branch owner 检查。改变标签、context、header、row 或 feature object 仍失败。
  可变/未知对象继续完整重哈希；新 neutral context 绑定新的准确状态特征，不改 action arrays。
- 数值数组在首次接纳时检查 finite/selector/shape，并隔离为 bytes-backed 数组。NumPy 的
  `writeable=False` 不单独作为证明；布局变更仍被拒绝。Dataset/InputAudit 和 PPG policy cache
  对此能力复用 selector admission 与数组 SHA，不在每个 epoch 再扫描同一内容。
- terminal、insolvency 和 reward targets 按不可变 game 与准确 spec/prior 复用。branch expansion
  复用已接纳的分支覆盖，不在训练准备阶段悄悄对所有 branch 再做完整概率/RNG 复演。
  每个 child 的 source-row、seed、chosen、excluded-fork 和 context/输入绑定仍持续检查。
- Native 和 legacy/reference derived-features/4 复用相同读写器，避免反复 hash/open 同一缓存。
  ordinary fallback 的 replay 用同一份已哈希 bytes 解码；CheckedGame 和特征凭据保持到读取端，
  不因途经旧路径而丢弃数组并重新编码。
- 精确审查保留旧 Features-4 cache encoder 身份；未知依赖不继承兼容性。已知旧 v2 receipt 的
  冷恢复保留原确定性 coverage；未知源或 strict 模式走完整语义/schema 检查。仅有数组不能绕过
  未知 observation 的 schema。真正的冷读仍逐字节校验，不用 mtime/文件大小冒充内容证明。
- RoutedPool 在固定运行期保留已检查 runtime，不在每次普通 collect 再读二进制和 build 文件。
  新部署/reload 必须新建进程。已准备过 PI00–PI03 的 runtime 可作为经审查的前驱保留，旧完整
  family 仍以其原 producer 恢复；最多保留 8 个有精确 pins 的前驱，不改写旧 receipt。
- 写入中断后只有 replay/cache、缺少最终 record marker 的当前 pending family 不被接纳。
  重采前在原 owned 目录内保留原字节到 `unclaimed/`，不会因新的 elapsed seconds 或 build ID
  不能匹配半次尝试而卡死；有最终 record 的完整 family 保持原恢复路径。

## 默认检查策略

训练配置 `execution.collector_validation="production_fast"` 与 `execution.replay_bindings=true` 为默认值。
它们已写入当前 shadow continuation JSON；不需要环境变量。正常训练按配置安装并传给 spawned
actor、publisher 和 reader，配置优先于旧环境变量。启动器保留 JSON 中显式修改的选项。
历史遗漏使用相同默认值；运行设置不进入 frozen learning signature，不清空/归档已有 pending work。
旧环境变量只保留给独立诊断工具。
strict 保留原语义容差；开关不改变策略、标签、seed、模型或已提交进度。

| 类别 | 正常路径 |
| --- | --- |
| RNG、完整概率/temperature/exploration 分布复演 | parent ordinal 每 64 个抽 1 个；每 collection 首个 table/policy 分层额外 full audit |
| branch 全历史/alternative 概率复演 | frozen job 稳定 hash 约 1/16，加上述 full audit；child 的基本亲缘/row/seed 必检 |
| 额外 reward/auxiliary 语义审核 | 随 semantic audit；准确 target 生成和 producer/result/seat 绑定照常 |
| observation/choice/history schema 深检 | 首次/入选 schema audit；typed native 普通帧只做必要契约；未知输入 strict |
| 完整 view/choices 重复比较 | 准确不可变身份相同时复用；borrowed/mutable/foreign 输入保留内容检查 |
| finite、selector、style/intent、seat layout | 首次接纳建立；下游准确数组能力复用。可变数组不靠只读 flag 跳过 |
| Python 重新编码 Rust 特征并对照 | 默认关闭；只在受影响 strict fixtures/显式短 probe |
| 每次 encoded/assert_metadata/pack 的 history/label 重哈希 | 不可变 owner 或 publisher 私有作用域复用；一般可变 CheckedGame 保留原检查 |
| replay/feature/chunk SHA | 封存、跨内容边界、冷读各一次；普通 reader 不先 hash 后 reopen/decode；临时文件退休不再重哈希 |
| 已知完整 receipt 冷恢复 | 准确 bytes、source/scope/job/commit/rows 必检；语义审核依原 coverage，不把 structural-only 冒充全审计 |
| 未知 producer/schema/缺失 receipt/外部导入 | strict 或拒绝；不能继承本地能力 |
| policy/context/config、source/build | 注册、game/seat reset、部署/依赖切换检查；固定作用域复用 |
| PPG/critic 热读取 | 完整保留 age/source/holdout/labels 绑定；已接纳不可变 arrays 复用，mmap 新内容边界与实际使用的轻量布局检查保留 |
| oracle/lockstep、重复 schema 遍历、per-row diagnostics | 正式路径关闭重复工作；必要生产检查与测试仍保留 |
| 进程巡检/资源/IPC clocks | coordinator 每 5s；资源既有低频采样；精细 IPC 1/64，calls/bytes 全量 |
| checkpoint/optimizer 全量恢复演练 | 仅受影响资格/明确恢复；实际 commit、strict loading、weights_only=True 不降频 |

该表实施的是准确条件下的复用和稀疏审计；没有 blanket disable。首次 numeric/content 检查、真正的
冷读、mmap/可变输入边界、动作合法性、token/revision/actor、IPC bounds、取消及 commit 原子性仍必要。

## 短测、回归与报告

当前最终代码的受影响调试组 **79 + 39 项 Python 全部通过**，另有 2 项 Rust unit tests 和原生
wire 集成检查（包含在 Python 组中）。另保留前一批 shadow/channel/recovery 的成功记录。
没有运行本地完整 NN 游戏或真实训练更新。

覆盖：完整动作/RNG/features/record SHA/targets 等价、forced/V4/branch、旧缓存和未知 schema、
mutable alias、neutral context、pickle 能力剥离、metadata/layout 篡改、stale token/cursor、取消、
自动 release 后新 game、权重/Adam/RNG 对照、checkpoint strict/weights-only 与 cache recovery。

`native_backend.collection` 记录 actual runtime/completion_export、validation policy、binding_checks、
shared buffers 和分项 timing；每个 publication 有实际 binding counter，reader metrics 另报。
计数区分 input/label/metadata hash 执行、不可变 row/array/target 复用和 private publication reuse。
actor 的计数用 context-local scope，避免两个 lane 的 process-global 差值重复计数。

曾测到提前 eager-freeze 整棵回放树使 publication/read 回退；原始负面报告保留。最终实现改用上述
私有发布作用域和按需只读视图。另一次开发中源码发生变化的 probe 明确标记无效，不用其耗时作结论；
组件工具现在强制检查 runtime/helper source 在前后完全相同，并保存每臂原始报告。
最终冻结组件与 native wire 的实测值见 distribution supplement；不外推整轮 Linux/GPU 吞吐。

最终两次正反顺序对照都是相同 2 family / 128 decisions / 2 publisher，只有 immutable bindings
off/on 不同，RNG/语义审计选择相同。没有 NN/checkpoint；每次均核对完整输出与 source 未变化。

| 阶段中位 wall time | off | on |
| --- | ---: | ---: |
| startup | 6.334s | 6.471s |
| handoff | 0.0096s | 0.0146s |
| publication | 0.651s | 0.526s |
| replay reading | 0.401s | 0.370s |
| training preparation | 0.0629s | 0.0349s |
| shutdown | 0.561s | 0.553s |
| accepted-work 含 startup | 8.021s | 7.969s |
| accepted-work 去除单列 startup | 1.686s | 1.498s |

publication 中位下降约 19.3%，preparation 约 44.5%，不含 startup 的 accepted-work 约 11.2%。
包含 startup 后仅约 0.6%；反序那一次 total 更慢，startup/handoff 的回退和 first-reader 噪声均保留。
该 off/on 对照不单独归因 PI04/canonical-write 等双方已启用的改动，也不是全训练加速比。

native wire 用两个固定 roots 交替对照：完整 late root 两决策由 9 次 RPC 降为 7 次，actions/RNG/
features/记录 SHA/branch 机会相同；生产收集还省去最后 release RPC。423-candidate complex root
只推进 24 决策、未完成导出，调用数保持 51，不能据此声称其完成导出加速。

剩余成本主要是进程启动、必要的完整 replay JSON 解码/最终压缩持久化、首次内容 SHA 与数组读取。
旧兼容 replay 仍保存完整历史；本次未新增紧凑持久化格式，也未宣称这些必要成本消失。

## Lab 续训与回退

同步全部代码后，在 lab 根目录、当前训练 Python 环境执行：

```bash
python scripts/prepare-rust-collector-performance.py --expected-generation 1257 --offline && \
python scripts/continue-rust-shadow.py --expected-generation 1257 --iterations 2 --share-target 0.25
```

第一条 Python 命令只准备普通 binary 和受影响短 wire checks，不启动训练或替换 teacher。
第二条才进行原来的 g1257→g1259 两轮 shadow continuation。若 checkpoint 已推进，使用实际起点；
不删除 pending work、不倒退 generation，不重做通过的 B01–B07。
同 Rust source 的重复 prepare 可复用 binary；新的 Python execution/source 身份仍真实记录。

可选短测和 24-publisher authored pressure 命令见 supplement，均不需要再打一份完整 bundle。
若需要取消本次不可变复用，退出 trainer 后将 shadow 训练 JSON 的 `execution.replay_bindings` 改为 `false`；
可进一步将 `execution.collector_validation` 改为 `strict` 做完整语义复核。回退旧 wire 要在退出后移开新的普通 runtime
selector，保留原 checkpoint/run/cache/资格证据；不得在运行中的 trainer 下替换文件。

成功/失败结果自动导出至 `results/rust-lab-qualification/`，Windows 长路径也受支持。
整轮性能的最后一项验收是 lab 新正常报告确认实际路径，并单列 ordinary/shadow/teacher 时间。
