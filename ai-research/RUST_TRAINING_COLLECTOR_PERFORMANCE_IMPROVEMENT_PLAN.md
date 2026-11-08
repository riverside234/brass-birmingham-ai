# Rust Training Collector Performance Improvement Plan

更新日期：2026-11-07。状态：**PI00–PI05 的正式代码实施与受影响本地短回归已完成。
最新 checkpoint 的 Linux/CUDA matched profile、24-publisher 压力实测与正常 shadow 整轮收益待 lab 验证。**
本次修订扩大检查关闭与降频范围：正常训练计划默认使用 `production_fast`，
将独立语义复演、深层结构审核和诊断检查从逐帧/逐 family 改为按策略抽检、首次检查或默认关闭。
已实现的模式通过训练配置 `execution.collector_validation` 选择 `production_fast`、`strict` 或 `profile`，
`execution.replay_bindings=true` 开启不可变绑定复用；当前 shadow JSON 已保存这两个默认值，续训不需 export。
它们是运行设置，跨 spawn 显式传递，不改变 frozen 学习配方/已有 job。旧环境变量仅供独立诊断工具使用。
检查矩阵现已按数据所有权/来源/模式实施；必要边界和可变数据检查保留，不能理解为全部检查关闭。
具体交付见 [PI00–PI03 实现记录](RUST_COLLECTOR_PERFORMANCE_PI00_PI03.md)、
[PI04–PI05 完成记录与检查清单](RUST_COLLECTOR_PERFORMANCE_PI04_PI05.md) 与
[distribution 操作说明](../distribution/rust-collector-performance/README.md)。

本计划承接 [B07 实现与测量记录](RUST_BACKEND_B07_IMPLEMENTATION.md) 和
[Rust 后端迁移计划](RUST_AI_BACKEND_MIGRATION_PLAN.md)，优化正式训练的普通自博弈采集路径。
目标是消除重复传输、序列化、文件中转和验证，使 Rust 引擎的收益能够体现在
**最终被训练器接纳的数据所需的总时间**上。只增加 publisher 数量不等于减少工作量。

## 本次实施范围与最新 shadow 配置

按用户最新指示，兼容现有命令：

```bash
python scripts/continue-rust-shadow.py --expected-generation 1257 --iterations 2 --share-target 0.25
```

保留该入口的 28 train + 4 held-out teacher shadow tables、零普通分支配置、诊断梯度和固定 beta
资格流程；不自动开启 teacher on-mode。本次普通 collector 优化也兼容已有 teacher-off 配置。
普通性能与 teacher search/barrier/梯度诊断时间分开报告，不能用 teacher-off 报告作为整轮 shadow 的匹配基线。

| 阶段 | 当前交付 |
| --- | --- |
| PI00 | 每 op 全量 bytes/calls、采样计时及覆盖计数；startup/recovery/publication/read/preparation/shutdown 分项工具；冻结 CPU/wire 短测 |
| PI05-A 前置 | 显式 fast/strict/profile、1/64 family RNG/概率审计、稳定哈希 1/16 branch 审计、每 collection 首个分层全检、5s 集中健康巡检、1/64 IPC 计时 |
| PI01 | 原生 actor 返回 raw pages，parent-owned 有界共享 chunks；publisher 唯一完成记录树解码；不写 native 临时 payload 文件 |
| PI02 | 新普通 Rust binary 按准确 SHA 验证并省去已推理 feature bytes；forced/V4/其他未推理行仍完整导出 |
| PI03 | lossless history 前缀/增量、静态 view 字段与相同候选集合复用、canonical record history 字节增量；旧最终 replay 格式可读 |
| PI04 | 完成 receipt/record/feature page 合并、最后一页退休、readinto、分段 Vec 输出及原 Request canonical JSON 复用 |
| PI05-B | 私有 publication row binding、按需只读 reader、immutable arrays/target memo、旧 cache/source 精确复用与 fallback 单次读取 |
| Lab 验收 | 最终普通/完整 shadow 整轮记录与实际 Linux/GPU 资源收益 Pending；不再有未实施的 PI04/PI05-B 代码项 |

启用 PI02–PI04 须先用新的 prepare 命令构建并验证普通 collector；只同步 Python 不会把旧 binary
变成新协议。未准备新 binary 时保留旧 wire 的兼容路径，并在 report 的 collector_runtime 明确记录 legacy。

## 1. 范围、现状和证据边界

正式训练调用链是 `recovery.league → lab_backend.RoutedPool → lab_pool.NativePool`。
以下改动必须进入这个实际调用链，不能只修改 qualification 脚本或 bundle 副本。
qualification 负责调用、测量和验证生产实现，不另写一套更快的测试采集器。

已有实现应先检查并复用，不能作为新增工作重复开发：

- 紧凑动作 ID、revision 绑定、二进制特征和按批导出记录。
- 与 reference 对齐的连续强制动作处理，以及原始 Python 策略、采样和分支 RNG。
- Rust 持有 V4 混合桌状态，原 C# V4 worker 保留决策、计划队列和预算语义。
- 保留 CheckedGame 特征及绑定、不可变输入复用、增量记录哈希和受限策略/进程缓存。
- 自适应 publisher 数量；当前 320 个 parent family 使用最多 24 个 publisher，
  actor、publisher、读取/准备任务仍受同一 60 CPU-slot admission 限制，小批次自动缩减。
- 缓存优先恢复、仅启动缺失 actor、冷数据完整验证、已验证热数据复用，以及较低频的进程存活检查。
- collection-boundary 推理 channel、共享槽位、进程句柄的释放与失败清理。
  该生命周期修复见 B07 的 g1257 descriptor-exhaustion 记录；本计划不得退化或重复实现它。

问题证据来自 results commit `097327ef4fab7190f8d91ce4996dd1d4db4c5ce2` 中的
`joint7-v11-2-g1290-trained-restart-g1253-20261107T145415686647Z/iteration-1252/report.json`：

| 历史指标 | 观测值 | 解释 |
| --- | ---: | --- |
| collection wall time | 496.159s | 320 Rust families，0 reference，0 cache hit |
| actor / lane / publisher | 24 / 48 / 2 | 发生在改为最多 24 publisher 之前 |
| 未完成 family 峰值 | 92 / 96 | 发布路径拥堵的证据 |
| IPC 接收数据 | 37,763,969,069 bytes | 约 37.8GB，跨所有 lane 的总量 |
| IPC 调用 | 186,321 | 跨所有 lane 的总量 |
| feature_arrays 调用 | 61,462 | 推理阶段的特征请求 |
| drain_records / drain_training 调用 | 5,549 / 4,443 | 结束后的批量记录与特征导出 |

这些历史数字用于定位问题，不能作为新增优化的当前基线。B07 R4 的三 family matched
短测曾将 accepted-work 从 30.632s 降到 26.820s，但未证明 320 family 的规模收益。
后续实施必须先读取最新正常训练报告，确认源版本、实际并发、缓存命中和资源释放状态。
旧 C# 的 246.754s / 258.527s 是不同迭代的运行参考，不能当作冻结任务的严格速度比。
worker 时间会重叠，不得把 worker 时间之和解释为 wall time，或直接除以 publisher 数量估计收益。

## 2. 不变条件和目标路径

保持 search_v1、V205 disabled、当前明确选择的 teacher off/shadow 模式、模型结构、特征版本、精度、训练配方、
reward、PPO/PPG/local-scoring 目标和优化器状态不变。已有任务、checkpoint、Adam、RNG、
population、critic ages、缓存所有权和恢复语义必须能够继续使用。

动作与候选顺序、opponent/policy 身份、采样概率、随机数消耗顺序、强制动作记录、
分支机会与 excluded-fork 语义、V4 多动作计划与预算保持等价。不用降低采样量、减少训练记录、
删除历史训练信息、改变浮点精度或放宽数值比较容差换取速度。
允许减少独立诊断/语义复演的执行范围；输出不变与审计覆盖降低是两件事，不能把抽检宣称为逐条完整验证。

目标数据流：

```text
Rust authoritative state
  → 已确定 revision/context 下的记录和特征
  → 分块、受限、完成后不可变的共享缓冲区
      ├─ 现有推理服务读取准确特征
      └─ 独立 publisher 执行必要接纳，深度语义审计按策略抽检，复用准确特征
  → 一次最终持久化与原子接纳
  → 训练准备/读取复用同一已验证内容
```

“一次持久化”指取消临时 payload 的重复写读，以及每个最终 replay/feature payload 只写一次；
不要求立即合并现有 replay、feature cache 和小型 receipt 文件。先保持旧持久化格式兼容。
共享内存不能取消 CPU/GPU 的必要读取、GPU 上传、最终存盘和恢复校验；不要承诺零开销。

检查优化分两层实施：先把 debug/重复深检默认关闭，并把昂贵独立复演降为稀疏审计；
随后结合不可变 chunk、缓存和内容绑定移除剩余扫描。不要等 PI01–PI04 全部完成才减少验证成本。
核心动作推进和接纳只保留轻量且不可替代的检查，其余逐项说明默认频率和适用数据来源。

## 3. PI00：建立短测和成本归属，再改数据路径

**状态：Implemented，CPU/wire 组件证据已保存；最新 checkpoint 的 GPU 与整轮测量 Pending。**

复用 `tests/RustBackendLab/profile_b07_publication.py`、现有 B06/B07 冻结任务、
正式 RoutedPool 和 report exporter。补齐以下分项，双方用相同低开销统计方式：

| 阶段 | 必须记录 |
| --- | --- |
| startup / shutdown | publisher、actor、推理 channel 初始化及释放的 wall time |
| IPC / serialization | 每种 op 的调用数、请求/响应字节、JSON 编解码、实际拷贝字节与等待 |
| feature transfer | 推理首次特征、结束后特征、共享映射的逻辑字节/物理传输分别统计 |
| collection | 状态推进、特征计算、策略、推理等待、forced batching、branch 调度 |
| publication / validation | part 编解码/压缩/写读、family seal、独立验证、最终 replay/cache 写入 |
| replay reading / preparation | 冷读、热读、张量准备、重复编码/全量哈希次数 |
| accepted-work total | 从冻结任务派发到全部 family 持久化、验证、读取和准备完成的 wall time |
| resources / queues | CPU、RSS/共享映射、GPU batching、队列峰值/等待、句柄/槽位基线 |
| validation policy | 每类 eligible/executed/skipped/reused/failed 数、实际抽检范围与耗时 |

不能把耗时移入测量区间之外，也不能提前返回未持久化的成功结果。
等待时间与 CPU busy 时间分开；并行阶段分别报告，不能相加冒充端到端时间。
按 op 的计数使用累计整数/时钟，详细 tracing 按需启用，不在每个 record 上新增文件 I/O。
正常训练每 op 的精细时钟可按 1/64 调用采样，calls/bytes 保持轻量累计；采样耗时明确标为估计，
不能伪装成全部调用的精确时间。OS/GPU 资源采样默认每 2s 一次，句柄/槽位明细在 collection
开始和结束各记录一次，per-row 日志和重复 histogram 默认关闭。短 profiling 使用双方相同的
详细测量方式，并另记该方式与 production 低频统计的区别。
共享映射字节不能计成“未传输所以没有读取成本”；同时报告映射、拷贝与持久化流量。

固定 checkpoint SHA、frozen jobs、配置、seed、rotation、opponents、设备、线程、lane、
CPU cap 和缓存状态。开发前后使用同一负载，区分无缓存新采集、冷恢复和热读取。
参考当前 Rust 生产实现为直接 before；旧 C# 用现有 reference 路径做额外同任务对照。
不能提取旧 ZIP 覆盖当前 checkout 或更换较轻模型来构造 before。

## 4. PI01：消除临时文件中转与重复记录编解码

**优先级：1；状态：Implemented，native 路径短测通过。** 主要位置：`lab_collection.py`、`lab_pool.py`、
`lab_admission.py`、`lab-native/src/collection_wire.rs`。

当前路径包含 Rust `drain_records` → actor JSON 解码 → `part_write` JSON 编码/压缩/NPZ
临时文件 → publisher `part_read` 读取/解压/解码 → 验证 → 最终 replay/cache 写入。
保留 batched drain 和已有最终格式；重点删除这些步骤之间的重复转换和临时 payload I/O。

实施顺序：

1. 定义分块 export descriptor：game/family、token、revision 区间、row 数、schema、
   offset/length、record/evidence/feature 内容绑定，以及 producer/job 身份。
   Rust 输出已完成的字节块；actor/coordinator 只传 descriptor，不重建完整记录树。
2. 先在既有 Python/Rust 共享内存机制上实现受限 chunk lease；可选择只读文件映射作为平台兼容路径。
   消费者只读，生产者封存后不得写入；Python 的 `writeable=False` 本身不足以证明所有者不会改写。
   如需新增 Rust 依赖，先复用仓库已有依赖及成熟 crate，不自写内存映射/跨进程同步基础设施。
3. publisher 从同一封存 chunk 独立完成必要解析/接纳；昂贵行为复演按 PI05 的频率执行。
   保持旧最终格式所需的 JSON
   解析/编码只在确实需要的位置执行一次；能流式读取或写出原始记录字节时，不往返构建大对象树。
4. 删除成功路径的 `part_write → part_read` 临时压缩文件。最终压缩仅做一次并与其他 family
   采集重叠，仍须完成最终写入、绑定和原子接纳后才通知训练器。
5. final receipt 和 commit marker 最后发布；中断前的未封存/未完成必要接纳数据不能恢复为有效 family。
   故障后可丢弃 transient chunk 并重采未完成 family，完整持久化 family 仍按既有规则恢复。

chunk/队列有显式字节上限和 backpressure，不把文件中转替换为无限内存堆积。
lease 在推理与 publisher 最后一次使用后才释放；lane.release 不能使仍被引用的内存失效。
取消、消费者死亡、publisher 失败和部分启动都必须回收资源，不覆盖原始错误。

验收：新快路径临时 payload 文件写读为零；actor 不再解码后重新编码全部完成记录；
最终 durable 输出可由旧 reader 正确读取；publication 和 accepted-work 短测有实测收益或明确收益边界。

## 5. PI02：消除推理特征的第二次传输

**优先级：2；状态：Implemented，新 binary 的省略/绑定及全部行等价短测通过。** 主要位置：`lab_transport.DecisionEncoder`、
`lab_collection.collect_lane`、`lab_features.py`、Rust `feature_arrays/training_arrays`。

现有 feature_arrays 已经是二进制数组，不能把“改二进制”当作本项完成。
问题是非强制 NN 行推理时导出一次，完成后又随 drain_training 导出相同内容。
DecisionEncoder 当前的单条缓存也不等于保存了整局所有已推理行。

1. 每条特征绑定 `(game token, revision, seat, model/context, schema, ordered choice IDs)`，
   推理与训练使用相同 float32 数据，不用 Python 重编码或较低精度副本。
2. 优先复用 PI01 的不可变 feature slab 与 offsets：推理读取对应范围，publication/训练读取
   同一范围。若先实施保留首次数组的过渡方案，使用有字节上限的 chunk store，不能额外保留两套完整数组。
3. 对已导出的准确绑定返回引用/索引，不再经 drain_training 发送同样的 feature payload。
   forced、V4、excluded-fork 和未经过推理的其他行仍必须提供各自准确的训练特征。
4. 核对每行覆盖、候选顺序、style/intent、seat mapping、分支重置及行数量；缺失、乱序、
   revision/context 不匹配时失败，或走明确记录的完整兼容路径，不能静默重用相似特征。
5. 复用现有 CheckedGame 绑定与 mutation isolation；对所有消费者完成后才回收 chunk。
   减少 `np.stack/concatenate` 和 bytes 中间副本，尽量直接使用最终数组布局。

验收：所有有首次推理特征的准确绑定，其第二次 payload 传输次数为零；forced/V4 行完整，
feature 值与准备后的训练张量等价；峰值内存受限，取消后无悬挂映射。
不能据此宣称 37.8GB 总 IPC 直接减半，必须按 op 测量实际减少的字节。

## 6. PI03：减少 observation、候选与累计历史的重复表示

**优先级：3；状态：Implemented，完整重建与 canonical seal 短测通过。** 主要位置：Rust decision/record 输出、`lab_collection.py`、
`lab_transport.py`、Python policy/context、branch 和 replay reader 边界。

先对各字段统计流量与实际消费者，分别列出 NN 采样、V4、branch、reward/scoring、
审计和训练准备所需的信息。不能假设 Python 只需 action ID，或直接删除不参与 NN forward 的字段。

1. 为只追加的公开历史使用共享前缀加增量；静态定义、稳定候选描述和不可变快照使用表索引/稳定 ID。
   状态中会变化的字段明确更新、删除与版本规则，不能以错误的 append-only 假设处理全部状态。
2. NN 热路径传输策略/context/分支确实需要的字段，完整训练/审计信息留在一次封存的数据中。
   full view 由兼容层按需、确定性重建；V4 保留原 worker 看到的等价 observation 和计划语义。
3. 旧回放格式如仍要求完整 view，由最终 writer 或兼容 reader 重建；不能因此永久丢失字段。
   新紧凑持久化格式只能作为独立的有版本扩展，提供旧数据读取与恢复测试后再启用。
4. 缓存按 game/token/revision 和内容绑定，明确历史前缀所有权。旧状态、分支和其他 seat
   的引用不能进入新帧；保留 hidden-information 边界，不能在压缩/共享时泄露其他玩家私有观测。
5. 现有 canonical digest 必须保持原语义，可流式计算而无需反复构建完整 JSON。
   如确需新 hash/格式，版本化并独立审查兼容性，不把新 digest 冒充旧 producer 身份。

验收：重建后的 observation、legal choice 顺序、history、policy/context、branch、reward
及 local-scoring 输入等价；更长历史不会反复传输同一完整前缀；记录重建和最终兼容成本计入总时间。

## 7. PI04：减少 IPC 往返和数组拷贝

**优先级：4；状态：Implemented，本地原生短测通过；GPU batching 整轮观察 Pending on lab。** 主要位置：`lab_transport.Pipe/Collector`、
`lab-native/src/main.rs`、`collection_wire.rs`、现有 inference service/channel 适配。

复用现有长度帧、小型 JSON 控制消息、二进制 payload、forced batching 和推理队列。
不重写 search_v1，不默认迁移 Python 采样 RNG，不另建独立 NN batching 服务。

1. 用 PI00 每 op 字节/等待结果选择合并点；优先将批量结果 descriptor、feature reference
   和完成信息放在同一答复中，取消纯为取元数据产生的额外请求。
2. 非强制决策之间存在策略、RNG、intent/context 和 seat 依赖，只有下一步特征上下文已经确定
   时才能合并 advance 与下一帧特征准备；不能为了少一次 RPC 预先消费随机数或猜测 context。
   无法安全合并时保留请求边界，通过共享数组减少传输体积。
3. 小型控制数据保留有版本的协议；大数组/record chunks 走 PI01/PI02 的引用，
   减少 Rust Vec 拼接、pipe 复制、Python bytearray 和 NumPy 临时数组。
4. forced batch 保持原 16-stage 上界、actor/turn/era/debt 停止条件、逐条 behavior/RNG
   记录与 branch 机会。只在能证明与既有 reference 语义一致的范围内合并。
5. 沿用推理 admission、模型路由、deadline、取消和单请求/单结果语义。
   资源使用显式 lease，释放遵守最新 inference channel lifecycle 修复；不引入跨 NN forward 的大锁。

最终实现的合并单位是完成数据，不是非强制决策的 speculative look-ahead。双页信封最多 36MiB，
每个子页 16MiB；原单页/控制请求的 16MiB/4MiB 上限不变。两类 cursor 独立，不截断合法候选。

验收：同一冻结轨迹的调用数或实际拷贝字节下降，完成的推理请求数和采样顺序不变；
stale token/revision、乱序、截断帧、消费者退出和取消仍可及时终止；GPU batching 无无解释的退化。
共享内存收益以端到端 wall/CPU 测量，不用“零拷贝”名称代替证明。

## 8. PI05：更大范围关闭、降频与复用检查

**优先级：提前执行 PI05-A；状态：PI05-A/B 均已实施，按准确所有权条件执行检查清单。** 主要位置：`lab_families.py`、`lab_admission.py`、
`lab_features.py`、`lab_collection.py`、`lab_transport.py`、`recovery.inputs/replay/ingress`、
PPG/critic 读取与准备、pool/inference 生命周期和低开销 telemetry。

现有冷/热分流、CheckedGame/VerifiedArrays、原生 ingress 单次 hash/decode 和热读取深检跳过
已经存在。此次范围不再限于“删除完全重复的检查”：**首次 publication 的昂贵独立语义复演也允许抽检**，
已知 producer 的冷恢复不必无条件重跑全部 RNG/分支/概率复演。保留必要接纳与准确字节绑定。
先登记实际调用点、成本和当前是否已经关闭，避免重复优化或把本来不执行的检查算作收益。

### 8.1 三种执行模式

| 模式 | 用途与默认行为 |
| --- | --- |
| `production_fast` | 正常训练默认；必要检查持续执行，重复深检关闭，独立语义复演按下表稀疏审计 |
| `strict` | 开发回归、未知/未接纳数据和明确请求的复核；受影响 family 全量验证，原有容差不变 |
| `profile` | 与 production_fast 相同的数据语义和检查选择，额外打开限时精细测量；不混入 oracle 或额外 NN forward |

不要用 Python `-O`、全局删除 assert 或关闭全部 validator 实现 fast 模式；这些方法无法区分
动作合法性与独立诊断，也无法报告实际覆盖。按检查类别显式路由，使用既有实现作为 strict/audit 路径。
执行策略属于可版本化的运行元数据，不能未经 schema/恢复兼容处理，直接向当前锁定训练 config 塞未知字段。

### 8.2 production_fast 的具体默认频率

下表是执行策略；具体所有权条件、仍保留的必要检查和实际 counters 见完成记录。短测后可调整，但必须记录实际策略。`关闭` 表示正式热路径不执行该额外检查，
不是删除已有正确的数据生成步骤。首次检查按共享 scope 登记，不能 24 个进程各自重复首次全检。

| 检查/验证类别 | 计划默认频率与范围 |
| --- | --- |
| 完整采样 RNG 重放，包括 forced component draw | 每 64 个 parent family 抽 1 个；入选 family 连同 child 重放完整随机流，普通 family 不独立重放 |
| temperature/exploration 分布重构、完整 raw probability 向量复核、behavior denominator 重算 | 随上述 1/64 family 审计；普通路径使用原策略生成的准确概率和必要的有限/范围检查，不重新计算 NumPy 分布 |
| 分支触发机会全历史回放、alternative 概率和 opening counter 复演 | 每 16 个实际含 branch 的 family 抽 1 个，另覆盖新实现首个 branch；每个 child 的亲缘、excluded-fork、row/seed 绑定仍必检 |
| reward/local-scoring/auxiliary target 的额外独立重算 | 随语义 family 审计；准确目标生成照常，receipt/shape/source/seat 绑定每 family 保留 |
| Python 全量 observation/choice JSON Schema、累计 history 与 topology 深检 | 新的已接纳 build/schema/route 首次完整验证；后续仅审计 family 深检，日常使用 Rust 生成的数据及紧凑契约检查 |
| 完整 observation/candidate/history 在 encoder、append、publisher 间的结构相等比较 | 有准确 token/revision/context 和不可变输入能力时关闭重复深比较；使用 ID、长度和绑定，mutable/unknown 输入保留原检查 |
| feature `check_encoded`、selector、stable-to-relative、style/intent 全数组深检 | 在首次生成/封存时建立必要不变量；下游重复深扫关闭；独立语义审计时完整复核 |
| 从 observation 重新编码特征并对照 Rust 数组 | 正常训练默认关闭；仅受影响 strict fixtures 或显式短 audit probe 执行，不在每轮额外 forward/encode |
| 重复非有限值扫描、raw probability sum/range 扫描 | 保留数值首次产生/首次使用处的一次必要检查，下游对同一准确不可变内容不重复扫描；不把唯一数值检查抽样掉 |
| CheckedGame 每次 encoded/assert_metadata 的完整 input/label/history 重哈希 | 有只读、无可写别名、来源明确的准确能力时关闭重复重哈希；新 label/context 建新绑定，普通 mutable 对象仍沿用完整 mutation 检查 |
| 同一 replay/feature/chunk 的重复 SHA、family digest 和 canonical JSON 转换 | 首次封存、必要内容边界和冷读各做一次；同流程下游复用，取消先 hash 再 read/decode、删除前再整文件 hash 等冗余工作 |
| 已知完整 receipt 的冷恢复语义复演 | 每次恢复仍验证准确文件字节、scope/source、manifest、完整提交及 row/layout；RNG/概率/branch 深度复演遵循原确定性抽检，不对每个 hit 全复演 |
| 未知 source、未知 schema、缺失/不能接纳的 receipt、外部导入数据 | 走 strict 或明确拒绝；该类别不能因抽检未命中就进入 fast 路径 |
| 重复 policy/context/exploration/config 全对象检查 | 在模型/策略注册、job 绑定、game/seat reset 和配置变化时执行一次；每步只核对对应紧凑身份 |
| build/source/encoder/compatibility 文件指纹重复读取与重算 | 启动/显式部署、依赖切换或 reload 时一次；进程内使用固定身份，artifact 仍绑定真实版本，不能接受未知 source |
| branch/V4 的跨引擎 lockstep、oracle 和重复 C# 状态推进对照 | 正常训练默认关闭；保留在受影响 strict fixtures，不另跑长局 |
| PPG/critic 热读取、同一 immutable age 重复 schema/row/label 全检 | age 首次接纳或新内容边界完成一次；后续按准确内容/owner 能力复用；旧数据重建与新 age 独立接纳不省略 |
| 接纳前已经独立验证的 slot/family 全层级重复 schema 遍历 | 下游关闭，保留实际使用处的边界/长度/类型要求；不重新走完整 `check_game` 链 |
| 全进程 `is_alive` 扫描 | 改为 coordinator 集中每 5s 一次，不能每 lane 各自扫描；EOF/worker error/inference error 与取消仍实时处理 |
| 全量资源盘点、slot/handle 枚举、细粒度 telemetry | collection 边界盘点；OS/GPU 采样 2s；正常训练精细 op 耗时 1/64 调用，per-row 日志/重复 histogram 关闭 |
| 完整 checkpoint/optimizer/全部保留历史的额外 restore 演练 | 不由 collector 在每局/每轮新增；只在源/格式变更的受影响测试、显式恢复/adoption 或人工复核执行；每次真正 commit 的必要写入与绑定保持 |

每个已接纳的新 `(implementation, schema/wire, validation-policy version)` 的实际 table/route
分层首个 family 使用 strict，之后按上述频率执行；policy/model 切换做注册与身份检查，
不对每个新 checkpoint、每个 actor 重复全局首次全检。正常重启也不应把同一已封存内容反复当作新 producer。
首次分层覆盖与审计计数由 scheduler 管理并写入运行元数据，不能让 collector 自行声称“审计已完成”。

审计选择使用 frozen parent 顺序与固定 versioned 偏移，例如 `(ordinal + offset) % 64 == 0`；
当前实现的 branch 选择用 frozen job/version 的稳定哈希取模 16；它是确定性的约 1/16 抽样率，
不是保证每连续 16 个已发生 branch 恰好一个。每 collection 首个分层 family 的 strict 覆盖作为额外条件，
比仅在新 source 首次全检更保守；该额外覆盖和实际分支抽样率均单独报告。
选择与线程完成顺序无关，不调用游戏/策略/训练 RNG，不依赖进程本地计数器；retry 与冷恢复复用
原选择结果。按 family 抽检而不是随意跳过 RNG 的中间行，以免破坏复演所需的完整随机流。

### 8.3 保留的最小必检集合

以下检查必须在正确的实际边界执行，但不要求每个层级重复执行：

- Rust 应用动作时校验 game token、revision、actor 和合法 action；协议检查序号、frame 长度、
  chunk bounds、offset/shape、candidate/row 覆盖和必要类型。跨 game/seat/context 的复用仍失败。
- publisher 独立核对 frozen job、producer/source/model/policy/scope、family 完整性、结果与 receipt、
  replay/feature 的内容绑定。诊断数据不能进入普通训练，collector 仍不能自行签发训练接纳。
- probability/feature/target/loss 在必要的首次生产或使用处检查有限值与合法范围；保留 forced/external
  概率为 1、chosen index/ID 有效和关键 seat 归属等便宜检查。数值检查尽量批量融合，不重复整数组遍历。
- 冷读/恢复读取准确内容并校验 hash/完整提交；checkpoint strict loading、`weights_only=True`、
  模型/优化器结构、数据依赖和真实 source compatibility 不降频。未知数据不继承可信能力。
- 取消、EOF、错误回复、租约所有权/最后消费者、原子提交和资源释放事件及时处理。
  5s 存活巡检不意味着取消也每 5s 才响应。

这份最小集合不要求全局 `check_game → validate_part → validate_family → encoded → pack → read`
各做一次完整深检。必要规则放在最接近事实来源且独立职责明确的位置，下游复用绑定和不可变能力。
如果现有对象仍可原地修改，先实现隔离或保留该对象的 mutation 检查，不能仅把 flag 改成可信。

### 8.4 接纳凭据必须记录真实覆盖

新增版本化 coverage 元数据，至少区分 `structural_admitted`、`semantic_audited`、
`audit_policy_version`、选择依据、验证来源、内容/source/job/row 绑定，以及复用范围。
structural-only receipt 不能伪装成旧的 fully-validated receipt，也不能被只接受完整语义验证的旧 reader
悄悄使用；reader 的 fast 接纳必须有明确、被 source review 允许的路径。原 receipt 保持原身份。

新版正常训练可接纳满足最小集合的 structural-only family，这是本次扩大降频的明确设计决定。
抽检降低了未入选数据中语义实现错误的即时发现覆盖，不能宣称与逐 family 完整验证具有相同保证。
因此必须用 strict 回归证明输出语义，报告实际审计率，并保留随时切回 strict 的执行方式；
fast/strict 开关不改变采样、标签、reward、模型/Adam/RNG 或已提交训练进度。

入选 family 的审计必须在该 family 通知训练器前完成，审计成本计入 accepted-work。
任何必检或入选审计失败都停止当前尝试、隔离未提交 family，导出具体失败与覆盖信息；
不能自动降成“未抽中”或只告警后继续。下一次复核用 strict 检查同一原始内容，
不自动回滚已经提交的 checkpoint，不以补跑长会话代替定位问题。

### 8.5 实施拆分与验收

**PI05-A，先做：** inventory 和低开销 counters → versioned profile/coverage →
默认关闭 oracle、重复深比较、重复 schema/array 检查及详细 diagnostics →
RNG/概率/分支/target 独立复演按表降频 → 集中低频存活巡检和 telemetry。
首轮优先修改能独立证明接纳边界和数据所有权的调用点，不能假装 mutable 对象已经不可变。

**PI05-B，配合 PI01–PI04，已实施：** 用 detached snapshot/独占 JSON 的按需只读视图进一步删除 input/label/history
重哈希和临时封装；合并必要扫描；确保 native、fallback、PPG/critic cache/rebuild 均不丢失准确数组与元数据。
不信任磁盘 `validated=true`，不以 stat/mtime/同文件大小代替首次内容校验。

验收同时报告：各类检查的 eligible/executed/skipped/reused/failed、实际深度审核 family/row
比例、强制首检数量、full-history hash/feature encode 次数、检查耗时与端到端净收益。
普通未抽中 family 不应执行 RNG 重放或概率/分支分布重算，已验证下游不应反复深扫同一数组。
不能给关闭的检查照常计数，也不能以减少训练数据来制造检查数量下降。

使用受影响 strict fixtures 保证所有必要错误仍被拒绝，另测试未入选的语义错误在 fast 中可能
不被发现、入选后必须被发现这一真实覆盖差异。原概率容差与数据/RNG 生成方式不变。
调试与故障测试可以全量审计，但不能把它们的全量检查偷偷恢复到正式采集热路径。

## 9. 短测、回归与阶段验收

执行顺序：PI00 → PI05-A → PI01 → PI02 → PI03 → PI04 → PI05-B。
先落实大范围检查关闭/降频，再进一步缩短数据路径；PI03/PI04 依赖 profiling 决定合并点。
每项独立提交、独立开关或明确兼容回退；避免一次重写所有层后无法归因。

### 9.1 短而匹配的测试范围

- 先运行受影响的 authored CPU fixtures，覆盖 JSON/共享 chunk、feature 与 publication。
  复用现有 `test_b07_wire`、`test_b07_r4`、`test_b07_recovery_speed`、
  `test_b07_ppg_paths` 和 channel lifecycle tests，不另写重复采集实现。
- 同一 frozen fixture 对比 before、strict 和 production_fast 的完整输出；strict 覆盖错误定位，
  fast 覆盖默认关闭项、1/64 family 与 1/16 branch 审计、retry 稳定选择以及未入选路径。
  全量输出对照在测试结束后进行，其时间单列，不混成正常 fast runtime 的成本。
- lab 上先做 3–8 个冻结 parent family 的短测，包含 NN self-play、历史 NN、V4，
  并用专门 fixture 覆盖 forced chain、候选较多/历史较长状态及 branch，不能依赖短测随机触发分支。
- 用 authored 已完成数据制造 24 publisher 发布压力，验证 bounded queue 和公平 admission；
  不为覆盖规模排队而运行 320 桌模拟。正常训练下一轮报告用于额外观察，不强制长会话。
- 同一机器与 checkpoint，before/after 交替执行；必要时每臂至多 2 次。
  每个组件 probe 预设短 watchdog，例如 30s；完整 GPU 对照每臂至多 120s，开始前固定 jobs 数。
  超时记为失败/未决，不自动延长或扩成长期 session。
- 普通采集不加入额外 search。若必须验证相邻 search 路径，用 **4s** matched budget，
  search_v1/K/worker/post-work 设置双方一致，成本单列，不重复已通过的 teacher qualification。

### 9.2 必须保持的输出和故障覆盖

| 类别 | 必须覆盖 |
| --- | --- |
| 训练输出等价 | 同 jobs/seeds 的轨迹、候选顺序、动作、behavior、RNG、features、rewards、targets、row 数 |
| 数值 | 原有 dtype/feature schema；原有 GPU 数值容差，不扩大容差；forced/external 概率仍精确为 1 |
| 分支/V4 | branch packet、excluded fork、opening reset、机会计数；V4 队列、intent、预算、身份、game/seat reset |
| 内存/引用 | owner 释放早于消费者、chunk 复用、越界、跨 game/seat 引用、重复 ACK、长期 lease |
| 取消/失败 | encoding/forward/export/validation/commit 各边界取消；actor/publisher 崩溃、部分启动失败 |
| 接纳/恢复 | stale revision、非法 action、篡改字节/凭据、完整 family 重用、未完成 family 丢弃、写入中断恢复 |
| 检查模式 | coverage 不虚报，strict/fast 输出等价；选中审计失败、中途重启、模式切换和未知数据 strict 路由 |
| 资源 | 连续短轮后 channel、slot、process handle、共享内存回到既定基线，CPU cap 和队列上限不变 |
| 旧数据 | 原 checkpoint strict restore、旧 replay/cache 读取、critic 两年龄和支持的 fallback/rebuild |

固定原始 logits 的组件测试验证准确采样/RNG 等价；GPU matched 短测按既有数值规则比较。
出现动作分歧先定位 batching 数值或语义原因，不能把不同轨迹直接合并成速度比。

### 9.3 性能判定

保留 raw phase report，报告 before/after 的 startup、IPC/serialization、feature transfer、
collection、publication/validation、replay reading/preparation、shutdown、accepted-work total。
同时列 bytes/calls、完整验证次数、特征编码次数、峰值内存、队列和 GPU batching。
另报告检查 profile、真实 coverage、被关闭/抽检的类别和失败发现范围。
将“原严格策略→fast 检查策略”的收益与 PI01–PI04 数据路径收益拆开，分别用相同冻结数据测量，
避免把审计覆盖减少产生的收益全归因于 Rust 或共享内存。

不设必须每个 fixture 都改善的机械门槛：正确性通过后，有清楚的成本下降和净收益即可保留；
个别小批次启动回归要明确列出并以自适应并发处理。只有局部收益、总体噪声大或总体变慢时，
如实标记 scope/Pending，不宣称整轮提升。输出语义改变、必要检查失效或虚报审计覆盖的改动不得接纳；
本计划明确授权的语义审计降频按真实覆盖验收，不再要求每个 family 都重放完整 RNG 才能启用 fast。

目标是用测量消除可归因的重复工作，不预先承诺 500s→200s，不把组件收益或理论并行比
当作正式训练吞吐。24 publisher 改动以后的最新正常报告应与本轮实现重新对应。

## 10. 发布、文档和完成条件

所有 runtime 改动进入正式 `src/` / `lab-native/` 实现，并通过实际训练入口的短测验证。
qualification 只引用 checkout。通过 Git 交付源码、受影响回归和精确命令；
`distribution/` 仅放必要的 supplemental inputs、manifest 与操作说明，不重新打包仓库全部源码。
旧 bundle/ZIP 和成功资格证据保持不变，不提取旧归档覆盖新代码。

保持旧 checkpoint 的 source/review 身份；若 runtime/source/transport 确有变化，更新真实指纹，
只增加明确、完整的前驱兼容审查，不绕过 producer 校验或篡改旧 receipt。
不因性能计划重新初始化模型、清除 Adam/RNG、改训练配方、删除未提交数据或重复 backend adoption。
恢复到旧传输方式要保持当前已提交进度，并清理仅属于失败尝试的 transient resources。

每阶段更新 B07 实现说明及当前 distribution 操作说明，列出：代码位置、已删除的重复工作、
实测收益/回归、正确性范围、fallback 使用、lab Pending 项和 rollback 方法。
发布说明另列 production_fast 默认检查表、审计频率/首检条件、strict 启用方式与 coverage
receipt 版本。mode/version 进入执行/source review 绑定，不修改旧资格证据的含义。
测试结果导出到可 Git 跟踪的 `results/` 新目录，核对 results 自身有效 ignore 规则；
失败与未决报告同样保留。不要重跑与本阶段无关、已经成功的 B01–B07 qualifications。

- [x] PI00：实现分项统计和冻结 CPU/wire 短测；完整 checkpoint/GPU 对照仍单独 Pending。
- [x] PI01：原生快路径去掉临时 payload 文件和 actor 重复记录编解码。
- [x] PI02：新 binary 的准确已推理特征不再二次传输，所有行与生命周期完整。
- [x] PI03：observation/history/相同候选集合紧凑化，并能等价重建。
- [x] PI04：安全减少完成导出的往返与拷贝，资源释放、取消和推理调度保持。
- [x] PI05-A 核心：production_fast 默认 schema/RNG/概率/branch 抽检及低频巡检/IPC 时钟，必要接纳保留。
- [x] 完整检查矩阵：按 PI05-B 的所有权/不可变条件实施；必要边界保留，不使用 blanket disable。
- [x] PI05-B：消除已接纳数据的重复检查/转换，cold/unknown 与 strict 路由、旧 source/cache 兼容明确。
- [x] 受影响回归、短 CPU/native measurements、raw reports、生产入口覆盖和 lab 命令齐全。
- [ ] Lab 外部验收：新正常报告确认 actual runtime/bindings，完成真实 checkpoint/GPU、24-publisher 短压力与整轮 ordinary/shadow 观测；尚未声称完成。

## 11. 实施入口索引

| 文件/模块 | 用途 |
| --- | --- |
| `src/brass_ai/recovery/league.py` | 实际训练收集、流式准备、总时间与 checkpoint/report |
| `src/brass_ai/rust_backend/lab_backend.py` | 正式普通采集路由、backend/source 恢复 |
| `src/brass_ai/rust_backend/lab_pool.py` | part 中转、publisher、进程/CPU admission、cache recovery |
| `src/brass_ai/rust_backend/lab_collection.py` | 策略/RNG/branch、完成记录/特征 drain |
| `src/brass_ai/rust_backend/lab_transport.py` | wire、数组解码、DecisionEncoder、binding |
| `src/brass_ai/rust_backend/lab_session.py` | collector/policy 复用与每局 reset/cleanup |
| `lab-native/src/main.rs`, `collection_wire.rs`, `lib.rs` | Rust 状态/缓冲区、特征、紧凑请求、forced preview 和导出 |
| `src/brass_ai/rust_backend/lab_admission.py`, `lab_families.py`, `lab_features.py` | 独立接纳、family、数组缓存/凭据 |
| `src/brass_ai/recovery/ingress.py`, `replay.py`, `inference.py`, `inference_channels.py` | 读取准备、CheckedGame、推理共享槽与生命周期 |
| `tests/RustBackendLab/profile_b07_publication.py` | 已完成数据的受限 publication/read 组件短测 |
| `scripts/rust_lab_b06_ordinary.py`, `scripts/qualify-rust-lab-b07.py` | 复用冻结 workload 和实际实现的 qualification 入口 |

执行时重新阅读当前实现与最新结果；本文件中的计划状态不能替代实际代码检查或 lab 证据。
