# Rust collector performance: PI00–PI03 implementation

后半部分现已完成；本文件保留第一阶段原始范围/测量，当前实现与完整续训命令见
[PI04–PI05 完成记录](RUST_COLLECTOR_PERFORMANCE_PI04_PI05.md)。

2026-11-07。本次实现用于正常 Rust 普通采集，保留当前 teacher shadow：28+4 tables、branching=0、
share target 0.25、诊断梯度与原固定 beta 资格流程。没有修改 search_v1、teacher Rust 算法、
模型/optimizer/RNG、reward、feature 精度或训练数据配方。

## 已接入的生产路径

`league → RoutedPool → NativePool → collect_lane → Export/shared chunks → Publisher → admission → ingress/PPG`。
测试调用这些实际模块；没有另做一套 qualification 专用的快速收集器。

- **PI00：** Pipe 按 op 保留全部调用/发送/接收字节数。正常训练 IPC 精细时钟每 64 次调用采样，
  key 明确写作 sampled_seconds，绝不当成完整总时间；strict/profile 模式完整计时。
  collector 分别报告 actor/publisher startup、cache recovery、live、shutdown、完整 collector wall、
  actual runtime、共享内存峰值/未释放量及实际 publication 检查覆盖。现有 learner 时钟继续计入 preparation。
- **PI01：** actor 不再解析完成记录的 JSON，不生成临时 gzip/NPZ。parent 按需分配 16MiB shared chunks，
  总计最多 1GiB，单 family 最大 512MiB；actor 关闭写入映射后发送封存 descriptor。
  publisher 独立校验、一次拷贝为不可变 bytes、核对 Rust canonical seals、唯一解码完整记录并写最终文件。
  共享数据不经过 learner；parent 在 publisher 成功/失败后释放，pool 失败退出也能清理已登记分配。
  预算等待不持有 CPU slot，实际填充仍走原 CPU admission，避免 publisher 被阻塞 actor 饿死。
- **PI02：** DecisionEncoder 保留首次推理的准确 feature bytes，Rust 对请求省略的行重核 SHA，
  drain_training_reuse 只发送未持有的行。forced/V4/excluded-fork 特征仍生成、完整绑定和输出。
  所有行保持 float32 与相同候选顺序，token/context/行覆盖错误不能借助复用绕过。
- **PI03：** live frame 对公共 history 传前缀长度+增量，对相同静态 view 字段和准确相同的候选集合传引用。
  原始记录用 serde_json RawValue 定位 history 字节，传 lossless byte-prefix patch，publisher 重建原始字节
  后核对原始 Rust SHA；不依赖 Python/Rust 浮点 JSON 重新编码一致性。
  分支、新 game 和 release 重置状态；共享候选的 root admission 独立，旧请求不会被后续帧替换能力。
  所有最终回放仍保留完整 observation/history，旧读取格式未删除。
- **PI05-A 核心：** production_fast 使用 frozen ordinal 每 64 family 抽一个完整 RNG/概率复演，
  每 collection 每种 table/policy 分层首个 family 额外全检；branch 按稳定 hash 约 1/16 复核。
  未抽中数据仍做独立结构、身份、内容、数值和特征绑定接纳；receipt v2 真实声明 structural/semantic coverage。
  strict 可重新完整审查 structural-only receipt，旧 v1 与旧 producer 完整冷读仍支持。
  集中健康巡检每 5s 一次，取消、EOF 和错误回复仍及时响应。

第一阶段结束时尚未实施的 PI04 请求合并/pipe 拷贝优化、PI05-B per-access history/label 与 pack/read
复用现已在后续交付完成。mutable/fallback 的必要边界仍保留，不能称为关闭全部验证或零拷贝。

## 短测证据与限制

原始记录在 [distribution supplement](../distribution/rust-collector-performance/README.md)。
没有运行本地完整游戏会话、NN checkpoint benchmark、PPO 更新或长训练。原生 wire 短测使用固定状态、
确定性概率和真实 Rust/Python 采样与编码；发布短测使用冻结 authored families，模型/checkpoint 为 none。

两轮发布组件对照采用正序和逆序，同一 4 family / 256 decisions / 两个真实 publisher 进程。
双方均包含 startup、handoff、publication、first read、实际 training preparation 与 shutdown。
额外结果等价比较单列；固定数据构造在计时外，两边相同。

| 阶段 | legacy strict | shared strict | shared fast |
| --- | ---: | ---: | ---: |
| handoff 正序 | 0.574s | 0.027s | 0.022s |
| handoff 逆序 | 0.664s | 0.027s | 0.019s |
| publication 正序 | 1.082s | 1.059s | 0.895s |
| publication 逆序 | 1.048s | 1.084s | 0.913s |
| accepted-work 含 startup 正序 | 10.581s | 9.757s | 9.959s |
| accepted-work 含 startup 逆序 | 10.833s | 10.402s | 10.807s |
| 临时 payload 写入，每轮 | 4,586,236 bytes | 0 | 0 |

fast handoff 中位耗时约下降 96.7%，publication 中位约下降 15.2%；去除单列的进程 startup 后，
这两个组件 probe 的 accepted work 中位约下降 23%。包含 Windows startup 的总中位收益仅约 3%，
并且 shared strict 的小批次总时间比 shared fast 更好；这些差异与原始 first-read/startup 值一并保留。
不能把其中任一组件百分比外推成 320 family 或含 shadow teacher 的训练吞吐。

CPU/wire 回归覆盖完整动作/候选/observations、RNG、特征、记录 SHA、branch 与 targets；
还覆盖故意未审计的有限概率语义错误不会被 falsely 标记为 fully validated、入选 strict 后拒绝，
mandatory identity/NaN/越界/篡改持续拒绝，以及跨 chunk、backpressure、取消、源/父 build 绑定与恢复。
已有 shadow 32-table/fixed-beta、checkpoint/source、PPG/replay、不可变输入回归只跑受影响部分。
完整 GPU/frozen-checkpoint 与正常 shadow 整轮收益保留 **Pending on lab**。
对应 lab 工具是 `scripts/profile-rust-collector-performance.py`，复用实际 frozen planner、
ActorPool、NativePool、B06 输出/target 比较和有界进程 watchdog；每臂最多三桌，输出自动导出到 results。
工具及其 snapshot/失败导出有本地 CPU 回归，真实 GPU 两臂运行仍留给 lab。

最终受影响回归为 **94 项 Python + 2 项 Rust 通过**，另复核了重复 decision 的准确候选复用。
源码审查可重建原 qualified source，实际执行与 qualified 身份分别记录，原 teacher producer 不变。
最终 native wire ABBA 短测中，423-candidate complex root 的 24 决策接收量为
8,198,846 → 7,640,445 bytes；late root 的两决策为约 367,477 → 273,264 bytes，
并省去 19,084 bytes 的重复训练特征。未完成的 complex 短轨迹不声称覆盖整局 feature drain。

lab 正常报告分别查看 `timings.collection_seconds`、`timings.shadow_collection_seconds`、
`timings.replay_load_seconds`、`timings.search_and_distill_seconds` 与
`native_backend.collection` 的 runtime/validation/shared/startup/shutdown 字段。
`inference` 是 invocation 累计值，不能直接当成单迭代 forward 时间。
actor 的原始 Export 在 handoff 后立即释放，已推理特征在 actor 内仅保留一份 bytes store，
最终导出使用引用，按原 Rust lane 的 256MiB 上限约束；Rust 到 Python 及 publisher detachment 的物理拷贝仍存在。

## 构建、恢复与 source 绑定

源码通过 Git 同步，新增 ordinary binary 由 `prepare-rust-collector-performance.py` 在 lab 的
新 artifacts 目录构建和短测。使用原封存 B01 core 与原 bootstrap；不覆盖旧普通/teacher 二进制或旧 ZIP。
新的 `artifacts/rust-backend/collector-performance/build.json` 是运行时选择凭据，绑定原 build SHA、
bootstrap、core、精确 ordinary source、测试和构建日志。没有此凭据时兼容旧 wire，并明确报告 legacy。

原 backend/qualification/checkpoint receipt 保留；runtime resolver 在实际 RoutedPool 入口选择新 build。
旧完整 family 只在明确的原 producer 绑定下读取，不能把旧记录改成新 producer。
同源码重复 prepare 复用已验证 runtime；新普通回放用真实的新 build SHA。

teacher producer 通过精确 reviewed source map 复用：允许变化的仅 ordinary main、新 transport 模块、
serde raw_value feature 与 producer-equivalence helper。teacher bin/module、overlay、shared lib/scoring
必须保持原 hash；额外变化仍产生未知 producer。review 文件本身进入完整 training execution fingerprint。
保留 g1257/shadow 前驱执行身份和原 checkpoint 元数据，下次提交记录真实新 execution。

构建成功和失败均导出新 `results/rust-lab-qualification/`，不需重跑通过的 B01–B07 gates。
回退时先退出当前 trainer，保留所有 run/checkpoint/cache，移开新的 runtime selector 文件，
使用原普通 binary 的兼容 wire；若要完整复核已接纳数据，退出 trainer 后将 shadow JSON 的
`execution.collector_validation` 改为 `strict`。正常训练的默认值已写入配置，不需 export。
strict 复核可能更慢，其覆盖与 fast 不同，不改变游戏采样或训练进度。
