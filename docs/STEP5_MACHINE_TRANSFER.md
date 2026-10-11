# Step 5：迁移到另一台 lab machine

不能只复制最后一个 `iteration-*` 文件夹或 `best.pt`。续训还需要最新 learner 的完整 checkpoint（含 Adam/RNG）、`state.json`、当前历史对手池、incumbent、evaluation anchor，以及 IL 来源记录。`best.pt` 可能比最新 learner 旧。

独立脚本 `scripts/step5-transfer.py` 默认使用 **compact + external + 100 MB 分卷**，适合目前的安排：新机器通过 Git clone/pull 获取当前代码，游戏规则运行时通过之前的 SSH zip 方法传输，其他训练文件由迁移分卷补齐。工具不改训练实现、签名、模型格式或超参数；本次工具更新不需要回退训练代码。

## 默认复制哪些文件

| 内容 | 默认处理 |
| --- | --- |
| `state.json`、精确续训 config、best 元数据、显式 pins | 保留 |
| 最新完整 checkpoint、state 指定的恢复 checkpoint | 保留原字节，包括 optimizer/RNG |
| 当前历史对手池、incumbent、evaluation anchor | 保留；不为缩小体积而缩减训练对手池 |
| 仍在磁盘、尚未由当前池代表的历史候选模型 | 保留，因为下一次更新对手池仍可能使用 |
| 最新 iteration 的 report、评估 report/spec、已选 champion 的证据 | 保留 |
| 未完成评估的缓存及其引用模型 | 保留，避免重新收集已经开始的评估 |
| IL 来源和祖先的 state/report、选中 checkpoint、dataset manifest | 保留；不复制无用的 epoch checkpoints |
| IL 训练数据分片 | 仅当 rehearsal 仍启用且未到 `rehearsal_until` 时复制训练分片；结束后省略 |
| 已消费的 PPO/teacher/已完成评估的 replay、feature caches | 省略，显式 pin 的文件除外 |
| 无引用旧模型、retention 历史和隔离区 | 省略 |
| 代码、仓库配置、冻结规则 AI 资料 | 从 Git 获取，分卷里只记录依赖哈希 |
| 游戏规则运行时/DLL | 从 SSH 获取，分卷里只记录依赖哈希，**不打包到 Git 分卷** |
| conda 环境、CUDA 驱动、其他 runs、旧机器 preflight 和进程锁 | 不复制 |

相同内容只压缩存储一次，例如完全相同的模型文件有多个路径时不会重复占用分卷空间。恢复时仍建立原来的文件布局。所有源文件原样保留，工具不会清理旧机器。实际体积取决于当前池和 rehearsal；`--dry-run` 显示逻辑大小、去重后的大小、类别大小及最大的十个文件。

## 1. 旧机器停止训练并导出

等待当前训练命令正常完成后执行。脚本拒绝活动 run、未提交 iteration/checkpoint、损坏或缺失的依赖；中断留下的半个 iteration 应先在旧机器用原命令恢复完成。

把 `RUN` 改成你实际的 self-play 目录，`BUNDLE` 放在该 run 和 IL run 之外：

```sh
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
RUN=$ROOT/artifacts/step5/selfplay-best-v1
BUNDLE=$ROOT/artifacts/step5-transfer/selfplay-migration.zip

# 先查看体积；不生成压缩包。
python "$ROOT/scripts/step5-transfer.py" export \
  --root "$ROOT" --run "$RUN" --bundle "$BUNDLE" --dry-run

# 以下均为默认设置，显式写出便于检查。
python "$ROOT/scripts/step5-transfer.py" export \
  --root "$ROOT" --run "$RUN" --bundle "$BUNDLE" \
  --mode compact --dependencies external --part-mb 100
```

生成 `selfplay-migration.zip.part0001`、`.part0002` 等分卷，以及 `selfplay-migration.zip.parts.json`。不会额外保留完整 zip。索引最后写入，包含分卷和完整归档的大小及 SHA-256。输出还有保存的 iteration、游戏数、Git commit、实际压缩字节数 `compressed_bytes` 和分卷数。每次导出使用新文件名，现存分卷不会被覆盖。

## 2. 用私有 GitHub 仓库传输分卷

GitHub 普通 Git push 拒绝超过 **100 MiB（104,857,600 字节）** 的单个文件；超过 50 MiB 会显示警告，但仍允许 push。默认每卷最多 **100 MB（100,000,000 字节，约 95.37 MiB）**，低于上限。此限制同样适用于私有仓库。[GitHub 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

单次 push 另有 **2 GiB** 上限，文件分卷不能消除这个总量限制。如果分卷较多，每批最多约 10 卷（约 1 GB），分别 commit 后立即 push；最后一批带上 `.parts.json`。不能先把全部大分卷放入一个超过限制的 commit，再指望多次 `git push` 自动拆开。[GitHub 单次 push 限制](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit)

`artifacts/` 默认被忽略，因此只强制添加**这次导出的精确分卷和索引**。以下适用于总量约 1 GB 以内、其他待推送内容较少的情况；更多分卷则分批选文件：

```sh
cd "$ROOT"
git add -f -- artifacts/step5-transfer/selfplay-migration.zip.part[0-9]* \
  artifacts/step5-transfer/selfplay-migration.zip.parts.json
git commit --only -m "Transfer saved Step 5 training state" -- \
  artifacts/step5-transfer/selfplay-migration.zip.part[0-9]* \
  artifacts/step5-transfer/selfplay-migration.zip.parts.json
git push
```

不需要放开整个 `artifacts/` 的 ignore，也不要添加游戏运行时 zip。默认 `--dependencies external` 的迁移分卷不含这些运行时。迁移脚本、说明和测试本身通过正常代码提交传到仓库；脚本不自动执行 Git add/commit/push。

## 3. 新机器 Git pull，然后恢复

使用与旧机器相同的绝对根目录 `/data/not_backed_up/yxu209/ai-training/`，并保留相同 run 相对路径。部分报告/配置含绝对路径，工具不会重写已签名的数据。游戏运行时先用已有 SSH zip 方法导入到原来的路径。

```sh
ROOT=/data/not_backed_up/yxu209/ai-training
git -C "$ROOT" pull --ff-only
BUNDLE=$ROOT/artifacts/step5-transfer/selfplay-migration.zip.parts.json

python "$ROOT/scripts/step5-transfer.py" restore --root "$ROOT" --bundle "$BUNDLE"
python "$ROOT/scripts/step5-transfer.py" verify --root "$ROOT" --bundle "$BUNDLE"
```

恢复直接读取分卷，不必手动合并，也不需要另存大 zip。它先检查所有分卷、归档内容和外部代码/运行时哈希，再写入训练文件。相同目标文件可复用；缺失外部依赖或不同内容会停止恢复，已有文件不会被覆盖。

**本次新代码只增加独立迁移工具，不改变训练签名。** 以后仍可正常 Git pull 更新；若某次更新实际改变了训练实现或规则，按那次更新的已有续训/迁移说明处理，不能手改 state 签名绕过检查。分卷记录导出时的 commit，便于核对版本。

`verify` 用于恢复后、开始续训前；开始训练后 state/checkpoint 改变是正常的，不应再用旧分卷验证正在更新的 run。

## 4. 新机器 preflight，再继续原 run

安装并激活 `ai-training` conda 环境，按[训练指南](TRAINING_RECOVERY_LAB.md#transfer-and-preflight)安装 PyTorch/CUDA、Python 依赖、Mono/OpenSpiel。迁移脚本本身只依赖 Python 标准库。

`restore` 和 `verify` 输出两条完整命令：

1. `preflight`：在新机器验证设备和规则运行时，不能沿用旧机器的通过记录。
2. `resume`：使用保存的 config、原 `--run` 和需要的 `--imitation-run`，继续 20 iterations。可修改追加轮数，不要重新传 `--initial`。

先执行 `preflight`，成功后执行 `resume`。已有 self-play state 会恢复 PPO checkpoint；`--imitation-run` 仅验证 IL 来源，不会重做 IL。课程表、历史池、teacher 预算和计数器保持原状态。

保留旧机器的 run，直到新机器成功续训。不同 GPU、库版本及搜索线程调度不保证后续结果逐步一致；本工具保证保存文件的字节完整性和恢复依赖，不保证跨机器游戏结果完全复现。

## 其他模式

- `--mode full`：额外保留选中 self-play/IL 目录中的历史文件，用于需要旧 replay/traces 的完整备份；当前迁移不需要它。
- `--dependencies repo`：省略干净的 Git tracked 文件，但包含未跟踪运行时；仅用于 SSH 等私人文件传输，不用于本次 Git 分卷。
- `--dependencies bundled`：代码和规则运行时也包含在包里，用于没有已有环境的私人传输；不上传 GitHub。
- `--part-mb 0`：关闭分卷，生成一个 zip；当前保留默认 100 MB。

## 本地验证

**17 项测试通过（24.7 秒）**，针对依赖选择、字节去重、分卷和续训，未启动训练 campaign 或修改游戏。覆盖新机器只有代码/SSH 运行时的恢复、缺失/冲突依赖、损坏或丢失分卷、活动/半成品 run、路径检查、过期 rehearsal 分片省略、当前历史池/未完成评估保留，以及旧版迁移包兼容。

真实 PyTorch checkpoint 的导出→恢复→现有 `load_resume`/`resume_optimizer` 检查覆盖权重、Adam 状态、RNG 和训练计数器。训练实现指纹仍为 `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`。尚未在两台实际 Linux lab 之间执行迁移；目标机器仍须实际运行 preflight。
