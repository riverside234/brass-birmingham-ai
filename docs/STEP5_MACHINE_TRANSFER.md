# Step 5：迁移到另一台 lab machine

不能只复制最后一个 `iteration-*` 文件夹或 `best.pt`。续训还需要最新 learner 的完整 checkpoint（含 Adam/RNG）、`state.json`、当前历史对手池、incumbent、evaluation anchor，以及 IL 来源记录。`best.pt` 可能比最新 learner 旧。

独立脚本 `scripts/step5-transfer.py` 默认使用 **compact + external + 多线程 tar.zst + 1900 MB 分卷**，通过 **GitHub Release 附件**中转。新机器通过 Git clone/pull 获取当前代码，游戏规则运行时通过之前的 SSH zip 方法传输，其他训练文件从 Release 下载后恢复。工具不改训练实现、签名、模型格式或超参数。

## 12 October：zstd 和 GitHub Release

默认 `--format zstd --zstd-level 3 --threads 0 --part-mb 1900`。`--threads 0` 自动使用可用 CPU 数，考虑进程 affinity 和 `SLURM_CPUS_PER_TASK`；共享机器上可明确指定 `--threads 8` 等上限，`--threads 1` 为单线程。小文件可能不足以让所有线程同时工作。压缩速度和体积取决于真实 checkpoint/数据，不保证固定提速倍数。

打包直接把 TAR 内容流送入 zstd，不先生成完整 TAR；恢复直接读取分卷，也不落地完整 TAR。内容去重、SHA-256 校验、外部依赖检查和原字节恢复保持不变。ZIP 单文件、ZIP 分卷以及旧版本清单仍会按内容自动识别，无需额外恢复参数。

脚本需要 **Python 3.11+**。Python 3.14+ 通常自带 `compression.zstd`；较早版本在两台机器的环境中安装独立迁移依赖：

```sh
ROOT=/data/not_backed_up/yxu209/ai-training
python -m pip install -r "$ROOT/requirements-transfer.txt"
```

若某个 Python 3.14+ 发行版没有编译原生 zstd，可安装 `python -m pip install backports.zstd==1.8.0`。缺少 zstd 时会提示安装方法，不会静默退回 ZIP；`--dry-run` 和显式 `--format zip` 不需要此依赖。该依赖不加入训练 requirements，也不改变训练签名。

## 11 October：适配新版 self-play

迁移工具的源码指纹已对齐当前训练器，包含 `host.py`、`src/BrassTrainingHost/*.cs`、两个 host 构建脚本和 `training/runtime/throughput/*/runtime.json`。新包记录指纹算法版本 `source_recipe: 2`；早期包仍按原算法校验，不会将旧训练状态重新标记为新版本。IL 自动收集方式不变，两台机器均需同步更新后的迁移脚本。原来的 100 MB ZIP 方案仍兼容，但新导出默认使用上面的 zstd/Release 方案。

**代码更新与搬机器是两件事。** 如果 run 已用当前版本完成训练迭代，可直接导出；如果只是 Git pull 了新版代码，但 `state.json` 仍属于旧训练版本，先按[训练指南中的受支持续训流程](TRAINING_RECOVERY_LAB.md#six-throughput-changes-11-october)完成版本迁移，必要时使用原训练命令的 `--resume-reviewed-source`。待迭代提交并停止训练后再导出。迁移脚本仍拒绝真实的源码/状态不匹配，不自动修改 checkpoint 或绕过审核记录。

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
| 新版 ordinary/search 加速 worker 的 EXE 和清单 | 从 Git 的 `training/runtime/throughput/` 获取，不加入训练分卷 |
| 自动生成的 `artifacts/training-host/` | 标准配置下无需复制；目标机 preflight 从 Git worker 和 SSH 规则 DLL 部署 |
| 游戏规则运行时/DLL | 从 SSH 获取，分卷里只记录依赖哈希，**不打包到默认 Release 附件** |
| conda 环境、CUDA 驱动、其他 runs、旧机器 preflight 和进程锁 | 不复制 |

相同内容只压缩存储一次，例如完全相同的模型文件有多个路径时不会重复占用分卷空间。恢复时仍建立原来的文件布局。所有源文件原样保留，工具不会清理旧机器。实际体积取决于当前池和 rehearsal；`--dry-run` 显示逻辑大小、去重后的大小、类别大小及最大的十个文件。

## 1. 旧机器停止训练并导出

等待当前训练命令正常完成后执行。脚本拒绝活动 run、未提交 iteration/checkpoint、损坏或缺失的依赖；中断留下的半个 iteration 应先在旧机器用原命令恢复完成。

把 `RUN` 改成你实际的 self-play 目录，`BUNDLE` 放在该 run 和 IL run 之外：

```sh
conda activate ai-training
ROOT=/data/not_backed_up/yxu209/ai-training
RUN=$ROOT/artifacts/step5/selfplay-best-v1
BUNDLE=$ROOT/artifacts/step5-transfer/selfplay-migration.tar.zst
python -m pip install -r "$ROOT/requirements-transfer.txt"

# 先查看体积；不生成压缩包。
python "$ROOT/scripts/step5-transfer.py" export \
  --root "$ROOT" --run "$RUN" --bundle "$BUNDLE" --dry-run

# 以下均为默认设置，显式写出便于检查。
python "$ROOT/scripts/step5-transfer.py" export \
  --root "$ROOT" --run "$RUN" --bundle "$BUNDLE" \
  --mode compact --dependencies external \
  --format zstd --zstd-level 3 --threads 0 --part-mb 1900
```

生成 `selfplay-migration.tar.zst.part0001`、`.part0002` 等分卷，以及 `selfplay-migration.tar.zst.parts.json`。默认每卷最多 **1,900,000,000 字节（约 1.77 GiB）**；相同归档的分卷数量约降为原来 100 MB 方案的 1/19，小于 1900 MB 的包只需一卷。不会额外保留完整归档。索引最后写入，包含分卷和完整归档的大小及 SHA-256。输出还记录 iteration、游戏数、Git commit、`compressed_bytes`、分卷数、格式、压缩等级和配置的线程数。进度写到 stderr，stdout 是可解析的 JSON。每次导出使用新文件名，现存分卷不会被覆盖。

## 2. 用 GitHub Release 附件传输

**Release 附件是版本发布页面上单独上传的文件，不进入 Git 提交历史，也不会随 `git clone/pull` 自动下载。** 两台 lab 机器都能连接 GitHub 即可中转，不要求它们相互 SSH 连通。游戏运行时仍沿用已有私人传输流程。

| 仓库可见性 | Release 附件的访问方式 | 文件大小限制 |
| --- | --- | --- |
| 私有仓库 | 需要登录，并拥有该仓库读取权限 | 每个附件小于 2 GiB |
| 公共仓库 | 公开下载；不能把其中某一个已发布附件单独设为私有 | 每个附件小于 2 GiB |

“私有”控制访问权限，不会提高大小上限。一份 Release 最多 1000 个附件（包括索引）；默认每卷 1900 MB，留在 2 GiB 以下。这里不受普通 Git 单文件 100 MiB / 单次 push 2 GiB 的限制。[GitHub Release 说明及限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

### 推荐：GitHub Actions 自动打包并上传

工作流位于 [`.github/workflows/step5-transfer-release.yml`](../.github/workflows/step5-transfer-release.yml)，名称 **Step 5 transfer release**，仅通过 **Run workflow** 手动触发。触发后自动打包、校验、上传并发布 Release；不会自动停止训练或启动续训。

一次性准备：

1. 将工作流、迁移脚本、`requirements-transfer.txt` 及匹配的训练代码提交到 GitHub；工作流需先存在于默认分支，GitHub 才会显示手动运行入口。
2. 在**保存原 run 的旧 lab 机器**上，按仓库 **Settings → Actions → Runners → New self-hosted runner** 的说明注册 Linux runner。为这台机器设置唯一自定义标签，例如 `step5-transfer-lab1`，填写到运行参数 `runner_label`。服务账号需能读写项目目录；不要把 runner 的工作目录设成正在训练的项目目录。[GitHub self-hosted runner 标签](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow)
3. runner 的 PATH 中需有 `git`、`gh` 和支持 `venv` 的 Python 3.11+；可通过 `python_executable` 指定 Python 完整路径。打包不需要 PyTorch/CUDA，工作流会在 runner 临时目录创建独立环境并安装迁移依赖。

每次迁移：等当前迭代完成并正常停止训练，确认旧机器保存 run 对应的代码已提交且推到 GitHub，然后进入 **Actions → Step 5 transfer release → Run workflow**。选择与旧机器 `HEAD` 相同的 branch/tag，并填写：

| 参数 | 默认值 / 用途 |
| --- | --- |
| `runner_label` | `step5-transfer`；改为源机器的唯一标签 |
| `source_root` | `/data/not_backed_up/yxu209/ai-training` |
| `run_path` | `artifacts/step5/selfplay-best-v1`，相对项目根目录 |
| `python_executable` | `python3`，需要 3.11+ 和 venv |
| `threads` | `0` 自动；共享机器可指定 `8` 等线程数 |
| `zstd_level` | `3`，范围 1–22 |
| `part_mb` | `1900`，范围 1–2000 MB |

GitHub 托管的 `ubuntu-latest` 访问不到旧 lab 的 `/data/...`，因此这里需要 self-hosted runner；尚未注册或标签不匹配时任务会排队。工作流不执行 checkout、pull 或 reset，源码版本不匹配、代码/config 有未提交改动、run 活动或迭代未提交时会失败，并留下明确错误。

工作流使用自带的 `GITHUB_TOKEN` 和 `contents: write` 权限上传到**当前仓库**，不需要另配个人 token。Release 标签为 `step5-transfer-<run_id>-<run_attempt>`；文件保存在源机器 `artifacts/step5-transfer/<标签>/`。先创建草稿，上传所有分卷，最后上传 `.parts.json`，全部成功后发布为 prerelease，且不标记为 Latest。失败时保留草稿和本地包，原训练数据不变；重新运行会使用新 attempt/tag。完成后 Actions 摘要提供 Release 链接和新机器下载命令。

若当前代码仓库公开，这个工作流产生的已发布附件也公开。要保持训练文件私有，应在私有仓库中运行迁移工作流；不会因为文件叫“private”或 Release 是 prerelease 就隐藏附件。

### 已在旧机器打好包：手动上传

安装 GitHub CLI 后，在旧机器执行 `gh auth login`，使用具有仓库写入权限的账号。下面以当前代码仓库为例，`TAG` 每次使用新值；代码 commit 必须已经 push：

```sh
REPO=riverside234/ai-training
TAG=step5-transfer-20261012-01
gh release create "$TAG" "$BUNDLE".part[0-9]* "$BUNDLE.parts.json" \
  --repo "$REPO" --target "$(git -C "$ROOT" rev-parse HEAD)" \
  --title "Step 5 transfer $TAG" --notes "Stopped Step 5 training state; restore using the parts index." \
  --prerelease --latest=false
```

不对这些附件执行 `git add/commit/push`，也不放开 `artifacts/` 的 ignore。默认 `--dependencies external` 不含私有规则 DLL；不要把现有游戏运行时 zip 加入附件列表。[GitHub CLI 上传命令](https://cli.github.com/manual/gh_release_create)

## 3. 新机器同步代码、下载附件，然后恢复

使用与旧机器相同的绝对根目录 `/data/not_backed_up/yxu209/ai-training/`，并保留相同 run 相对路径。部分报告/配置含绝对路径，工具不会重写已签名的数据。游戏运行时先用已有 SSH zip 方法导入到原来的路径。

```sh
ROOT=/data/not_backed_up/yxu209/ai-training
git -C "$ROOT" pull --ff-only
python -m pip install -r "$ROOT/requirements-transfer.txt"
REPO=riverside234/ai-training
TAG=step5-transfer-20261012-01  # 改为实际 Release 标签，Actions 摘要中也有。
DEST=$ROOT/artifacts/step5-transfer/$TAG
mkdir -p "$DEST"
# 私有仓库先执行 gh auth login，登录有读取权限的账号。
gh release download "$TAG" --repo "$REPO" \
  --pattern 'selfplay-migration.tar.zst.part*' --dir "$DEST"
BUNDLE=$DEST/selfplay-migration.tar.zst.parts.json

python "$ROOT/scripts/step5-transfer.py" restore --root "$ROOT" --bundle "$BUNDLE"
python "$ROOT/scripts/step5-transfer.py" verify --root "$ROOT" --bundle "$BUNDLE"
```

通配符 `part*` 同时下载数字分卷和 `.parts.json` 索引。`git pull` 只同步代码；不要使用 Release 页面自动生成的 **Source code (zip/tar.gz)** 来代替训练附件。若旧机器使用其他 branch/tag，新机器也需同步到包记录的匹配版本；仅拉取最新默认分支并不保证匹配。[附件下载命令](https://cli.github.com/manual/gh_release_download)

恢复直接读取分卷，不必手动合并，也不需要另存大 TAR/ZIP。它先检查所有分卷、归档内容和外部代码/运行时哈希，再写入训练文件。相同目标文件可复用；缺失外部依赖或不同内容会停止恢复，已有文件不会被覆盖。

**本次更新只调整独立迁移工具和传输流程，不改变训练签名。** 以后仍可正常 Git pull 更新；若某次更新实际改变了训练实现或规则，按那次更新的已有续训/迁移说明处理，不能手改 state 签名绕过检查。分卷记录导出时的 commit，便于核对版本。

`verify` 用于恢复后、开始续训前；开始训练后 state/checkpoint 改变是正常的，不应再用旧分卷验证正在更新的 run。

## 4. 新机器 preflight，再继续原 run

安装并激活 `ai-training` conda 环境，按[训练指南](TRAINING_RECOVERY_LAB.md#transfer-and-preflight)安装 PyTorch/CUDA、Python 依赖、Mono/OpenSpiel。迁移脚本需要 Python 3.11+；zstd 依赖按本文开头安装，旧 ZIP 模式仍只依赖标准库。

`restore` 和 `verify` 输出两条完整命令：

1. `preflight`：在新机器验证设备和规则运行时，不能沿用旧机器的通过记录。
2. `resume`：使用保存的 config、原 `--run` 和需要的 `--imitation-run`，继续 20 iterations。可修改追加轮数，不要重新传 `--initial`。

先执行 `preflight`，成功后执行 `resume`。已有 self-play state 会恢复 PPO checkpoint；`--imitation-run` 仅验证 IL 来源，不会重做 IL。课程表、历史池、teacher 预算和计数器保持原状态。

新版 preflight 还会安装并检查两个加速 worker；输出中的 `throughput_workers` 应显示 ordinary/search 均为 `optimized: true`。现有规则运行时 zip 继续提供它们需要的私有 DLL，无需将这些 DLL 放进 Git 分卷。GPU 分配是目标机器的运行选项，不从旧机器自动复制；若要多卡推理，按新机器分配设置 `CUDA_VISIBLE_DEVICES`，并在输出的续训命令中、`train` 子命令之前加入 `--inference-devices auto`。不要把这个参数加到 `preflight` 命令。

保留旧机器的 run，直到新机器成功续训。不同 GPU、库版本及搜索线程调度不保证后续结果逐步一致；本工具保证保存文件的字节完整性和恢复依赖，不保证跨机器游戏结果完全复现。

## 其他模式

- `--mode full`：额外保留选中 self-play/IL 目录中的历史文件，用于需要旧 replay/traces 的完整备份；当前迁移不需要它。
- `--dependencies repo`：省略干净的 Git tracked 文件，但包含未跟踪运行时；仅用于 SSH 等私人文件传输，不用于默认 Release 附件。
- `--dependencies bundled`：代码和规则运行时也包含在包里，用于没有已有环境的私人传输；不上传 GitHub。
- `--format zip`：使用旧 ZIP/DEFLATE 导出；相应 bundle 文件名使用 `.zip`。恢复会自动识别 ZIP/zstd，不按后缀猜测格式。
- `--part-mb 0`：关闭分卷，生成一个 `.tar.zst`（或显式 ZIP）。手动上传 Release 时须确认整个文件小于 2 GiB；工作流使用分卷以始终保留索引校验。
- `--part-mb 100`：保留普通 Git 传输兼容性。100 MB 小于 GitHub 单文件 100 MiB 限制；单次 push 仍受 2 GiB 总量限制，需要分批 commit/push。默认 1900 MB 分卷仅用于 Release，不能直接 `git add` 后 push。[普通 Git 限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[push 限制](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit)

## 本地验证

**27 项迁移测试通过（39.7 秒，Windows/Python 3.12）**，未启动训练 campaign 或修改游戏。覆盖默认多线程 zstd、20 MB 随机大文件、1900 MB 分卷配置、单线程/等级参数、CLI JSON 输出、内容去重和独立文件恢复、无 zstd 时的 ZIP 导出/恢复、旧 ZIP 单文件/分卷、旧版本清单、缺失依赖、目标冲突、活动/半成品 run、损坏/缺失分卷、zstd 尾部截断/损坏/追加内容、TAR 路径/链接/重复成员、过期 rehearsal 分片省略，以及当前历史池/未完成评估保留。

真实 PyTorch checkpoint 的 zstd 导出→恢复→现有 `load_resume`/`resume_optimizer` 检查覆盖权重、Adam 状态、RNG 和训练计数器。测试也检查源码指纹与当前训练器一致、扩展 host 输入、LF/CRLF 一致性、新 worker 从 Git 复用和旧指纹兼容。

工作流通过 actionlint 1.7.12、Bash 语法和嵌入 Python 检查；本地模拟覆盖 Release 元数据生成、源码 commit 不匹配拒绝、全部附件上传成功后发布、分卷上传失败不发布、索引上传失败不发布。模拟 `gh` 不执行任何真实网络上传。尚未在实际 self-hosted runner 上运行，也未在两台 Linux lab 间迁移或验证真实 CUDA 恢复；目标机器仍须实际运行 preflight。
