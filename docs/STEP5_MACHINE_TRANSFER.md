# Step 5：迁移到另一台 lab machine

不能只复制最后一个 `iteration-*` 文件夹或 `best.pt`。续训还需要最新 learner 的完整 checkpoint（含 Adam/RNG）、`state.json`、当前历史对手池、incumbent、evaluation anchor，以及 IL 来源记录。`best.pt` 可能比最新 learner 旧。

独立脚本 `scripts/step5-transfer.py` 默认使用 **compact + repo + 多线程 tar.zst + 1900 MB 分卷**，通过 **GitHub Release 附件**中转。新机器从 GitHub 检出 Release 对应的代码版本，再从附件恢复游戏规则运行时/DLL 和训练状态。上传成功后，恢复该快照不再需要访问旧机器。**现在支持 `restore --relocate --root 新目录`**，不需要管理员权限、bind mount 或创建原来的绝对目录。恢复不改写历史训练文件；旧训练代码须在原字节校验后按下文升级。

公有和私有仓库均可运行工作流：公有仓库的附件公开下载，私有仓库需要读取权限。这里的“只靠 GitHub”指项目代码、运行时和续训数据都从 GitHub 获取；系统、Python/conda、PyTorch、CUDA 驱动和 Mono/OpenSpiel 仍需按指南安装，这不是离线系统镜像。

## 12 October：zstd 和 GitHub Release

本次将默认依赖模式由 `external` 改为 `repo`，把 `training/runtime/distribution.json` 列出的全部规则运行时以及保存 config 选定 host 的文件装入附件，逐文件校验 SHA-256。代码和已跟踪 worker 继续从匹配的 Git commit 获取；其他恢复依赖进入包。重复 DLL 按内容去重，不会为每个运行时目录重复压缩。缺失或修改过的 DLL 会使导出失败，工作流也会在上传前确认全部运行时已打包。

默认 `--format zstd --zstd-level 3 --threads 0 --part-mb 1900`。`--threads 0` 自动使用可用 CPU 数，考虑进程 affinity 和 `SLURM_CPUS_PER_TASK`；共享机器上可明确指定 `--threads 8` 等上限，`--threads 1` 为单线程。小文件可能不足以让所有线程同时工作。压缩速度和体积取决于真实 checkpoint/数据，不保证固定提速倍数。

打包直接把 TAR 内容流送入 zstd，不先生成完整 TAR；恢复直接读取分卷，也不落地完整 TAR。内容去重、SHA-256 校验、外部依赖检查和原字节恢复保持不变。ZIP 单文件、ZIP 分卷以及旧版本清单仍会按内容自动识别，无需额外恢复参数。

脚本需要 **Python 3.11+**。Python 3.14+ 通常自带 `compression.zstd`；较早版本在两台机器的环境中安装独立迁移依赖：

```sh
ROOT=/data/not_backed_up/yxu209/ai-training
python -m pip install -r "$ROOT/requirements-transfer.txt"
```

若某个 Python 3.14+ 发行版没有编译原生 zstd，可安装 `python -m pip install backports.zstd==1.8.0`。缺少 zstd 时会提示安装方法，不会静默退回 ZIP；`--dry-run` 和显式 `--format zip` 不需要此依赖。该依赖不加入训练 requirements，也不改变训练签名。

## 当前 self-training 兼容性

目录迁移审计修订为 `step5-project-relocation/1.1.0`，源码指纹为 `f6f8028538c13fdaa5cbcd40ffde5e81fe5d9cb58b649875d8a5cd1ea8487191`，共享路径层属于这个指纹。新包记录 `source_recipe: 3` 和 `relocation_version: step5-project-relocation/1.1.0`；recipe 3 在原有 Python、host/build 输入之外纳入 `bank.py` 和 `legacy.py`，`learning/paths.py` 也属于源码指纹。旧包仍按原 recipe 1/2/3 校验，不需要重新导出，不能用新版代码冒充旧版通过校验。

`configs/learning/selfplay-resume-sources.json` 明确登记最初的目录迁移版（`c9c9bb914477feea94b61b1698db93fd5e767d0f2f88b0c715f0cfea12074140`）、**12 October comprehensive bug audit**（`8d19aff88a93edd9b734561a73c83a440f5d3517429be007d9913aceab8cffd1`）到本修订的审核升级，以及此前已支持、仍受原配置限制的升级路径。learning compatibility 保持 `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`。完成的 IL 来源另由 `imitation-handoff-sources.json` 审核，仍检查原 manifest、checkpoint 和祖先签名。未知源码、改变学习契约或未经登记的 config 变化依然拒绝。未完成的旧源码 IL 训练不因此获得跨版本续训许可。

**恢复和代码升级分两步。** 已保存的旧包先恢复到匹配的旧源码目录并 `verify`，随后在新机器升级到本修订，用 `--resume-reviewed-source` 继续原 run；不需要旧机器参与。已采用本修订的包在匹配源码下直接续训。升级保留模型、Adam/RNG、计数器、课程表和 provenance，照常记录 source transition；现有升级机制要求用新普通对局重新校准 utility，并拒绝旧源码 teacher 目标。目录映射不会绕过这些检查。

不同目录恢复旧 recipe 3 包时，也要检查 `requires_code_update`：缺少本次 `relocation_version` 的旧迁移版会要求升级，不能仅凭 recipe 3 判断训练代码已经包含审计修复。sidecar 格式仍为 `step5-project-relocation/1.0.0`，现有映射无需改写。新代码须先由项目维护者发布到 GitHub；本地未提交的修订不会自动出现在新机器。

## 目录映射与再次迁移

例如源目录 `/data/not_backed_up/yxu209/ai-training`，目标目录 `/scratch/bob/ai-training`。源目录从归档 `source_root` 读取，目标由 `--root` 指定；不同目录必须明确加 `--relocate`。原目录恢复继续支持原命令。已恢复项目的后续 `verify` 会识别与该归档匹配的映射。

根目录的 **`.step5-relocation.json`** 是单独的本机元数据，记录目标根目录、历史源目录、归档 ID 和自身摘要；不进入 Git。读取 checkpoint、state/config 引用、IL 祖先、数据集/calibration、评估、保留清理和运行时依赖时，共享解析器才将历史目录组件映射到目标。`/old/project-other` 不会匹配 `/old/project`。相对路径保持相对含义；IL 文件名仍相对所属 IL run。新生成的模型/评估/IL 引用优先使用项目相对路径，已签名的历史文本保留原样。

映射文件必须随项目保留，子进程和以后启动的会话自动读取，不依赖临时环境变量。恢复逐文件校验 SHA-256，checkpoint/report/config/state 保持原字节；不同内容的目标文件、冲突/损坏的映射、重叠根目录、`..`、链接/junction 越界和项目外依赖均拒绝。项目外数据须在**导出前**纳入项目并通过其原有数据/签名流程；工具不会自动把外部任意目录映射进来。若旧包没有包含某项外部依赖，目录迁移也无法凭空补齐。

映射包含当前目录在内最多 64 个根目录，序列化文件必须小于 64 KiB。超出限制时会在恢复写文件之前报错，不能生成一个训练程序读不进的 sidecar 后宣称成功。

再次搬迁时，先完成本机受支持的源码续训、停止 run，再从**当前根目录**运行 `export`。新清单携带所有历史根目录，下一次 `restore --relocate` 将它们直接映射到第三个目录，不需要前两台机器。不要手改 sidecar 或直接移动整个目录后沿用旧 sidecar；它会校验目标根目录。目标与各历史根目录应彼此独立，不能互为子目录。

`export` 会在包内生成对应新 state 的精确 resume config，不把它写回源项目。`verify` 校验的是**已恢复的目录**，所以应在第三目录 `restore` 后运行。若要在导出机器原地验证新包，先对原地执行一次 `restore`（不需 `--relocate`，匹配文件不会改写），补齐这个新 config 后再 `verify`；不能忽略“Missing restored file”。

## 默认复制哪些文件

| 内容 | 默认处理 |
| --- | --- |
| `state.json`、精确续训 config、best 元数据、显式 pins | 保留 |
| 最新完整 checkpoint、state 指定的恢复 checkpoint | 保留原字节，包括 optimizer/RNG |
| 当前历史对手池、incumbent、evaluation anchor | 保留；不为缩小体积而缩减训练对手池 |
| 仍在磁盘、尚未由当前池代表的历史候选模型 | 保留，因为下一次更新对手池仍可能使用 |
| 最新 iteration 的 report、评估 report/spec、已选 champion 的证据 | 保留 |
| 未完成评估的缓存、引用模型及单独选用的 host | compact/full 都保留，包括 run 外、项目内的模型和 runtime 依赖 |
| IL 来源和祖先的 state/report、选中 checkpoint、dataset manifest | 保留；不复制无用的 epoch checkpoints |
| IL 训练数据分片 | 仅当 rehearsal 仍启用且未到 `rehearsal_until` 时复制训练分片；结束后省略 |
| 已消费的 PPO/teacher/已完成评估的 replay、feature caches | 省略，显式 pin 的文件除外 |
| 无引用旧模型、retention 历史和隔离区 | 省略 |
| 代码、仓库配置、冻结规则 AI 资料 | 从 Git 获取，分卷里只记录依赖哈希 |
| 新版 ordinary/search 加速 worker 的 EXE 和清单 | 从 Git 的 `training/runtime/throughput/` 获取，不加入训练分卷 |
| 自动生成的 `artifacts/training-host/` | 标准配置下由目标机 preflight 从 Git worker 和附件中的规则 DLL 部署；若 config 直接选用其中的 host，则保留该 host 全部文件 |
| 游戏规则运行时/DLL、各运行时清单 | **包含在默认 Release 附件**；按 distribution 和训练/未完成评估 host 清单校验、去重；保留运行时加载会校验的 source pins |
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
  --mode compact --dependencies repo \
  --format zstd --zstd-level 3 --threads 0 --part-mb 1900
```

生成 `selfplay-migration.tar.zst.part0001`、`.part0002` 等分卷，以及 `selfplay-migration.tar.zst.parts.json`。默认每卷最多 **1,900,000,000 字节（约 1.77 GiB）**；相同归档的分卷数量约降为原来 100 MB 方案的 1/19，小于 1900 MB 的包只需一卷。不会额外保留完整归档。索引最后写入，包含分卷和完整归档的大小及 SHA-256。输出还记录 iteration、游戏数、Git commit、`compressed_bytes`、分卷数、格式、压缩等级和配置的线程数。进度写到 stderr，stdout 是可解析的 JSON。每次导出使用新文件名，现存分卷不会被覆盖。

## 2. 用 GitHub Release 附件传输

**Release 附件是版本发布页面上单独上传的文件，不进入 Git 提交历史，也不会随 `git clone/pull` 自动下载。** 两台 lab 机器都能连接 GitHub 即可中转，不要求它们相互 SSH 连通；本次附件已包含规则运行时，无需另传运行时 zip。

**self-hosted runner 只负责从旧机器读取、打包并上传。全部附件上传成功后，数据保存在 GitHub，旧机器或 runner 后来不可用不会让附件消失。** 新机器可独立下载和恢复；恢复的是最后一次成功上传的快照，后续未上传的训练进度不会自动进入 Release。保留该仓库、Release/tag 和全部分卷，附件不使用 Actions artifact 的按天过期机制。

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

工作流支持公有和私有仓库，不设置私有仓库门槛。当前代码仓库公开时，规则 DLL 和训练数据附件也公开；私有仓库则只允许有读取权限的用户访问。文件名中的 `private`、草稿转 prerelease 都不会改变仓库自身的访问权限。

### 已在旧机器打好包：手动上传

安装 GitHub CLI 后，在旧机器执行 `gh auth login`，使用具有仓库写入权限的账号。下面以当前代码仓库为例，`TAG` 每次使用新值；代码 commit 必须已经 push：

```sh
set -euo pipefail
REPO=riverside234/ai-training
TAG=step5-transfer-20261012-01
gh release create "$TAG" --draft \
  --repo "$REPO" --target "$(git -C "$ROOT" rev-parse HEAD)" \
  --title "Step 5 transfer $TAG" --notes "Stopped Step 5 training state; restore using the parts index." \
  --prerelease --latest=false
for part in "$BUNDLE".part[0-9]*; do
  gh release upload "$TAG" "$part" --repo "$REPO"
done
gh release upload "$TAG" "$BUNDLE.parts.json" --repo "$REPO"
gh release edit "$TAG" --repo "$REPO" --draft=false --prerelease --latest=false
```

不对这些附件执行 `git add/commit/push`，也不放开 `artifacts/` 的 ignore。默认 `--dependencies repo` 已包含规则 DLL，不需要再上传旧的运行时 zip。[GitHub CLI 上传命令](https://cli.github.com/manual/gh_release_create)

## 3. 新机器同步代码、下载附件，然后恢复

可以选择任意有写权限的新目录，保留 run 的相对布局。以下 Bash 示例把旧 `/data/not_backed_up/yxu209/ai-training` 恢复到 `/scratch/bob/ai-training`；请按实际磁盘修改 `ROOT`。`TOOLS` 是**独立的新版工具 checkout**，用于兼容 Release 中尚无 `--relocate` 的旧代码。两者均从 GitHub 获取，无需旧机器在线。

```bash
set -euo pipefail
ROOT=/scratch/bob/ai-training
TOOLS=/scratch/bob/step5-transfer-tools
REPO=riverside234/ai-training
TAG=step5-transfer-20261012-01  # 改为实际 Release 标签，Actions 摘要中也有。
CODE_REF=main                 # 应指向已发布、包含本目录迁移修订的 commit/分支；推荐填写固定 SHA。
# 首次使用 GitHub CLI 时执行 gh auth login；私有仓库还需要账号具有读取权限。
# 私有仓库通过 HTTPS clone 时，可执行 gh auth setup-git 配置 Git 凭据。
mkdir -p "$(dirname "$ROOT")"
git clone --branch "$TAG" --single-branch "https://github.com/$REPO.git" "$ROOT"
git -C "$ROOT" rev-parse HEAD  # 应等于 Release 说明中的 Source commit。
git clone "https://github.com/$REPO.git" "$TOOLS"
git -C "$TOOLS" switch --detach "$CODE_REF"
CODE_COMMIT=$(git -C "$TOOLS" rev-parse HEAD)
python -m pip install -r "$TOOLS/requirements-transfer.txt"
DEST=$ROOT/artifacts/step5-transfer/$TAG
mkdir -p "$DEST"
gh release download "$TAG" --repo "$REPO" \
  --pattern 'selfplay-migration.tar.zst.part*' --dir "$DEST"
BUNDLE=$DEST/selfplay-migration.tar.zst.parts.json

RESTORE_JSON=$DEST/restored.json
python "$TOOLS/scripts/step5-transfer.py" restore \
  --root "$ROOT" --bundle "$BUNDLE" --relocate > "$RESTORE_JSON"
python "$TOOLS/scripts/step5-transfer.py" verify \
  --root "$ROOT" --bundle "$BUNDLE" --relocate

# 缺少本次 relocation_version 的旧迁移代码也需升级；原文件校验必须先完成。
NEEDS_UPDATE=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["requires_code_update"]))' "$RESTORE_JSON")
if [ "$NEEDS_UPDATE" = 1 ]; then
  git -C "$ROOT" fetch origin "$CODE_COMMIT"
  git -C "$ROOT" switch --detach "$CODE_COMMIT"
fi
```

通配符 `part*` 同时下载数字分卷和 `.parts.json` 索引。Release 标签定位到导出时的 commit，因此新机器不会误用后来的训练代码。已有目标仓库时，先确认没有需要保留的改动，再 `git fetch origin tag "$TAG"`、`git switch --detach "$TAG"`；恢复后需要改代码时可从此处创建分支。不要使用 Release 页面自动生成的 **Source code (zip/tar.gz)** 来代替训练附件。[附件下载命令](https://cli.github.com/manual/gh_release_download)

恢复直接读取分卷，不必手动合并，也不需要另存大 TAR/ZIP。它先检查所有分卷、归档内容和 Git 代码依赖哈希，再恢复规则运行时和训练文件。无需执行 `import-runtime`。相同目标文件可复用；缺失外部依赖或不同内容会停止恢复，已有文件不会被覆盖。

首次检出旧 tag 是为了满足归档的**原始源码/外部文件哈希**。直接在新版 checkout 恢复旧包会报冲突，应使用上述两步流程，不能删哈希检查或改 state。工具依赖同一 checkout 的 `src/brass_ai/learning/paths.py`，不要只单独复制一个新版脚本。若 Git 提示有本地修改/冲突，先保留并核对这些文件，不要强制覆盖。新版训练源码必须包含对应的显式审核记录；单纯换一个更新的 commit 并不自动获得兼容性。

旧版导出器可能没有收集未完成评估单独选用的 host，或代码目录之外的运行时 source pins。这类旧包只能在另有**完全匹配**的依赖时继续；迁移工具无法恢复从未上传的文件。新的导出器会提前拒绝缺失依赖。`verify` 证明归档及清单内文件的完整性，不能替代实际 preflight、数据/IL 检查和 resume 的签名验证。

`verify` 用于恢复后、升级代码和开始续训**之前**。升级后的源码与旧包不同，或训练后的 state/checkpoint 更新，均会使旧包完整校验失败，这是预期行为。原归档可永久保留并在另一个干净目录重新恢复校验。

## 4. 新机器 preflight，再继续原 run

安装并激活 `ai-training` conda 环境，按[训练指南](TRAINING_RECOVERY_LAB.md#transfer-and-preflight)安装 PyTorch/CUDA、Python 依赖、Mono/OpenSpiel。迁移脚本需要 Python 3.11+；zstd 依赖按本文开头安装，旧 ZIP 模式仍只依赖标准库。

`restore` 和 `verify` 输出两条完整命令：

1. `preflight`：在新机器验证设备和规则运行时，不能沿用旧机器的通过记录。
2. `resume`：使用保存的 config、原 `--run` 和需要的 `--imitation-run`，继续 20 iterations。可修改追加轮数，不要重新传 `--initial`。

先执行 `preflight`，成功后执行 `resume`。已有 self-play state 会恢复 PPO checkpoint；`--imitation-run` 仅验证 IL 来源，不会重做 IL。课程表、历史池、teacher 预算和计数器保持原状态。

接着执行以下 Bash 命令，沿用上节变量；路径从恢复输出读取，适用于不同 run 名称及有/无 IL 来源的包：

```bash
set -euo pipefail
CONFIG=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"])' "$RESTORE_JSON")
RUN=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"])' "$RESTORE_JSON")
IL_RUN=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["imitation_run"] or "")' "$RESTORE_JSON")
RESUME_ARGS=()
if [ -n "$IL_RUN" ]; then RESUME_ARGS+=(--imitation-run "$IL_RUN"); fi
if [ "$NEEDS_UPDATE" = 1 ]; then RESUME_ARGS+=(--resume-reviewed-source); fi

python "$ROOT/scripts/step5.py" --root "$ROOT" --config "$CONFIG" --device cuda preflight
python "$ROOT/scripts/step5.py" --root "$ROOT" --config "$CONFIG" --device cuda \
  train --run "$RUN" --iterations 20 "${RESUME_ARGS[@]}"
```

CPU 验证把两处 `cuda` 改成 `cpu`。旧包的恢复 JSON 中 `requires_code_update: true` 表示必须先安装支持映射的训练源码，并在第一次源码续接时使用 `--resume-reviewed-source`；输出的普通 `resume` 字符串本身不会替你批准升级。不同 SHA 若不在审核列表中将明确报错。升级完成以后，新会话直接用相同 config/run 续训，映射仍然有效。

新版 preflight 还会安装并检查两个加速 worker；输出中的 `throughput_workers` 应显示 ordinary/search 均为 `optimized: true`。需要的规则 DLL 已由 Release 附件恢复。当前 self-training 默认使用一张 GPU；GPU 分配是目标机器的运行选项，不从旧机器自动复制。只有确实使用多卡推理时才按新机器分配设置 `CUDA_VISIBLE_DEVICES`，并在输出的续训命令中、`train` 子命令之前加入 `--inference-devices auto`。不要把这个参数加到 `preflight` 命令。

旧机器仍可用时，建议在回收前完成一次目标机恢复验证；之后该快照的恢复无需旧机器。不同 GPU、库版本及搜索线程调度不保证后续结果逐步一致；本工具保证保存文件的字节完整性和恢复依赖，不保证跨机器游戏结果完全复现。

## 其他模式

- `--mode full`：额外保留选中 self-play/IL 目录中的历史文件，用于需要旧 replay/traces 的完整备份；当前迁移不需要它。
- `--dependencies repo`：**默认**。干净且已跟踪的代码从 Git 获取；全部规则运行时/DLL、选定 host 和其他未跟踪依赖进入 Release 附件。
- `--dependencies bundled`：代码和规则运行时都包含在包里，体积更大；恢复脚本及 Python/zstd 仍需先可用。自动工作流固定使用 `repo`。
- `--dependencies external`：保留旧行为，代码和运行时均需另行准备。它不满足“只靠 GitHub 代码和当前附件恢复”的目标，自动工作流不使用此模式；旧 external 包也不会因此自动补齐 DLL。
- `--format zip`：使用旧 ZIP/DEFLATE 导出；相应 bundle 文件名使用 `.zip`。恢复会自动识别 ZIP/zstd，不按后缀猜测格式。
- `--part-mb 0`：关闭分卷，生成一个 `.tar.zst`（或显式 ZIP）。手动上传 Release 时须确认整个文件小于 2 GiB；工作流使用分卷以始终保留索引校验。
- `--part-mb 100`：保留普通 Git 传输兼容性。100 MB 小于 GitHub 单文件 100 MiB 限制；单次 push 仍受 2 GiB 总量限制，需要分批 commit/push。默认 1900 MB 分卷仅用于 Release，不能直接 `git add` 后 push。[普通 Git 限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[push 限制](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit)

## 本地验证

### 目录迁移审计（1.1.0）

本次先复现、再修复以下问题，未修改真实训练数据或上传/推送任何内容：

| 问题 | 原因与修复 |
| --- | --- |
| 未完成评估的独立 host、部分运行时 source pins 未进入归档 | 原收集器只遍历训练 host 和固定代码树；现在 compact/full 都收集评估模型、host 清单及加载时校验的源文件。缺失依赖在发布归档前报错。 |
| 相同 host 改用新绝对路径或相对路径后，已有评估不能复用 | 原 spec 对 host 文本做精确比较；现在只对该字段验证组件映射后的同一文件，其余配置仍精确比较，并保留原 spec/provenance 字节与摘要。 |
| 某些畸形归档错误地返回 `verified: true` | 文件与目录/临时 run lease 冲突时，恢复阶段把临时创建的路径视为已恢复；过大的映射也可能写出无法读取的 sidecar。现在先拒绝这些清单、活动标记、非目录父节点及超限映射，再写入。另拒绝大小写变体的私有配置路径及不可移植文件名。旧 full 包中不活动的 OS guard 文件仍按原字节校验恢复；新导出省略它们。 |
| runtime 安装可能越过链接写文件，或缓存返回绕过检查 | ordinary/search 安装器现在在复制和缓存返回之前检查源、目标及目标文件；preflight 不能先写到链接目录后才报错。Windows junction 回归检查实际执行。 |

路径追踪覆盖如下调用边界；底层 checkpoint/replay/inference API 接收上层已解析的实际文件路径，历史 JSON 中的旧字符串不会被统一替换：

| 阶段 | 路径和完整性边界 |
| --- | --- |
| export | `inventory → below/resolve_project_path`：state/live models、IL ancestry、dataset/calibration、pins、pending evaluations、runtime/source pins；原字节哈希。 |
| restore / verify | 清单、分卷和 payload 校验 → 路径布局/目标冲突检查 → 原子不覆盖写入 → 原 recipe 源码校验 → 独立 sidecar。 |
| preflight | distribution/assets、冻结 bank、LegacyWorker、规则/加速 worker；新目录内安装，DLL/source pins 与真实规则 parity 仍检查。 |
| resume | `load_resume` 校验 checkpoint/state/model/generation/signature；恢复 Adam/RNG，显式 source receipt 才可升级；历史引用经 `project_path` 解析。 |
| IL / rehearsal | completed imitation 的原 manifest/parent/checkpoint 绑定；`load_dataset` 在父进程解析分片/calibration，再交给 spawned readers。 |
| workers / evaluation | actor 子进程从目标根目录读取 sidecar；parent 创建目标目录下的 jobs，inference 加载已解析 checkpoint；已有 evaluation 保留原 spec 摘要。 |
| retention / re-export | live/pinned/cleanup 引用继续通过映射；再次导出用当前源码/state，携带历史根目录，下一目标不需访问前两处目录。 |

最终单元组合命令如下（先安装迁移/训练依赖，并将 `src`、`tests/Transfer`、`tests/Recovery`、`tests/R2` 加入 `PYTHONPATH`）：

```sh
python -m unittest test_relocation_audit test_relocation test_step5_transfer -v
python -m unittest test_audit_resume test_time_resume test_resume_evaluations test_imitation \
  test_audit_retention test_revision test_training_runtime_review test_step5_working_directory -v
python -m unittest test_native_runtime_review.NativeRuntimeReviewTests.test_search_transformation_is_pinned_and_changed_helper_is_rejected -v
```

在 Windows / Python 3.12 / PyTorch 2.8 CPU 上，最终迁移/归档组 **57 项全部通过**；相关恢复组 **38 项中 36 通过、2 跳过**（真实 CUDA 硬件、POSIX 删除当前工作目录）；runtime source pin 检查 **1 项通过**，合计 **96 项独立测试，94 通过、2 跳过**。首次在沙箱运行的 junction 跳过项，最终已在普通用户上下文实际运行并通过，不需要管理员权限。相关现有测试没有失败。源码兼容测试同时拒绝未知 source、未经审核 config 变化及错误 checkpoint/state 绑定；实际 CPU checkpoint 逐项比较权重、Adam、RNG/下一次随机输出和完整 state。

原始失败日志、最终单元结果及本次原生演练日志均位于 `artifacts/step5-relocation-audit/`。首次原生演练已通过新目录 preflight 和 PPO 续训，但嵌套测试目录使 Windows retention rename 路径达到 267 字符，触发系统路径限制。演练改用短临时目录；Windows 上仍须让项目及生成文件路径处于系统支持的长度内，本修订没有修改系统长路径设置。演练使用单独的新 evaluation config 保留 replay，以检验第三目录能复用真实缓存，不改写归档的训练 config 或既有 spec。

**最终真实运行时演练通过**：`tests/Transfer/smoke_migration_audit.py` 使用 Git commit `05def96720df7635626b7a3ea0de1a4ab2888da3` 的真实旧 exporter，在归档前构造包含绝对 host/checkpoint 引用的有界 fixture。删除源目录后恢复到第二目录，校验 **2,799 个文件**，更新到上面的已审核源码，完成真实 DLL preflight、一轮两局 PPO 和四局评估。再次导出为 45 个 1 MB 测试分卷、删除第二目录，恢复到第三目录并校验 **2,819 个文件**；第三目录再次 preflight，**四局评估全部命中原 replay 缓存，原 spec 字节和结果不变**，再续训一轮。训练计数 **iteration 2 → 3 → 4、games 4 → 6 → 8**；第二次搬迁没有增加 source transition。1 MB 仅用于分卷压力测试，实际默认仍为 1900 MB。

演练在独立 Python 启动及 spawned workers 中禁止对前两处目录的 Python 文件访问，结束时两处目录仍不存在；输入测试 run 的全部文件哈希保持不变。执行上下文的管理员标志为 false。耗时约 **864 秒**，结果见 `native-result.json`、`second-*`/`third-*` 日志和 `user-context.json`。`scope-check.json` 确认与任务无关的未提交文件和以前的审核记录未变。`final-transfer-tests.log`、`final-tests.log`、`native-source-pin.log` 保存单元结果。

离线的公有/私有 Release 工作流模拟、actionlint 及本文六个 shell 代码块的 Bash 语法检查也通过，未执行任何真实 GitHub 上传/下载。未验证真实 Linux/CUDA 机器、跨设备逐步一致性、大于 2 GB 的实际归档或完整强度评估；模型/Adam/RNG 的 CPU 恢复与此处有限演练不能证明不同设备和时钟预算下后续训练逐步相同。旧包遗漏的依赖仍须另行提供，系统/Python/Mono/CUDA/OpenSpiel 环境仍按指南准备。

### 上一版验证记录（1.0.0）

目录迁移修订在 **Windows / Python 3.12 / PyTorch 2.8 CPU** 验证。最终组合运行 **46 项**（30 项原迁移测试、11 项新增迁移测试、5 项 audit/time-budget 续训测试），45 项通过，1 项 junction 测试因沙箱无法创建链接跳过；随后在普通用户上下文单独实测 junction 越界拒绝、原子发布不覆盖、非法映射及原目录旧包恢复，**4 项全部通过**。另运行 **38 项**相关恢复/IL/评估/保留清理/运行时/启动目录测试，36 项通过，CUDA 和 POSIX 各跳过 1 项；其中 5 项与组合运行重叠，不重复计数。

新增用例导出临时项目后**删除源根目录**，恢复到另一个目录，再次导出并搬到第三个目录。覆盖绝对/相对路径、ZIP/zstd/分卷和旧 recipe 1/2 清单、所有归档文件原字节、后续独立进程读取映射、IL 祖先及真实 checkpoint、并行 rehearsal 读取与 calibration 校验、评估模型依赖和原 spec 绑定、损坏归档、目标冲突、非法映射/越界引用，以及保留清理的引用保护。真实 utility-head checkpoint 通过原有 `load_resume`/`resume_optimizer` 恢复并比较全部权重、Adam 状态、RNG（包括下一次随机输出）及完整 state；篡改 checkpoint/state 绑定仍被拒绝。评估用例验证真实模型加载和已有 summary/spec 复用，没有进行完整强度评估 campaign。

**旧归档的真实运行时演练通过**：`tests/Transfer/smoke_relocation.py` 从 commit `05def96720df7635626b7a3ea0de1a4ab2888da3` 提取历史代码，用该版本的真实 exporter 打包已提交的两局/轮 fixture；删除临时源项目，在不同根目录恢复并逐一校验 **2,799 个文件**。原始 state/checkpoint 在恢复过程中不变；随后更新目标源码、运行真实两局规则 parity 和两个加速 worker 的 preflight，再以 `--resume-reviewed-source` 实际续训一轮。结果 **iteration 2 → 3、games 4 → 6**，原 checkpoint 字节、原 state 备份、源码升级记录全部检查通过。原始测试 fixture 未修改，总耗时约 777 秒。

本修订证据在 `artifacts/step5-relocation-validation/`：`final-tests.log`、`final-security.log`、`recovery-tests.log`、`native-smoke-result.json`、`native-preflight.log`、`native-resume.log`。演练共享已安装的第三方 Python 环境，系统依赖未打包；未实际联网下载/上传，未在两台 Linux lab 或真实 CUDA 硬件间迁移，目标机器仍须运行 preflight。

工作流重新通过 actionlint 1.7.12、Bash/嵌入 Python 语法和 `tests/Transfer/check_release_workflow.py` 的公有/私有仓库模拟上传检查。Release 下载行为不变，分卷或索引上传失败仍不会发布；未实际运行 GitHub self-hosted runner。
