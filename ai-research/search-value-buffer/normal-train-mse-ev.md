# Search-value 训练 buffer 的 MSE 和 EV

在 lab 的 `ai-training` 根目录运行下面的命令。它用最新已提交的 checkpoint，评估最近 8 个 collection 中保留在 `search-values` buffer 内的 normal 训练局。每代一行，`POOLED` 用所有成功读取的位置合并计算。

- 换训练 run 时，修改 `run = Path(...)`。
- 想缩小检查范围时，修改两个 `g - 8`。更早的数据只有在 checkpoint 仍引用对应 shard 时才可读取。
- 目标用终局效用 `u`，与 held-out 诊断的指标口径相同。这里衡量的是训练集上的拟合效果。
- 命令只读。若输出 `SKIP`，`POOLED` 只包含成功读取的批次。

```bash
PYTHONPATH=src python - <<'PY'
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.recovery.search_value_training import predict_rows

run = Path("artifacts/step5/joint7-v11-2-g1190-restart-3")
state = json.loads((run / "state.json").read_text())
g = state["iteration"]
entry = next(r for r in state["history"] if r["generation"] == g)

torch.set_num_threads(2)
model, raw = load_checkpoint(Path(entry["path"]))
if raw["step"] != g or raw["model_id"] != entry["model_id"]:
    raise ValueError("checkpoint identity differs from state.json")
model.eval()

by_age = {}
for r in raw["training"]["search_value"]["buffer"]["receipts"]:
    if r["split"] == "train" and g - 8 <= r["generation"] < g:
        by_age.setdefault(r["generation"], []).append(r)

archive = {}
for name, opener in (("retention-history.json.gz", gzip.open),
                     ("retention-history.json", open)):
    path = run / name
    if path.is_file():
        with opener(path, "rt", encoding="utf-8") as f:
            archive = json.load(f)
        break

def train_tables(age):
    folder = run / f"iteration-{age:04d}"
    collection = folder / "teacher-shadow/collection.json"
    report = folder / "report.json"
    if collection.is_file():
        c = json.loads(collection.read_text())
    else:
        r = json.loads(report.read_text()) if report.is_file() else \
            archive.get(f"iteration-{age:04d}/report.json", {})
        c = r.get("summary", r).get("teacher_phase", {}).get("collection", {})
    jobs = list(c.get("post_admission_per_table_seconds", {}))
    tables = c.get("teacher_actions", {}).get("per_table", [])
    if not jobs or len(jobs) != len(tables):
        raise ValueError("missing table-to-job mapping")
    return {
        "teacher/" + job: table["cotton_exploration"]
        for job, table in zip(jobs, tables)
        if table["split"] == "train"
    }

def metrics(target, main, corrected):
    variance = np.var(target)
    base_err, corrected_err = target - main, target - corrected
    return (
        np.mean(base_err**2),
        np.mean(corrected_err**2),
        1 - np.var(base_err) / variance if variance > 1e-12 else np.nan,
        1 - np.var(corrected_err) / variance if variance > 1e-12 else np.nan,
    )

def show(label, games, target, main, corrected):
    values = "  ".join(f"{v:13.6f}" for v in metrics(target, main, corrected))
    print(f"{label:>7}  {games:5}  {len(target):5}  {values}")

print(f"Model g{g} | train buffer | normal games only")
print("  batch  games   rows    utility_mse  corrected_mse     utility_ev   corrected_ev")
batches = []
total_games = 0
for age in range(max(0, g - 8), g):
    try:
        if age not in by_age:
            raise ValueError("no train shard in checkpoint buffer")
        tables = train_tables(age)
        normal = []
        for r in by_age[age]:
            path = run / "search-values" / r["file"]
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != r["sha256"]:
                raise ValueError(f"shard hash mismatch: {path.name}")
            for row in json.loads(gzip.decompress(data)):
                if not tables[row["game_id"]]:
                    normal.append(row)
        games = Counter(r["game_id"] for r in normal)
        if not games or any(n != 64 for n in games.values()):
            raise ValueError("no complete normal games or incomplete game rows")
        main, delta = predict_rows(model, normal)
        batch = (np.asarray([r["u"] for r in normal]), main, main + delta)
        show(f"g{age+1}", len(games), *batch)
        batches.append(batch)
        total_games += len(games)
    except (OSError, KeyError, ValueError) as e:
        print(f"SKIP g{age+1}: {e}")

if batches:
    show("POOLED", total_games, *(np.concatenate(v) for v in zip(*batches)))
else:
    print("No normal train games could be scored.")
PY
```
