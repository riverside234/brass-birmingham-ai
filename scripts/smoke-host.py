"""Coverage smoke; deliberately unskilled legal play, never a strength benchmark."""
import argparse
from collections import Counter
import json
import random
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brass_ai.host import RulesHost

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=3)
parser.add_argument("--seed", type=int, default=1200)
parser.add_argument("--no-verify", action="store_true")
args = parser.parse_args()
reports = []
with RulesHost(verify=not args.no_verify) as host:
    for seed in range(args.seed, args.seed + args.games):
        rng = random.Random(seed + 987654)
        context = host.reset(seed)
        counts = Counter()
        start = time.monotonic()
        trace = []
        try:
            while not context["terminal"]:
                choices = host.choices()
                groups = {}
                for choice in choices:
                    groups.setdefault(choice["kind"], []).append(choice)
                # Exercise strategic and special choices, instead of sampling a huge
                # build Cartesian product so often that nothing can ever be sold.
                kinds = sorted(groups)
                weights = {"pass": .1, "scout": .2, "loan": 1, "build": 4, "network": 2,
                           "sell": 5, "develop": 1, "finish": 2}
                kind = rng.choices(kinds, [weights.get(k, 2) for k in kinds])[0]
                choice = rng.choice(groups[kind])
                trace.append(choice["id"])
                context = host.apply(context, choice)
                counts[kind] += 1
                if len(trace) > 2500:
                    raise RuntimeError("progress_limit")
            checkpoint = host.call("checkpoint")
            before = host.call("canonical_hash")
            host.call("restore", checkpoint=checkpoint)
            assert host.call("canonical_hash") == before
            report = {"seed": seed, "ok": True, "decisions": len(trace), "kinds": counts,
                      "seconds": round(time.monotonic() - start, 3), "result": host.call("result")}
        except Exception as error:
            report = {"seed": seed, "ok": False, "error": str(error), "context": context,
                      "last_choice": trace[-1] if trace else None, "decisions": len(trace), "kinds": counts}
            Path("artifacts/r1-host/failed-trace.json").write_text(json.dumps({"seed": seed, "trace": trace}, indent=2))
        reports.append(report)
        print(json.dumps(report), flush=True)
        if not report["ok"]:
            break
Path("artifacts/r1-host/smoke.json").write_text(json.dumps(reports, indent=2))
sys.exit(0 if all(r["ok"] for r in reports) else 1)
