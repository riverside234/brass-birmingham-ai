"""Long-trace replay, phase restoration, sampled-world privacy and parity inputs."""
import json
import hashlib
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brass_ai.host import RulesHost
from brass_ai.beliefs import sample_world_setup

composition = json.loads((ROOT / "configs/contracts/v0.2.0/card-composition.json").read_text())
out = ROOT / "artifacts/r1-host"
reports, checkpoints = [], []
coverage = set()
privacy_cases = 0
sampled_continuations = 0
started = time.monotonic()
host_sha = hashlib.sha256((out / "guarded/BrassHost.exe").read_bytes()).hexdigest()
with RulesHost() as host, RulesHost() as branch:
    for seed in (1200, 1201, 1202, 1300, 1301):
        rng = random.Random(seed + 987654)
        ctx = host.reset(seed)
        count = 0
        seen = set()
        while not ctx["terminal"]:
            key = (ctx["era"], ctx["phase"])
            if key not in seen:
                seen.add(key)
                cp = host.call("checkpoint")
                checkpoints.append({"label": f"{seed}:{key[0]}:{key[1]}", "checkpoint": cp})
                branch.call("restore", checkpoint=cp)
                assert branch.call("canonical_hash") == cp["hash"]
                assert branch.choices() == host.choices()
                # Test all four information boundaries at every distinct era/phase.
                for seat in range(4):
                    view = host.observe(seat)
                    for sample in (991, 1133):
                        setup = sample_world_setup(view, composition, sample)
                        branch_context = branch.reset(sample, setup)
                        assert branch.observe(seat) == view, (seed, key, seat, "view")
                        if seat == ctx["actor"]:
                            assert branch.choices() == host.choices(), (seed, key, "choices")
                        privacy_cases += 1
                        # Reconstructed private action/bonus/debt fields must support
                        # real continuation, not just render the same observation.
                        for step in range(128):
                            branch_choices = branch.choices()
                            selected = next((c for c in branch_choices if c["kind"] in ("finish", "bonus_skip")), branch_choices[0])
                            branch_context = branch.apply(branch_context, selected)
                            if branch_context["terminal"] or branch_context["phase"] == "action":
                                sampled_continuations += 1
                                break
                        else:
                            raise AssertionError("Sampled continuation failed to settle")
                assert host.call("canonical_hash") == cp["hash"], "A branch changed its source"
            choices = host.choices()
            groups = {}
            for choice in choices:
                groups.setdefault(choice["kind"], []).append(choice)
            kinds = sorted(groups)
            weights = {"pass": .1, "scout": .2, "loan": 1, "build": 4, "network": 2,
                       "sell": 5, "develop": 1, "finish": 2}
            kind = rng.choices(kinds, [weights.get(k, 2) for k in kinds])[0]
            choice = rng.choice(groups[kind])
            coverage.add(kind)
            ctx = host.apply(ctx, choice)
            count += 1
            assert count < 2000
        cp = host.call("checkpoint")
        checkpoints.append({"label": f"{seed}:terminal", "checkpoint": cp})
        branch.call("restore", checkpoint=cp)
        assert branch.call("canonical_hash") == cp["hash"]
        reports.append({"seed": seed, "decisions": count, "final_hash": cp["hash"],
                        "phases": sorted([list(p) for p in seen]), "result": host.call("result")})
        print(json.dumps({"seed": seed, "decisions": count, "privacy_cases_so_far": privacy_cases}), flush=True)
report = {"passed": True, "games": reports, "privacy_cases": privacy_cases, "covered_choice_kinds": sorted(coverage),
          "host_sha256": host_sha, "sampled_continuations": sampled_continuations, "restore_points": len(checkpoints), "seconds": round(time.monotonic() - started, 3)}
assert hashlib.sha256((out / "guarded/BrassHost.exe").read_bytes()).hexdigest() == host_sha, "Host changed during validation"
(out / "traces.json").write_text(json.dumps(checkpoints), encoding="utf-8")
(out / "trace-checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "games"}))
