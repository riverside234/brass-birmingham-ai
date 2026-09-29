"""Verify current R1 evidence/pins and write a compact, non-private exit summary.

Run build-host, check-r1-traces, test-r1, check-r1-runtime (both platforms) and
capture-h0 first. This command does not silently regenerate failed evidence.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brass_ai.host import RulesHost
from brass_ai.runtime_contracts import validator


def read(path): return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))
def sha(path): return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, default=ROOT)
    args = parser.parse_args()
    validator("observation")
    runtime = read("artifacts/r1-host/guarded/runtime.json")
    for entry in runtime["sources"]:
        assert sha("src/BrassHost/" + entry["name"]) == entry["sha256"], "Host source changed since build"
    with RulesHost() as host:
        host.reset(1)
    digest = sha("artifacts/r1-host/guarded/BrassHost.exe")
    units = read("artifacts/r1-host/unit-checks.json")
    traces = read("artifacts/r1-host/trace-checks.json")
    dotnet = read("artifacts/r1-host/dotnet-result.json")
    unity_wrapper = read("artifacts/r1-unity/unity-result.json")
    assert unity_wrapper["passed"], "Unity validation failed"
    unity = unity_wrapper["result"]
    startup = read("artifacts/r1-unity/BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll.startup-hook.json")
    assert startup["only_two_startup_methods_changed"]
    assert startup["input_sha256"] == sha("artifacts/r1-host/guarded/Assembly-CSharp.dll")
    assert startup["output_sha256"] == sha("artifacts/r1-unity/BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll")
    for report in (units, traces, dotnet, unity):
        assert report["passed"] and report["host_sha256"] == digest, "Failed/stale validation evidence"
    trace_sha = sha("artifacts/r1-host/traces.json")
    assert dotnet["trace_sha256"] == unity["trace_sha256"] == trace_sha
    assert traces["privacy_cases"] == traces["sampled_continuations"]
    h0 = []
    for identity, expected in (("stock-hard-original", "600ebf309670b2726001c16a3375f8bbf3de76a23c0202be7cf198ba338981fb"),
                               ("stock-hard-guarded-1.1.0", "1c8cef281b81474200268eed23448faa3f5247f8ea49409a69efc3982c5a2d0a")):
        path = "artifacts/h0/" + identity
        capture = read(path + "/capture.json")
        plain = read(path + "/capture-plain.json")["games"][0]
        assert capture["assembly_sha256"] == expected
        instrumented = next(g for g in capture["games"] if g["seed"] == plain["seed"])
        for key in ("initial", "scores", "ranking", "completed", "failure", "actions", "turns", "debts"):
            assert instrumented[key] == plain[key], (identity, "instrumentation changed", key)
        def decision_trace(game):
            return [{k: t[k] for k in ("seat", "era", "round", "action", "events")} for t in game["trace"]]
        assert decision_trace(instrumented) == decision_trace(plain), "Logging changed a stock decision"
        for game in capture["games"]:
            assert game["player_count"] == 4 and all(t["selected_rule"] for t in game["trace"])
        h0.append({"identity": identity, "assembly_sha256": expected, "games": len(capture["games"]),
                   "completed": sum(g["completed"] for g in capture["games"]),
                   "failures": sum(g["failure"] is not None for g in capture["games"]),
                   "logging_control_seed": plain["seed"], "logging_noninterference": True,
                   "capture_sha256": sha(path + "/capture.json")})
    profile = read("configs/rules/steam-brass-patch-1.1.0.json")
    for f in profile["files"]:
        assert hashlib.sha256((args.game_root / f["path"]).read_bytes()).hexdigest() == f["sha256"], "Installed patch changed"
    for path in ("artifacts/r1-host/guarded/BrassHost.exe", "artifacts/r1-host/traces.json",
                 "artifacts/r1-unity/BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll",
                 "artifacts/h0/stock-hard-original/capture.json", profile["files"][0]["path"]):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0, "Generated/game file not ignored"
    summary = {"milestone": "R1", "passed": True, "contract": "0.2.0", "rules": profile["id"],
               "host_sha256": digest, "managed_dependencies": len(read("configs/rules/runtime-dependencies-1.1.0.json")["files"]),
               "unit_tests": units["tests"], "full_games": len(traces["games"]),
               "decisions_in_full_games": sum(g["decisions"] for g in traces["games"]),
               "restore_points": traces["restore_points"], "privacy_cases": traces["privacy_cases"],
               "sampled_continuations": traces["sampled_continuations"], "covered_choice_kinds": traces["covered_choice_kinds"],
               "dotnet": dotnet, "unity": unity_wrapper, "h0": h0, "installed_patch_unchanged": True,
               "limits": ["Windows .NET Framework/Unity Mono only; lab/Linux unvalidated", "finite fixtures, not exhaustive full-game proof",
                          "R2/R3 training/search/live replacement not implemented", "H0 candidate-alternative diagnosis and fairness conversion remain"]}
    (ROOT / "artifacts/r1-host/exit-checks.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
