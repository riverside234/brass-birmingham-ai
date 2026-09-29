"""Offline R0 checks, with explicit optional reference downloads / game reads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brass_ai.contracts import Contracts, ContractError, canonical_digest, read_json
from brass_ai.references import LOCK_PATH, confined, validate_lock, verify_rules_files, verify_sources


def verify_project():
    contracts = Contracts()
    lock = validate_lock(read_json(LOCK_PATH))
    mapping = read_json(ROOT / "configs/references/components.json")
    manifest = read_json(ROOT / "configs/contracts/v0.1.0/manifest.json")
    expected = {
        "format": "brass-ai-contract-manifest", "format_version": 1,
        "compatibility": contracts.compatibility,
        "reference_lock_sha256": canonical_digest(lock),
        "component_map_sha256": canonical_digest(mapping),
    }
    if manifest != expected:
        raise ContractError("R0 manifest mismatch: review changed contracts/pins/map and version deliberately")
    references = {r["id"]: r for r in lock["references"]}
    component_ids = set()
    categories = set()
    for component in mapping["components"]:
        if component["id"] in component_ids:
            raise ContractError("duplicate component ID")
        component_ids.add(component["id"])
        reference = references[component["reference"]]
        if component["source_path"] not in {item["path"] for item in reference["files"]}:
            raise ContractError("component source is not pinned")
        if not component["contracts"] or any(name not in contracts.schema["$defs"] for name in component["contracts"]):
            raise ContractError("component refers to an unknown project contract")
        if component["reuse_status"] != "reference_only":
            raise ContractError("R0 does not vendor adapted upstream source")
        categories.update(component["assumption_categories"])
    if not {"scalar_value", "seat_index", "hidden_state", "two_player"}.issubset(categories):
        raise ContractError("required adaptation assumptions are missing")
    examples = read_json(ROOT / "tests/R0/fixtures/messages.json")
    kinds = set()
    for message in examples:
        contracts.validate(message)
        kinds.add(message["kind"])
    required_kinds = {entry["$ref"].rsplit("/", 1)[-1] for entry in contracts.schema["oneOf"]}
    if kinds != required_kinds:
        raise ContractError("each top-level message contract needs a valid example")
    for message in examples:
        if message["kind"] == "model_metadata" and message["reference_lock_sha256"] != manifest["reference_lock_sha256"]:
            raise ContractError("example model metadata has different source pins")
    requests = {m["request_id"]: m for m in examples if m["kind"] == "decision_request"}
    for message in examples:
        if message["kind"] == "decision_response":
            contracts.validate_response(requests[message["request_id"]], message)
    for evidence in contracts.rules["evidence"]:
        if not confined(ROOT, evidence).is_file():
            raise ContractError(f"missing declared rules evidence: {evidence}")
    return contracts, lock, len(component_ids), len(examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-cache", action="store_true", help="verify every locally cached upstream file")
    parser.add_argument("--fetch-references", action="store_true", help="download missing pinned source files into ignored artifacts")
    parser.add_argument("--game-root", type=Path, help="read and compare the installed rules binary pair")
    args = parser.parse_args()
    try:
        contracts, lock, components, examples = verify_project()
        print(f"R0 metadata/contracts: OK; {components} mapped components, {examples} valid message examples")
        if args.check_cache or args.fetch_references:
            count = verify_sources(lock, fetch=args.fetch_references)
            print(f"Pinned source cache: OK; {count} SHA-256/Git-blob checks")
        if args.game_root is not None:
            count = verify_rules_files(contracts.rules, args.game_root)
            print(f"Installed rules pair: OK; {count} files match {contracts.rules['id']} (R1 certification still pending)")
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"R0 verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
