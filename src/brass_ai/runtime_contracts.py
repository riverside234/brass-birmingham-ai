"""Strict validation at the R1 environment/policy boundary (distinct from R0)."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

DIRECTORY = Path(__file__).resolve().parents[2] / "configs/contracts/v0.2.0"


@lru_cache(maxsize=None)
def validator(kind):
    raw = (DIRECTORY / "wire.schema.json").read_bytes()
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    if hashlib.sha256(raw).hexdigest() != manifest["contract_sha256"]:
        raise ValueError("R1 contract fingerprint mismatch")
    schema = json.loads(raw)
    Draft202012Validator.check_schema(schema)
    if kind not in schema["$defs"]:
        raise ValueError("Unknown R1 contract")
    return Draft202012Validator({**schema, "$ref": f"#/$defs/{kind}"})


def validate(kind, value):
    validator(kind).validate(value)
    if kind == "observation":
        if [p["seat"] for p in value["players"]] != list(range(4)) or sorted(value["order"]) != list(range(4)):
            raise ValueError("Four stable seats required")
        if len(value["own_hand"]) != value["players"][value["seat"]]["hand_count"]:
            raise ValueError("Private hand count inconsistent")
    if kind == "result":
        if [s["seat"] for s in value["seats"]] != list(range(4)) or abs(sum(s["utility"] for s in value["seats"]) - 1) > 1e-9:
            raise ValueError("Invalid four-seat result")
    return value


def policy_inputs(environment):
    """Called by trusted orchestration. Return only detached permitted inputs.

    Invoke policies with the returned view and choices, never with environment.
    Transport identity/revision is held separately by the coordinator.
    """
    context = validate("context", environment.call("context"))
    if context["terminal"]:
        raise ValueError("Terminal position has no policy decision")
    view = validate("observation", environment.observe(context["actor"]))
    choices = validate("choices", environment.choices())
    if view["context"]["revision"] != context["revision"]:
        raise ValueError("Observation became stale")
    return view, choices
