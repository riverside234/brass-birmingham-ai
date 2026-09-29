"""Author the R1 wire schema. R0's 0.1.0 files remain immutable."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "configs/contracts/v0.2.0"

def obj(properties, optional=()):
    return {"type": "object", "properties": properties, "required": [k for k in properties if k not in optional], "additionalProperties": False}

def arr(items, count=None):
    return {"type": "array", "items": items, **({"minItems": count, "maxItems": count} if count is not None else {})}

def ref(name): return {"$ref": f"#/$defs/{name}"}
def enum(*values): return {"enum": list(values)}
def integer(low=0, high=None): return {"type": "integer", "minimum": low, **({"maximum": high} if high is not None else {})}

string = {"type": "string"}
boolean = {"type": "boolean"}
seat = integer(0, 3)
common = {"protocol": {"const": "0.2.0"}, "rules": {"const": "steam-brass-patch-1.1.0"}}
types = enum("Brewery", "CoalMine", "CottonMill", "IronWorks", "Manufacturer", "Pottery")
era = enum("CanalEra", "RailEra")
card = {"type": "string", "pattern": r"^(location:[A-Za-z]+|industry:[A-Za-z]+(\+[A-Za-z]+)*|wild:(industry|location))$"}
defs = {}
context = {**common, "revision": integer(), "actor": integer(-1, 3), "turn_owner": seat,
           "phase": enum("action", "payment", "continuation", "bonus", "debt", "terminal"),
           "actions_remaining": integer(0, 2), "era": era, "round": integer(1, 8), "turn": integer(1, 4), "terminal": boolean}
defs["public_context"] = obj(context)
defs["context"] = obj({**context, "game_id": {"type": "string", "pattern": "^[a-f0-9]{32}$"}})
choices = []
for kinds, fields, optional in [
    (("pass", "loan"), {"card": card}, ()),
    (("build",), {"card": card, "target": string, "industry": types}, ()),
    (("develop",), {"card": card, "industry": types}, ("card",)),
    (("network",), {"card": card, "target": string}, ("card",)),
    (("sell",), {"card": card, "target": string, "merchant": string}, ("card",)),
    (("scout",), {"card": card, "cards": arr(card, 2)}, ()),
    (("coal", "iron", "beer", "liquidate"), {"target": string}, ()),
    (("finish", "bonus_skip"), {}, ()),
    (("bonus_develop",), {"industry": types}, ()),
]:
    choices.append(obj({"id": string, "kind": enum(*kinds), **fields}, optional))
defs["choice"] = {"oneOf": choices}
defs["choices"] = arr(ref("choice"))
defs["tile"] = obj({"id": string, "seat": seat, "type": types, "level": integer(1, 8), "flipped": boolean,
                    **{k: integer() for k in ("resource", "vp", "link_vp", "income", "cost", "coal", "iron", "beer")},
                    "developable": boolean, "canal_only": boolean, "rail_only": boolean})
defs["history_entry"] = obj({"seat": seat, "era": era, "choice": ref("choice")})
defs["observation"] = obj({**common, "seat": seat, "context": ref("public_context"), "own_hand": arr(card),
    "players": arr(obj({"seat": seat, "cash": integer(), "income_marker": integer(), "income": integer(-10, 30),
        "vp": integer(), "bonus_vp": integer(), "link_tiles": integer(0, 14), "hand_count": integer(0, 8),
        "spent": integer(), "discards": arr(card), "hidden_discards": integer(0, 1),
        "known_wild": arr(enum("wild:industry", "wild:location")),
        "mat": arr(obj({"type": types, "tiles": arr(ref("tile"))}), 6), "debt": integer()}), 4),
    "order": arr(seat, 4), "slots": arr(obj({"id": string, "types": arr(types), "tile": ref("tile")}, ("tile",))),
    "links": arr(obj({"id": string, "owner": seat, "regions": arr(string), "era": enum("Canal", "Rail")}, ("owner",))),
    "merchants": arr(obj({"id": string, "types": arr(types), "beer": integer(0, 1), "bonus": enum("Develop", "VP", "Income", "Money")}, ("types",))),
    "coal_market": integer(0, 14), "iron_market": integer(0, 10), "deck_count": integer(0, 64),
    "wild_industry_count": integer(0, 4), "wild_location_count": integer(0, 4),
    "pending": obj({"kind": enum("", "build", "network", "sell", "develop", "loan", "pass", "scout"), "target": string,
                    "merchant": string, "parts": integer(), "coal": integer(), "iron": integer(), "beer": integer()}),
    "history": arr(ref("history_entry"))})
defs["result"] = obj({"objective": {"const": "win-share/1.0.0"}, "seats": arr(obj({"seat": seat, "vp": integer(),
    "income": integer(-10, 30), "cash": integer(), "rank": integer(1, 4), "utility": {"type": "number", "minimum": 0, "maximum": 1}}), 4)})
defs["cards"] = obj({"hands": arr(arr(card), 4), "discards": arr(arr(card), 4), "discard_starts": arr(integer(), 4),
                     "deck": arr(card), "wild_industry": arr(enum("wild:industry")), "wild_location": arr(enum("wild:location"))})
defs["setup"] = obj({"cards": ref("cards"), "order": arr(seat, 4),
    "merchants": arr(obj({"id": string, "types": {"anyOf": [arr(types), {"type": "null"}]}})), "position": ref("observation")}, ("position",))
defs["checkpoint"] = obj({**common, "format": {"const": "brass-trusted-checkpoint"}, "game_id": string,
    "seed": {"type": "integer", "minimum": -2147483648, "maximum": 2147483647}, "setup": ref("setup"),
    "commands": arr(obj({"choice": string, "cards_after": ref("cards")})), "hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"}})
document = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "urn:brass-ai:runtime:0.2.0",
            "title": "Four-player R1 host contracts; trusted checkpoints are not policy observations", "$defs": defs}
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "wire.schema.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
digest = hashlib.sha256((DEST / "wire.schema.json").read_bytes()).hexdigest()
(DEST / "manifest.json").write_text(json.dumps({"contract_version": "0.2.0", "contract_sha256": digest,
    "rules_profile_id": "steam-brass-patch-1.1.0", "player_count": 4, "action_schema": "phase-choice/0.2.0",
    "observation_schema": "public-state/0.2.0", "objective": "win-share/1.0.0",
    "compatibility": "R0 0.1.0 remains a reference prototype. Runtime 0.2.0 is a separate contract; do not mix datasets.",
    "checkpoint_access": "trusted environment only", "chance_reproduction": "explicit setup and per-command card outcomes"}, indent=2) + "\n", encoding="utf-8")
print(digest)
