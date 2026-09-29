"""Extract only public card counts. Game implementation/assets remain local."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brass_ai.host import RulesHost

with RulesHost() as host:
    host.reset(0)
    cards = host.call("checkpoint")["setup"]["cards"]
    catalog = Counter(cards["deck"])
    for group in cards["hands"] + cards["discards"]:
        catalog.update(group)
    dest = Path("configs/contracts/v0.2.0/card-composition.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(sorted(catalog.items())), indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {sum(catalog.values())} normal cards across {len(catalog)} semantic types")
