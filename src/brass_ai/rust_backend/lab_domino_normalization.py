"""Causal phase denominator proof over complete, B02-parity-bound replay."""
import numpy as np
from brass_ai.learning.contracts import require,digest
from brass_ai.recovery.domino_features import event,occupancy,width
from brass_ai.recovery.teacher_diversity import COUNTS,NORMALIZATION


def prove(game):
    require(game.get('result') is not None and not game.get('branch'),'Normalization requires a complete standard parent game')
    seats=[]
    for seat in range(4):
        rows=[r for r in game['records'] if r['seat']==seat]
        events=[event(r) for r in rows]
        ordinary,counts=occupancy(events)
        require(counts==list(COUNTS),'Unsupported truncated/variant DOMiNO phase counts: '+str(counts))
        causal=np.zeros((4,width()),dtype=np.float64)
        for e in events:
            if e is not None:causal[e['phase']]+=np.asarray(e['x'])/(4*COUNTS[e['phase']])
        require(np.array_equal(ordinary,causal.reshape(-1)),'Causal and ordinary occupancy normalization differ')
        seats.append(dict(seat=seat,counts=counts,occupancy_sha256=digest(ordinary.tolist())))
    return dict(contract=NORMALIZATION,game_id=game['game_id'],seats=seats,passed=True,
                provenance='Complete parent from sealed B02 native/reference parity; forced actions retained, subdecisions excluded')
