"""V205 small scalar oracle, deliberately using complete checkpoint copies.

Production uses a bounded journal and moves an arena. This reference reconstructs
the eligible snapshot from immutable completed trajectories instead, so partial
work cannot enter the answer. Counts below the root are edge visits, not newly
completed root simulations. Run with --write to reproduce the pinned fixture.
"""
import copy
import json
from pathlib import Path
import sys


def reference():
    nodes={i:dict(count=[0,0],returns=[[0.0]*4 for _ in range(2)]) for i in range(6)}
    def backup(path,value,spent):
        for n,e,prior in path:
            nodes[n]['count'][e]+=1
            for s in range(4):nodes[n]['returns'][e][s]+=value-(spent-prior)
    for _ in range(3):backup([(0,0,0),(1,0,.01),(2,0,.02),(3,1,.03)],1,.05)
    completed=copy.deepcopy(nodes)
    backup([(0,0,0),(2,0,.02),(3,1,.03)],9,.05)  # unbalanced, never retained
    nodes=completed
    promoted=copy.deepcopy(nodes[2]);promoted['count']=[0,0];promoted['returns']=[[0.0]*4 for _ in range(2)]
    return dict(version='descendant-reuse-reference/1',watermark=3,promoted=promoted,
                descendant=nodes[3],retained_nodes=[2,3,4],old_root_visits_reset=3,inherited_edge_visits=3)


if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'GumbelISMCTS/fixtures/reuse-v1.json'
    result=reference()
    if '--write' in sys.argv:path.write_text(json.dumps(result,indent=2)+'\n')
    else:assert json.loads(path.read_text())==result
    print('Completed-watermark scalar reuse oracle passed')
