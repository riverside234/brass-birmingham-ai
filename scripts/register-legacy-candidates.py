"""Record content-addressed candidates, without promoting them into H5 training pools."""
import argparse
import hashlib
import json
from pathlib import Path

LAB_ROOT=Path('/data/not_backed_up/yxu209/ai-training/')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);args=p.parse_args();root=args.root.resolve()
    runtime=json.loads((root/'artifacts/legacy-ai/runtime/runtime.json').read_text(encoding='utf-8-sig'))
    core={str(f.relative_to(root)).replace('\\','/'):sha(f) for f in sorted((root/'src/BrassLegacyAI').glob('*.cs'))}
    policy={k:v for k,v in core.items() if k.rsplit('/',1)[-1] in ('Belief.cs','Evaluator.cs','Planner.cs')}
    contracts={str(f.relative_to(root)).replace('\\','/'):sha(f) for f in sorted((root/'configs/contracts/v0.2.0').glob('*.json'))}
    common=dict(player_count=4,rules='steam-brass-patch-1.1.0',observation_schema='0.2.0',action_schema='0.2.0',
                contract_files=contracts,policy_source_files=policy,policy_sha256=canonical(policy),source_files=core,
                dependencies=runtime['files'],runtime_sources=runtime['sources'],
                tie_break='ordinal semantic choice IDs; independent xorshift belief seed 1; no real engine seed',
                benchmark_eligible=False,training_eligible=False,qualification='candidate; H5 not passed')
    entries=[]
    for name in ('balanced-l1-v1','balanced-v1','rail-tempo-v1'):
        path=root/'configs/legacy-ai'/f'{name}.json';config=json.loads(path.read_text())
        identity=canonical(dict(policy=policy,profile=config,contracts=contracts,dependencies=runtime['files'],fixed_work=False))
        entry=dict(common,**dict(id=config['id']+'@'+identity[:12],profile_path=f'configs/legacy-ai/{name}.json',
                                profile_sha256=sha(path),settings=config,fixed_work=False))
        entries.append(entry)
    stock=[]
    for name in ('stock-hard-original','stock-hard-guarded-1.1.0'):
        directory=root/'artifacts/h0'/name
        files=[dict(name=f.name,sha256=sha(f)) for f in sorted(directory.glob('*.dll'))]
        stock.append(dict(id=name,player_count=4,dependencies=files,selector_calls_per_action=1,
                          fairness='historical vendor policy; not a strict-observation policy',
                          rules='historical-original' if name.endswith('original') else 'steam-brass-patch-1.1.0',
                          benchmark_eligible=True,training_eligible=False,purpose='historical comparison only; H5 training admission is separate'))
    registry=dict(format='legacy-opponent-registry/0.1.0',player_count=4,lab_root=LAB_ROOT.as_posix().rstrip('/')+'/',
                  release_policy='Content-addressed candidate records. Never mutate an admitted H5 opponent; append a new version.',
                  candidates=entries,baselines=stock)
    destination=root/'training/opponents/registry.json';destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and json.loads(destination.read_text()).get('banks'):
        raise ValueError('Frozen bank registry exists. Use freeze-legacy-bank.py and qualify-legacy-bank.py; never overwrite admitted versions.')
    destination.write_text(json.dumps(registry,indent=2)+'\n');print(destination)

if __name__=='__main__':main()
