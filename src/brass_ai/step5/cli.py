from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brass_ai.learning.contracts import LAB_ROOT, project_path, read_json, require
from .assets import export_assets, import_runtime, preflight, require_preflight
from .league import train, validate_config
from .evaluation import evaluate, promote, validate_evaluation_config


def parser():
    p = argparse.ArgumentParser(description='Fresh four-player MLP-ResNet training with frozen v3 opponents and Masked ISMCTS NN distillation')
    p.add_argument('--root', type=Path, default=LAB_ROOT)
    p.add_argument('--config', help='Evaluation uses evaluation.json; recovery uses lab.json; historical recipes are archived')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--mono')
    sub = p.add_subparsers(dest='command', required=True)
    assets = sub.add_parser('assets'); assets.add_argument('--bundle', default='artifacts/step5-transfer/private-rules-runtime.zip')
    transfer = sub.add_parser('import-runtime'); transfer.add_argument('--bundle', required=True)
    check = sub.add_parser('preflight'); check.add_argument('--record-parity', action='store_true')
    run = sub.add_parser('train')
    run.add_argument('--run', default='artifacts/step5/selfplay')
    initialization = run.add_mutually_exclusive_group()
    initialization.add_argument('--initial')
    initialization.add_argument('--from-run', help='Rejected: only this fresh run can resume its own checkpoints')
    initialization.add_argument('--imitation-run', help='Completed fresh standalone imitate run; starts PPO with its selected weights and a new optimizer')
    run.add_argument('--iterations', type=int, default=1)
    run.add_argument('--demonstrations',help='Audited heuristic manifest for fresh in-run imitation and rehearsal')
    demos = sub.add_parser('demo-generate')
    demos.add_argument('--output',default='artifacts/demonstrations/v3')
    demos.add_argument('--games',type=int,default=1000)
    demos.add_argument('--seed',type=int,default=17100000)
    demos.add_argument('--partition',choices=['train','validation','test'])
    audit = sub.add_parser('demo-audit')
    audit.add_argument('--replays',nargs='+',required=True)
    audit.add_argument('--output',default='training/datasets/v3-demonstrations.json')
    bc = sub.add_parser('imitate')
    bc.add_argument('--manifest',default='training/datasets/v3-demonstrations.json')
    bc_initialization=bc.add_mutually_exclusive_group()
    bc_initialization.add_argument('--initial',help='Rejected: imitation starts from random residual weights, not arbitrary checkpoints')
    bc_initialization.add_argument('--from-imitation-run',help='Continue a completed fresh imitation model on this manifest in a new run directory')
    bc.add_argument('--run',default='artifacts/step5/imitation')
    bc.add_argument('--epochs',type=int,default=10)
    bc.add_argument('--min-epochs',type=int,default=3)
    bc.add_argument('--patience',type=int,default=2)
    bc.add_argument('--learning-rate',type=float,default=1e-4)
    bc.add_argument('--batch-size',type=int,help='Effective supervised batch; defaults to the selected config')
    bc.add_argument('--microbatch-size',type=int,help='Maximum positions per GPU forward/backward batch')
    bc.add_argument('--max-candidate-cells',type=int,help='Bound padded legal-action cells per microbatch')
    bc.add_argument('--workers',type=int,help='CPU replay/feature loader processes; defaults to audit_workers in config')
    bc.add_argument('--precision',choices=['fp32','bf16'],help='Training precision; BF16 requires a supported CUDA GPU')
    sub.add_parser('lab-bank-prepare')
    cleanup=sub.add_parser('cleanup');cleanup.add_argument('--run',required=True)
    cleanup.add_argument('--apply',action='store_true');cleanup.add_argument('--purge',action='store_true')
    restore=sub.add_parser('restore');restore.add_argument('--manifest',required=True)
    pin=sub.add_parser('pin');pin.add_argument('--run',required=True);pin.add_argument('--paths',nargs='+',required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--candidate', required=True)
    evaluation.add_argument('--references', nargs='+', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--setups', type=int)
    evaluation.add_argument('--seed', type=int)
    evaluation.add_argument('--partition', choices=['validation', 'test'])
    evaluation.add_argument('--include-legacy', action=argparse.BooleanOptionalAction, default=None)
    evaluation.add_argument('--search', action=argparse.BooleanOptionalAction, default=None)
    evaluation.add_argument('--focus')
    evaluation.add_argument('--lineups',nargs='+',choices=['recent-neural','mixed-neural','mixed-v3','heldout-pottery-v3'])
    promotion = sub.add_parser('promote')
    promotion.add_argument('--checkpoint', required=True); promotion.add_argument('--report', required=True)
    promotion.add_argument('--output', default='training/models/promoted.json')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    require(a.root.is_absolute(), 'Pass an absolute --root on this platform')
    root = a.root.resolve()
    require(a.device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    default_config = 'configs/step5/evaluation.json' if a.command == 'evaluate' else 'configs/step5/lab.json'
    config = read_json(project_path(root, a.config or default_config))
    if a.command == 'evaluate' and config.get('format','').startswith('step5-evaluation-config/'):
        validate_evaluation_config(config)
    else:
        validate_config(config)
    torch.set_num_threads(config['threads'])
    if a.command == 'assets':
        result = export_assets(root, bundle=a.bundle)
    elif a.command == 'import-runtime':
        result = import_runtime(root, a.bundle)
    elif a.command == 'preflight':
        result = preflight(root, device=a.device, mono=a.mono, record_parity=a.record_parity)
    elif a.command == 'promote':
        result = promote(root, a.checkpoint, a.report, a.output)
    elif a.command=='cleanup':
        from brass_ai.recovery.retention import prune,purge,active,owned
        from brass_ai.recovery.config import RETENTION
        _,run=owned(root,a.run);require(not active(run),'Cannot clean an active run')
        result=prune(root,run,config.get('retention',RETENTION),apply=a.apply)
        require(not a.purge or a.apply,'Purge requires --apply')
        if a.purge: result['purge']=purge(root,run,config.get('retention',RETENTION))
    elif a.command=='restore':
        from brass_ai.recovery.retention import restore
        result=restore(root,a.manifest)
    elif a.command=='pin':
        from brass_ai.recovery.retention import owned,safe,active
        from brass_ai.step5.league import write_json
        _,run=owned(root,a.run);require(not active(run),'Cannot change pins for an active run')
        paths=[safe(run,project_path(root,p)) for p in a.paths]
        require(all(p.is_file() for p in paths),'Pin files must exist inside the owned run')
        path=run/'pins.json';previous=read_json(path) if path.exists() else dict(paths=[])
        previous['paths']=sorted(set(previous['paths']+[p.relative_to(root).as_posix() for p in paths]))
        write_json(path,previous);result=previous
    elif a.command=='lab-bank-prepare':
        from brass_ai.recovery.era_bank import current
        result=current(root)
    else:
        require_preflight(root)
        if a.command == 'demo-generate':
            from brass_ai.recovery.demonstrations import generate
            from brass_ai.recovery.config import EXECUTION
            result=generate(root,a.output,a.games,config.get('execution',EXECUTION),seed=a.seed,mono=a.mono,partition=a.partition,budget=config.get('legacy_budget'))
            result={k:result[k] for k in ('format','games','eligible_for_bc','note')}
        elif a.command == 'demo-audit':
            from brass_ai.recovery.demonstrations import audit
            paths=[]
            for pattern in a.replays:
                path=project_path(root,pattern)
                paths.extend(root.glob(path.relative_to(root).as_posix()))
            require(paths,'No replay paths matched')
            manifest=audit(root,paths,a.output,mono=a.mono,workers=config.get('audit_workers',1),budget=config.get('legacy_budget'))
            result=dict(manifest=str(project_path(root,a.output)),partitions=manifest['partitions'],
                eligible_games=len(manifest['entries']),excluded=len(manifest['excluded']))
        elif a.command == 'imitate':
            from brass_ai.recovery.demonstrations import train_bc
            from brass_ai.recovery.config import UPDATE
            update=dict(config.get('update',UPDATE),value_weight=.5)
            for key,value in [('effective_batch_size',a.batch_size),('microbatch_size',a.microbatch_size),
                              ('max_candidate_cells',a.max_candidate_cells),('precision',a.precision)]:
                if value is not None:update[key]=value
            workers=a.workers if a.workers is not None else config.get('audit_workers',1)
            result=train_bc(root,a.run,a.manifest,a.initial,update,device=a.device,
                epochs=a.epochs,min_epochs=a.min_epochs,patience=a.patience,lr=a.learning_rate,retention=config.get('retention'),workers=workers,
                from_imitation_run=a.from_imitation_run)
            result={k:result[k] for k in ('candidate','candidate_sha256','epochs','promoted')}
        elif a.command == 'train':
            require(not (a.demonstrations and a.imitation_run),'Use --demonstrations OR --imitation-run')
            if a.demonstrations:
                config['imitation']=dict(config['imitation'],manifest=a.demonstrations)
                config['rehearsal_manifest']=a.demonstrations
            result = train(root, a.run, config, iterations=a.iterations, device=a.device, initial=a.initial,
                           mono=a.mono, from_run=a.from_run,imitation_run=a.imitation_run)
            result.pop('summaries', None)
        elif a.command == 'evaluate':
            result = evaluate(root, a.output, a.candidate, a.references, config, setups=a.setups, seed=a.seed,
                partition=a.partition, include_legacy=a.include_legacy, device=a.device, mono=a.mono,
                search=a.search, focus=a.focus,lineups=a.lineups)
            result = {k: result[k] for k in ('promotion_eligible', 'failed_games', 'candidate_sha256')}
    print(json.dumps(result, indent=2, allow_nan=False))
