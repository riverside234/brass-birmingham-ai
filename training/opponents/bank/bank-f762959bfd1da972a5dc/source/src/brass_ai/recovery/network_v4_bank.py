"""Bounded v4 network extension; only a winning two-control screen admits it."""
from pathlib import Path
from brass_ai.bank import freeze, load_bank, read, sha, validate_admission
from brass_ai.learning.contracts import require
from brass_ai.step5.league import write_json
from .legacy_search import STYLES

CURRENT_V4='bank-3dc65253a610191d66a5'
ECONOMIC_V3='bank-f5107c644af2fbf4ac99'
CONTROLS=(CURRENT_V4, ECONOMIC_V3)
ADMISSION='masked-v4-network-smoke/1.0.0'
RECEIPT='training/opponents/evaluations/masked-v4-network-1.0.0/report.json'


def prepare(root, *, publish=False):
    root=Path(root).resolve()
    bank=freeze(root,[s+'-v3-masked-economy' for s in STYLES])
    manifest=load_bank(root,bank)
    original=load_bank(root,CURRENT_V4)['payload'];payload=manifest['payload']
    require(payload['profiles']==original['profiles'],'Preserve all six v4 styles and clock profiles')
    for path in ('src/BrassLegacyAI/Tactics.cs','src/BrassLegacyAI/CashManagement.cs','src/BrassLegacyAI/Conversion.cs',
                 'src/BrassMaskedSearch/TournamentOpenings.cs','src/BrassMaskedSearch/StrategicOpenings.cs','src/BrassMaskedSearch/EraEconomy.cs'):
        require(payload['source_files'][path]==original['source_files'][path],'Do not replace v4 strategy/intent/economic logic: '+path)
    require('src/BrassMaskedSearch/NetworkOptions.cs' in payload['runtime_source_paths'] and
            'src/BrassMaskedSearch/NetworkAndHand.cs' not in payload['runtime_source_paths'],'Exclude rejected Scout/brewery-demand code')
    record=dict(format='masked-v4-bank/1.1.0', generation='v4', bank=bank,
        manifest_sha256=sha(root/f'training/opponents/bank/{bank}/manifest.json'),
        backend='masked_ismcts', seconds=1., search_workers=4, previous_bank=CURRENT_V4,
        profiles=list(manifest['payload']['profiles']), comparison_controls=list(CONTROLS),
        productive_progress_fix=True, opening_candidate_bias=True, network_extension='bounded-network-options/1.0.0',
        strength_promoted=False, note='Same v4 strategies/budgets. Three bounded network features; no rejected Scout or speculative brewery-demand logic.')
    if publish:
        report=read(root/RECEIPT)
        require(report['format']==ADMISSION and report['passed'] and report['promote'] and
                report['new_bank']==bank and report['games']==24 and not report['failures'],
                'Complete and win the current-v4/economic-v3 screen before activation')
        record.update(qualification=RECEIPT, qualification_sha256=sha(root/RECEIPT),
                      strength_promoted=True, promotion_rule=report['selection_rule'])
        pool=dict(format='legacy-opponent-pool/1.0.0',generation='v4',backend='masked_ismcts',training={},evaluation={})
        for name in manifest['payload']['profiles']:
            purpose='evaluation' if name.startswith('pottery-flex') else 'training';identity=bank+'/'+name
            entry=dict(admission=ADMISSION,manifest_sha256=record['manifest_sha256'],qualification=RECEIPT,
                       sha256=record['qualification_sha256'],profile=name)
            validate_admission(root,identity,entry,purpose);pool[purpose][identity]=entry
        old=read(root/'training/opponents/pool.json');archive=root/'training/opponents/archive/pre-v4-network-pool.json'
        if old!=pool:
            if archive.exists(): require(read(archive)==old,'Preserve the selected v4 pool archive')
            else: write_json(archive,old)
        for relative,value in [('training/opponents/pool.json',pool),('training/opponents/current.json',record),
                               ('training/opponents/lab/current.json',record)]: write_json(root/relative,value)
    write_json(root/'artifacts/v4-network/candidate-bank.json',record)
    return record
