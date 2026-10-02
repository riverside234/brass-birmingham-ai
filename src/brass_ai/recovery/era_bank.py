"""Freeze the era-aware heuristic bank without changing historical evidence."""
from pathlib import Path
from brass_ai.bank import freeze,load_bank,read,sha,validate_admission
from brass_ai.learning.contracts import require
from brass_ai.step5.league import write_json
from .legacy_search import STYLES

PREVIOUS_BANK='bank-981744ddae4aaf07f1cf'
RECEIPT='training/opponents/evaluations/masked-era-1.3.0/report.json'


def current(root):
    root=Path(root).resolve();record=read(root/'training/opponents/current.json')
    require(record['format'] in ('masked-v3-bank/1.2.0','masked-v3-bank/1.3.0','masked-v4-bank/1.0.0','masked-v4-bank/1.1.0'),'Import the selected Masked distribution')
    load_bank(root,record['bank']);pool=read(root/'training/opponents/pool.json')
    for purpose in ('training','evaluation'):
        for identity,entry in pool[purpose].items():
            require(identity.startswith(record['bank']+'/'),'Pool/current identity differs')
            validate_admission(root,identity,entry,purpose)
    return record


def prepare(root, *, publish=False):
    root=Path(root).resolve()
    bank=freeze(root,[s+'-v3-masked-era' for s in STYLES])
    manifest=load_bank(root,bank)
    record=dict(format='masked-v3-bank/1.2.0',bank=bank,
        manifest_sha256=sha(root/f'training/opponents/bank/{bank}/manifest.json'),
        seconds=1.,search_workers=4,backend='masked_ismcts',profiles=list(manifest['payload']['profiles']),
        previous_bank=PREVIOUS_BANK,strength='eight-game-smoke; no human-strength claim')
    if publish:
        screen=read(root/RECEIPT)
        require(screen['new_bank']==bank and screen['passed'] and screen['games']==8 and not screen['failures'],'Complete the matched smoke before admission')
        record.update(qualification=RECEIPT,qualification_sha256=sha(root/RECEIPT),smoke_winner=screen['winner'])
        pool=dict(format='legacy-opponent-pool/1.0.0',backend='masked_ismcts',training={},evaluation={})
        for name in manifest['payload']['profiles']:
            purpose='evaluation' if name.startswith('pottery-flex') else 'training'
            identity=bank+'/'+name
            entry=dict(admission='masked-era-smoke/1.0.0',manifest_sha256=record['manifest_sha256'],
                qualification=RECEIPT,sha256=sha(root/RECEIPT),profile=name)
            validate_admission(root,identity,entry,purpose);pool[purpose][identity]=entry
        archive=root/'training/opponents/archive/pre-era-pool.json'
        if not archive.exists():write_json(archive,read(root/'training/opponents/pool.json'))
        current=read(root/'training/opponents/pool.json')
        if any(p.startswith('bank-e809802ae27fe9f0ee14/') for p in current['training']):
            preserved=root/'training/opponents/archive/pre-era-r2-pool.json'
            if not preserved.exists():write_json(preserved,current)
        if any(p.startswith('bank-e18172c1ffee94fc8bca/') for p in current['training']):
            preserved=root/'training/opponents/archive/pre-era-r3-pool.json'
            if not preserved.exists():write_json(preserved,current)
        for path,value in [('training/opponents/pool.json',pool),('training/opponents/current.json',record),
                           ('training/opponents/lab/current.json',record)]:write_json(root/path,value)
    write_json(root/'artifacts/rail-era/candidate-bank.json',record)
    return record
