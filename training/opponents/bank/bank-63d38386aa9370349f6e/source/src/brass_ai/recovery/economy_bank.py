"""Win-focused v3 qualification; preserve both prior Masked versions as controls."""
from pathlib import Path
from brass_ai.bank import freeze, load_bank, read, sha, validate_admission
from brass_ai.learning.contracts import require
from brass_ai.step5.league import write_json
from .legacy_search import STYLES

CONTROLS = ('bank-3f7221cadcd472aafdb9', 'bank-340a20d72b23c652c267', 'bank-f5107c644af2fbf4ac99')
RECEIPT = 'training/opponents/evaluations/masked-economy-1.1.0/report.json'
ADMISSION = 'masked-economy-smoke/1.1.0'


def prepare(root, *, publish=False):
    root = Path(root).resolve()
    bank = freeze(root, [s+'-v3-masked-economy' for s in STYLES])
    manifest = load_bank(root, bank)
    record = dict(format='masked-v3-bank/1.3.0', bank=bank,
        manifest_sha256=sha(root/f'training/opponents/bank/{bank}/manifest.json'),
        seconds=1., search_workers=4, backend='masked_ismcts',
        profiles=list(manifest['payload']['profiles']), previous_bank=CONTROLS[1],
        comparison_controls=list(CONTROLS), source_book='bilibili-opening-families/1.0.0',
        strength='directional screen requiring first places, VP and rank against economic control; not established human strength')
    if publish:
        screen = read(root/RECEIPT)
        require(screen['format']==ADMISSION and screen['new_bank']==bank and screen['passed'] and screen['promote'] and screen['games']==36
                and not screen['failures'], 'Complete all 36 games and win more against EVERY control')
        record.update(qualification=RECEIPT, qualification_sha256=sha(root/RECEIPT),
                      smoke_winner=screen['winner'], promotion_rule=screen['selection_rule'])
        pool = dict(format='legacy-opponent-pool/1.0.0', backend='masked_ismcts', training={}, evaluation={})
        for name in manifest['payload']['profiles']:
            purpose = 'evaluation' if name.startswith('pottery-flex') else 'training'
            identity = bank+'/'+name
            entry = dict(admission=ADMISSION, manifest_sha256=record['manifest_sha256'],
                         qualification=RECEIPT, sha256=sha(root/RECEIPT), profile=name)
            validate_admission(root, identity, entry, purpose)
            pool[purpose][identity] = entry
        old = read(root/'training/opponents/pool.json')
        if old != pool:
            archive = root/'training/opponents/archive/pre-economy-pool.json'
            if archive.exists(): require(read(archive)==old, 'Preserve archived industry pool')
            else: write_json(archive, old)
        for path, value in [('training/opponents/pool.json', pool), ('training/opponents/current.json', record),
                            ('training/opponents/lab/current.json', record)]:
            write_json(root/path, value)
    write_json(root/'artifacts/economy-v3/candidate-bank.json', record)
    return record
