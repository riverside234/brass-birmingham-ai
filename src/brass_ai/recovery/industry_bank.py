"""Freeze useful-development/group-sale v3; promote only on the requested screen."""
from pathlib import Path
from brass_ai.bank import freeze, load_bank, read, sha, validate_admission
from brass_ai.learning.contracts import require
from brass_ai.step5.league import write_json
from .legacy_search import STYLES

CURRENT_CONTROL = 'bank-3f7221cadcd472aafdb9'
S1_CONTROL = 'bank-d82438ccec287dbc8a64'
RECEIPT = 'training/opponents/evaluations/masked-industry-1.0.0/report.json'
ADMISSION = 'masked-industry-smoke/1.0.0'


def prepare(root, *, publish=False):
    root = Path(root).resolve()
    bank = freeze(root, [s + '-v3-masked-industry' for s in STYLES])
    manifest = load_bank(root, bank)
    record = dict(format='masked-v3-bank/1.2.0', bank=bank,
        manifest_sha256=sha(root/f'training/opponents/bank/{bank}/manifest.json'),
        seconds=1., search_workers=4, backend='masked_ismcts',
        profiles=list(manifest['payload']['profiles']), previous_bank=CURRENT_CONTROL,
        comparison_controls=[CURRENT_CONTROL, S1_CONTROL],
        strength='matched smoke only; no human-strength claim')
    if publish:
        screen = read(root/RECEIPT)
        require(screen['new_bank'] == bank and screen['passed'] and screen['promote']
                and screen['games'] == 24 and not screen['failures'],
                'Candidate must complete and improve a requested metric against BOTH controls')
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
            archive = root/'training/opponents/archive/pre-industry-pool.json'
            if archive.exists():
                require(read(archive) == old, 'Do not replace archived evidence')
            else:
                write_json(archive, old)
        for path, value in [('training/opponents/pool.json', pool),
                            ('training/opponents/current.json', record),
                            ('training/opponents/lab/current.json', record)]:
            write_json(root/path, value)
    write_json(root/'artifacts/industry-v3/candidate-bank.json', record)
    return record
