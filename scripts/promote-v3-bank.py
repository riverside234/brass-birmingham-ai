"""Select a separately qualified v3 generation; retain previous pool snapshots."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.bank import LAB_ROOT, below, digest, load_bank, read, require, sha
from brass_ai.qualification import STYLES, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--report', required=True)
    args = parser.parse_args(); root = args.root.resolve()
    path = below(root, args.report); report = read(path)
    manifest = load_bank(root, report['bank']); profiles = manifest['payload']['profiles']
    names = {style+'-v3' for style in STYLES}
    require(set(profiles) == names and set(report['profiles']) == names, 'Expected six v3 S1 profiles')
    require(report['checks']['passed'] and report['checks']['bank'] == report['bank'] and report['failed_games'] == 0,
            'V3 correctness or match qualification failed')
    amendment=report.get('scope_amendment')
    if amendment:
        require(amendment['kind']=='user-requested-limited-validation' and amendment['requested_games']>=25
                and report['total_games']==amendment['retained_games']>=amendment['requested_games']
                and not report['complete_original_study'],'Incomplete user-accepted limited study')
    else: require(report['total_games'] == 118, 'Incomplete declared v3 study')
    pool = dict(format='legacy-pools/1.0.0', training={}, evaluation={})
    for name, profile in profiles.items():
        require(profile['mode']=='S1' and profile['cash_policy']==1 and profile['conversion_policy']==1,
                'Wrong v3 policy contract')
        result = report['profiles'][name]
        purpose = 'evaluation' if profile['tactic']=='pottery-flex' else 'training'
        require(result['niche_qualified'] and result[purpose+'_eligible'] and not result['failures']
                and result.get('economic_stalls') == [],
                'Unqualified v3 profile: '+name)
        identity = report['bank']+'/'+name
        require(result['policy_id']==identity, 'Qualification identity mismatch')
        pool[purpose][identity] = dict(qualification=path.relative_to(root).as_posix(),
                                      sha256=sha(path), profile=name)
    pool_path = root/'training/opponents/pool.json'
    old = read(pool_path)
    archive = root/'training/opponents/archive'/('pool-'+digest(old)[:20]+'.json')
    if archive.exists(): require(read(archive)==old, 'Pool archive changed')
    else: write_new(archive, old)
    current = dict(format='legacy-current-bank/1.0.0', generation=3, bank=report['bank'],
                   qualification=path.relative_to(root).as_posix(), qualification_sha256=sha(path),
                   previous_pool=archive.relative_to(root).as_posix(),
                   scope=('User-accepted limited validation; ' if amendment else '')+
                         'Four-player S1 niche opponents; Pottery Flex held out. Windows qualification only.',
                   scope_amendment=amendment)
    for target, value in ((pool_path,pool),(root/'training/opponents/current.json',current)):
        temporary = target.with_suffix('.new')
        temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        temporary.replace(target)
    print(json.dumps(current))


if __name__ == '__main__': main()
