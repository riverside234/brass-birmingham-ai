"""Run exactly 20 four-player v3-versus-stock-Hard headless games.

The frozen research bank supplies v3 behavior; only its live-play time and
work limits are raised. Full game traces stay under ignored artifacts/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.bank import FrozenOpponent, digest, load_bank, read, require, sha
from brass_ai.legacy import LAB_ROOT
from brass_ai.qualification import grouped_interval, write_new


STYLES = ('balanced', 'rail-tempo', 'cotton', 'goods-beer', 'supplier', 'pottery-flex')
SEEDS = (101001, 101002, 101003, 101004, 101005)
STOCK = 'stock-hard-guarded-1.1.0'
PUBLIC = 'training/opponents/evaluations/v3-vs-hard-20'
PRIVATE = 'artifacts/legacy-ai/v3-vs-hard-20'
INSTALLED_DLL = 'BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll'


def prepare(root: Path) -> dict:
    current = read(root / 'training/opponents/current.json')
    require(current['generation'] == 3, 'Current research bank is not v3')
    bank = current['bank']
    manifest = load_bank(root, bank)
    profiles = manifest['payload']['profiles']
    require(all(style + '-v3' in profiles for style in STYLES), 'Incomplete six-style v3 bank')
    require(all(profiles[style + '-v3']['cash_policy'] == 1 and
                profiles[style + '-v3']['conversion_policy'] == 1 for style in STYLES),
            'v3 cash/conversion policies are not enabled')
    installed = root / INSTALLED_DLL
    require(installed.is_file(), 'Installed game AI DLL was not found')
    jobs = []
    for seed_index, seed in enumerate(SEEDS):
        for seat in range(4):
            style = STYLES[(seed_index + seat) % len(STYLES)]
            jobs.append(dict(id=f'{seed}-seat-{seat}', seed=seed, seat=seat,
                             profile=style + '-v3'))
    require(len(jobs) == 20 and len({j['id'] for j in jobs}) == 20,
            'Benchmark must contain exactly 20 unique games')
    payload = dict(format='v3-vs-stock-hard-spec/1.0.0', player_count=4,
                   bank=bank, bank_manifest_sha256=sha(root / 'training/opponents/bank' / bank / 'manifest.json'),
                   frozen_worker_sha256=next(f['sha256'] for f in manifest['payload']['runtime_files']
                                             if f['name'] == 'BrassLegacyAI.exe'),
                   installed_game_ai_dll_sha256=sha(installed),
                   installed_game_package='brass-local-ai/1.4.1',
                   original_hard=STOCK, milliseconds=10000, work_limit=1024,
                   fixed_work=False, jobs=jobs,
                   protocol='One v3 seat versus three stock Hard seats; five seeds, all four seats per seed. Run every scheduled game once. No result-based stopping or retuning.',
                   scope='Frozen headless four-player rules-1.1.0 proxy; installed game uses package 1.4.1 and separate rules assemblies.')
    return dict(id='v3-vs-hard-' + digest(payload)[:20], payload=payload)


def run(root: Path, spec: dict, workers: int) -> None:
    public = root / PUBLIC
    private = root / PRIVATE
    require(spec == prepare(root), 'Bank or installed package changed after benchmark setup')
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    spec_path = public / 'spec.json'
    if spec_path.exists():
        require(read(spec_path) == spec, 'Existing benchmark spec differs')
    else:
        write_new(spec_path, spec)
    manifest = load_bank(root, spec['payload']['bank'])
    expected_count = len(spec['payload']['jobs'])

    def execute(job: dict) -> dict:
        path = private / (job['id'] + '.json')
        if path.exists():
            record = read(path)
            require(record['spec_id'] == spec['id'] and record['job'] == job,
                    'Saved game belongs to another benchmark')
            return record
        profile = dict(manifest['payload']['profiles'][job['profile']])
        profile['milliseconds'] = spec['payload']['milliseconds']
        profile['work_limit'] = spec['payload']['work_limit']
        table = [None] * 4
        table[job['seat']] = profile
        began = time.perf_counter()
        opponent = None
        try:
            opponent = FrozenOpponent(root, spec['payload']['bank'] + '/' + job['profile'],
                                      purpose='diagnostic')
            match = opponent.worker.call('run', seed=job['seed'], player_count=4,
                                         profile=profile, table_profiles=table,
                                         fixed_work=False, record_selectors=True)
            require(match['completed'] and not match.get('failure'),
                    'Game did not complete: ' + str(match.get('failure')))
            require(match['seed'] == job['seed'] and match['player_count'] == 4 and
                    match['table_profiles'] == [profile['id'] if i == job['seat'] else STOCK
                                                for i in range(4)], 'Incorrect matchup')
            require(match['turns'] == 64 and match['actions'] == 124 and
                    len(match['result']['seats']) == 4 and
                    match['result']['objective'] == 'win-share/1.0.0',
                    'Incomplete four-player game')
            failure = None
        except Exception as error:
            match = None
            failure = str(error)
        finally:
            if opponent is not None:
                opponent.close()
        record = dict(spec_id=spec['id'], job=job,
                      elapsed_seconds=time.perf_counter() - began,
                      completed=failure is None, failure=failure, match=match)
        write_new(path, record)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, job) for job in spec['payload']['jobs']]
        for number, future in enumerate(as_completed(futures), 1):
            record = future.result()
            print(json.dumps(dict(progress=f'{number}/{expected_count}',
                                  job=record['job']['id'], profile=record['job']['profile'],
                                  completed=record['completed'], failure=record['failure'],
                                  seconds=round(record['elapsed_seconds'], 1))), flush=True)
    require(sha(root / INSTALLED_DLL) == spec['payload']['installed_game_ai_dll_sha256'],
            'Installed AI DLL changed during the benchmark')


def summarize(root: Path, spec: dict) -> dict:
    private = root / PRIVATE
    outcomes = []
    failures = []
    evidence = {}
    for job in spec['payload']['jobs']:
        path = private / (job['id'] + '.json')
        record = read(path)
        require(record['spec_id'] == spec['id'] and record['job'] == job,
                'Game record does not match preregistered job')
        evidence[job['id']] = sha(path)
        if not record['completed']:
            failures.append(dict(job=job['id'], reason=record['failure']))
            continue
        match = record['match']
        seats = match['result']['seats']
        own = seats[job['seat']]
        others = [seat for seat in seats if seat['seat'] != job['seat']]
        outcomes.append(dict(job=job['id'], seed=job['seed'], seat=job['seat'],
                             profile=job['profile'], win_share=own['utility'],
                             first_place=own['rank'] == 1, rank=own['rank'], vp=own['vp'],
                             margin=own['vp'] - statistics.mean(x['vp'] for x in others),
                             original_hard_mean_vp=statistics.mean(x['vp'] for x in others),
                             cash=own['cash'], income=own['income'],
                             mean_v3_turn_ms=statistics.mean(t['whole_turn_ms'] for t in match['turn_reports']
                                                             if t['seat'] == job['seat'])))
    def metrics(rows):
        if not rows:
            return None
        return dict(games=len(rows), first_places=sum(x['first_place'] for x in rows),
                    win_share=sum(x['win_share'] for x in rows) / len(rows),
                    mean_rank=statistics.mean(x['rank'] for x in rows),
                    mean_v3_vp=statistics.mean(x['vp'] for x in rows),
                    mean_original_hard_vp=statistics.mean(x['original_hard_mean_vp'] for x in rows),
                    mean_vp_margin=statistics.mean(x['margin'] for x in rows),
                    mean_v3_turn_seconds=statistics.mean(x['mean_v3_turn_ms'] for x in rows) / 1000)
    report = dict(format='v3-vs-stock-hard-report/1.0.0', spec_id=spec['id'],
                  spec_sha256=sha(root / PUBLIC / 'spec.json'),
                  scheduled_games=len(spec['payload']['jobs']), completed_games=len(outcomes),
                  failures=failures, overall=metrics(outcomes),
                  win_share_setup_bootstrap_95=grouped_interval(outcomes, 'win_share') if not failures else None,
                  by_profile={style: metrics([x for x in outcomes if x['profile'] == style + '-v3'])
                              for style in STYLES},
                  by_seat={str(seat): metrics([x for x in outcomes if x['seat'] == seat])
                           for seat in range(4)},
                  outcomes=outcomes, raw_record_sha256=evidence,
                  interpretation='Descriptive 20-game headless sample only. Five setup clusters and 3-4 games per style cannot prove human-level play or rank styles. Stock Hard uses the preserved rule selector under guarded repaired rules. The v3 bank is a source-matched proxy for the installed 1.4.1 policy, not direct UI gameplay.')
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    require(1 <= args.workers <= 4, 'Use 1 to 4 workers')
    root = args.root.resolve()
    spec = prepare(root)
    print(json.dumps(dict(spec=spec['id'], games=20, workers=args.workers,
                          bank=spec['payload']['bank'], time_limit=10000,
                          work_limit=1024)), flush=True)
    run(root, spec, args.workers)
    report = summarize(root, spec)
    report_path = root / PUBLIC / 'report.json'
    if report_path.exists():
        require(read(report_path) == report, 'Existing report differs')
    else:
        write_new(report_path, report)
    print(json.dumps(dict(report=str(report_path), completed=report['completed_games'],
                          failures=len(report['failures']), overall=report['overall'])), flush=True)


if __name__ == '__main__':
    main()
