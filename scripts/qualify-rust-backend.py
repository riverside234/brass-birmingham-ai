"""L01/L02 local evidence driver; no installation, training or backend selection.

Use the repository's existing Python dependencies (requirements-r2/searchlab).
See tests/RustBackend/README.md for the exact local invocation and gates.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['freeze','capture','qualify','traces','tests','benchmark','report'])
    parser.add_argument('--run',required=True,help='New or existing evidence directory under artifacts/rust-backend')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--upstream',type=Path,default=Path('C:/Users/BobXu/Desktop/brass'))
    parser.add_argument('--save-root',type=Path)
    parser.add_argument('--games',type=int,help='Smaller diagnostic only; never passes the full L02 gate')
    parser.add_argument('--roots-only',action='store_true',help='Diagnostic restore only; does not qualify full games')
    args=parser.parse_args(); out=(ROOT/args.run).resolve()
    if not out.is_relative_to(ROOT/'artifacts/rust-backend'): parser.error('--run must be inside artifacts/rust-backend')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from brass_ai.learning.contracts import read_json
    from brass_ai.rust_backend.baseline import freeze,verify_reference
    from brass_ai.rust_backend.fixtures import capture,qualify,write_json,file_sha,harness_receipt,receipt_current
    if args.operation=='freeze':
        if args.checkpoint is None or args.save_root is None: parser.error('freeze requires --checkpoint and --save-root')
        baseline=freeze(ROOT,out,args.checkpoint,args.upstream,save_root=args.save_root)
        result=dict(frozen=True,model=baseline['model']['model_id'],value_ready=baseline['model']['value_readiness']['ready'])
    else:
        baseline=read_json(out/'baseline.json');drift=verify_reference(baseline)
        if drift:write_json(out/'local/source-drift.json',dict(runtime_reference_unchanged=True,archived_provenance=drift))
        if args.operation=='capture':
            corpus=capture(ROOT,out/'local/fixtures',baseline,games=args.games)
            result=dict(roots=corpus['root_count'],games=corpus['game_count'],complete=corpus['complete'])
        elif args.operation=='qualify':
            result=qualify(ROOT,out/'local/fixtures',baseline,replay_games=not args.roots_only)
        elif args.operation=='traces':
            from brass_ai.rust_backend.traces import capture_traces
            result=capture_traces(ROOT,out,baseline)
        elif args.operation=='tests':
            import io,unittest
            os.environ['BRASS_RUST_RUN']=str(out.relative_to(ROOT))
            suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/RustBackend'),pattern='test_*.py')
            log=io.StringIO();ran=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
            (out/'local/negative-tests.log').write_text(log.getvalue(),encoding='utf-8')
            result=dict(passed=ran.wasSuccessful() and not ran.skipped,tests=ran.testsRun,
                        failures=len(ran.failures),errors=len(ran.errors),skipped=len(ran.skipped),
                        harness=harness_receipt('tests/RustBackend/test_contracts.py','tests/RustBackend/test_state_parity.py',
                                                'tests/RustBackend/support.py','src/brass_ai/rust_backend/benchmark.py'))
            write_json(out/'local/negative-tests.json',result)
            if not result['passed']:
                print(log.getvalue());raise SystemExit(1)
        elif args.operation=='benchmark':
            from brass_ai.rust_backend.benchmark import benchmark
            result=benchmark(ROOT,out,baseline)
        else:
            result=dict(baseline_sha256=file_sha(out/'baseline.json'),
                        model=baseline['model']['model_id'],value_ready=baseline['model']['value_readiness']['ready'],
                        progress=read_json(out/'local/progress.json'))
            print(json.dumps(result,indent=2));return
    drift=verify_reference(baseline)
    if drift:write_json(out/'local/source-drift.json',dict(runtime_reference_unchanged=True,archived_provenance=drift))
    progress=read_json(out/'local/progress.json')
    progress['commands'].append(dict(argv=sys.argv,finished_utc=datetime.now(timezone.utc).isoformat(),result=result))
    if args.operation=='freeze':
        progress['steps']['L01']=dict(status='reference frozen; baseline A pending',
            readiness='mechanical-test-only; utility calibration missing',evidence='baseline.json')
    elif args.operation=='capture':
        progress['steps']['L02']=dict(status='captured; replay and negative tests pending',evidence='local/fixtures/corpus.json')
    elif args.operation=='qualify' and not args.roots_only:
        progress['steps']['L02']=dict(status='reference corpus replay passed; negative tests pending',
            evidence='local/fixtures/qualification.json',roots=result['roots'],games=result['games'])
    elif args.operation=='benchmark':
        progress['steps']['L01']=dict(status='complete for mechanical testing',evidence='local/baseline-a.json',
            production_gate='pending utility calibration; no live or strength admission')
    if all((out/path).exists() for path in ('local/fixtures/qualification.json','local/fixtures/trace-qualification.json','local/negative-tests.json')):
        corpus=read_json(out/'local/fixtures/corpus.json');qualified=read_json(out/'local/fixtures/qualification.json')
        traces=read_json(out/'local/fixtures/trace-qualification.json');negative=read_json(out/'local/negative-tests.json')
        if (corpus['complete'] and qualified['passed'] and qualified['games']>=20 and qualified['roots']>=100
            and qualified['corpus_sha256']==file_sha(out/'local/fixtures/corpus.json')
            and traces['passed'] and negative['passed']
            and all(receipt_current(report.get('harness')) for report in (qualified,traces,negative))):
            progress['steps']['L02']=dict(status='complete',evidence='local/fixtures/qualification.json',
                roots=qualified['roots'],games=qualified['games'],transitions=qualified['transitions'],
                negative_tests=negative['tests'],scripted_replies=traces['scripted_replies'],sampled_worlds=traces['sampled_worlds'],
                remaining='Native engine/search and full search traces are L03-L15')
    write_json(out/'local/progress.json',progress)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
