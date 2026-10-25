"""Read-only utility evaluation and a weight-preserving migration-test receipt."""
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


CONTRACT = 'migration-utility-readonly-validation/1.0.0'
DEFAULTS = dict(games=4, rows_per_game=128, seed=1700000000,
                device='cpu', threads=1, host=None)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def options(values=None):
    result = dict(DEFAULTS, **(values or {}))
    require(set(result) == set(DEFAULTS), 'Unknown validation options')
    for key, low, high in (('games', 2, 32),
                           ('rows_per_game', 16, 512), ('threads', 1, 4),
                           ('seed', 0, 2000000000)):
        require(type(result[key]) is int and low <= result[key] <= high,
                f'Invalid validation {key}: expected {low}..{high}')
    require(result['device'] in ('cpu', 'cuda', 'cuda:0'), 'Unsupported validation device')
    require(result['host'] is None or isinstance(result['host'], str), 'Invalid validation host')
    return result


def sample_steps(game, limit):
    """Bound memory while retaining seats, eras, phases and forced choices."""
    groups = {}
    for row in game['records']:
        context = row['view']['context']
        key = (context['era'], context['phase'], row['seat'], len(row['choices']) == 1)
        token = hashlib.sha256(f'{game["game_id"]}:{row["step"]}'.encode()).hexdigest()
        groups.setdefault(key, []).append((token, row['step']))
    queues = [sorted(rows) for _, rows in sorted(groups.items())]
    selected = set()
    while len(selected) < limit and any(queues):
        for rows in queues:
            if rows and len(selected) < limit:
                selected.add(rows.pop()[1])
    return selected


def utility_error(model, dataset, update):
    import torch
    from brass_ai.recovery.updates import value_output
    require(dataset.utility is not None, 'No complete-game utility targets')
    groups = {}

    def accumulate(key, predicted, target):
        total = groups.setdefault(key, dict(squared=0., absolute=0., signed=0.,
                                           predicted=0., target=0., values=0, decisions=0))
        difference = predicted - target
        total['squared'] += float(difference.square().sum())
        total['absolute'] += float(difference.abs().sum())
        total['signed'] += float(difference.sum())
        total['predicted'] += float(predicted.sum())
        total['target'] += float(target.sum())
        total['values'] += difference.numel()
        total['decisions'] += 1

    with torch.inference_mode():
        parts = dataset.microbatches(range(len(dataset.data)), update['microbatch_size'],
                                     update['max_candidate_cells'], update['bucket'])
        for ids, ix, inputs in dataset.value_batches(parts, model):
            predicted = value_output(model, inputs, dataset, ix)[-1].detach().cpu().double()
            target = dataset.utility[ix].detach().cpu().double()
            require(predicted.shape == target.shape == (len(ids), 4), 'Invalid utility target shape')
            require(torch.isfinite(predicted).all().item() and torch.isfinite(target).all().item(),
                    'Non-finite utility prediction/target')
            for i, index in enumerate(ids):
                era = dataset.data[index][0]['view']['context']['era']
                require(era in ('CanalEra', 'RailEra'), f'Unknown validation era: {era}')
                game_id = dataset.data[index][2][0]
                for key in (('overall',), ('era', era), ('game', game_id), ('game-era', game_id, era)):
                    accumulate(key, predicted[i], target[i])

    def metrics(key):
        total = groups.get(key)
        require(total and total['values'] > 0, f'No utility validation targets for {key}')
        require(all(math.isfinite(v) for v in total.values()), 'Non-finite utility error')
        count = total['values']
        return dict(mse=total['squared'] / count, rmse=math.sqrt(total['squared'] / count),
                    mae=total['absolute'] / count, bias=total['signed'] / count,
                    prediction_mean=total['predicted'] / count, target_mean=total['target'] / count,
                    values=count, decisions=total['decisions'])

    games = sorted({d[2][0] for d in dataset.data})
    require(len(games) >= 2, 'At least two independent validation games are required')
    return dict(overall=metrics(('overall',)),
                by_era={era: metrics(('era', era)) for era in ('CanalEra', 'RailEra')},
                by_game={game: dict(overall=metrics(('game', game)),
                    by_era={era: metrics(('game-era', game, era)) for era in ('CanalEra', 'RailEra')})
                    for game in games},
                games=len(games), bias_definition='prediction minus realized utility',
                aggregation='Equal weight per sampled decision and each of its four seat predictions')


def evaluate(model, dataset, update):
    """Measure fixed weights; readiness records completion, with no error cutoff."""
    import torch
    from brass_ai.learning.contracts import digest
    frozen = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    reward = copy.deepcopy(model.config.utility)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        metrics = utility_error(model, dataset, update)
    finally:
        for module, training in modes:
            module.training = training
    for name, tensor in frozen.items():
        require(torch.equal(tensor, model.state_dict()[name].detach().cpu()),
                f'Read-only validation changed a weight or buffer: {name}')
    require(model.config.utility == reward, 'Validation changed the reward specification')
    require(all(torch.isfinite(p).all().item() for p in model.state_dict().values()), 'Non-finite weights')
    receipt = dict(version=CONTRACT, validation_contract=CONTRACT, mode='read-only-validation',
                   reward_id=digest(reward), ready=True, completed=True, steps=0,
                   weights_unchanged=True, descriptors='all-unknown', metrics=metrics,
                   readiness_rule='Completed finite evaluation with both eras in every game; no MSE/bias cutoff',
                   scope='Migration-test validation of existing weights on fresh complete games; no fitting')
    return dict(contract=CONTRACT, ready=True, completed=True, calibration_receipt=receipt,
                metrics=metrics, checks=dict(finite=True, both_eras_per_game=True, weights_unchanged=True))


def collect_data(root, stage, model, config, opt, parent_id):
    """Fresh policy-only complete games; never read/write the active run replay."""
    from brass_ai.host import RulesHost
    from brass_ai.learning.batching import model_id
    from brass_ai.recovery.execution import run_game
    from brass_ai.recovery.replay import write_shard
    from brass_ai.recovery.telemetry import Timings
    from brass_ai.recovery.updates import prepared
    from brass_ai.step5.policy import NeuralPolicy
    io = _exporter()
    host_name = opt['host'] or config.get('host', 'artifacts/r1-host/guarded')
    host_dir = io.below(root, host_name)
    require((host_dir / 'runtime.json').is_file(),
            f'Prepared rules host missing: {host_dir}. Prepare the lab host or pass --validation-host.')
    timings, rows, receipts = Timings(), [], []
    # A copied evaluator freezes the actor while collection is in progress.
    with ExitStack() as cleanup:
        policy = NeuralPolicy(model, device=opt['device'], exploration=.05, temperature=1.,
                              focus='neutral', exploration_version=3,
                              batch_size=1, max_candidate_cells=131072)
        cleanup.callback(policy.close)
        host = cleanup.enter_context(RulesHost(host_dir))
        runtime_hash = io.receipt(host_dir / 'runtime.json')['sha256']
        for i in range(opt['games']):
            seed = opt['seed'] + i
            game_id = f'{CONTRACT}:{parent_id}:{seed}'
            provenance = dict(purpose='evaluation', evaluation=CONTRACT, kind='policy-only-selfplay',
                              model_id=model_id(model), descriptors='all-unknown')
            job = dict(seed=seed, sampling_seed=seed, stable_game_id=game_id, provenance=provenance)
            print(f'Utility read-only validation: complete game {i + 1}/{opt["games"]}', flush=True)
            game = run_game(host, job, [policy] * 4, timings,
                            max_decisions=int(config.get('max_decisions', 4096)))
            steps = sample_steps(game, opt['rows_per_game'])
            selected = prepared([game], predicate=lambda row: row['step'] in steps,
                                insolvency=model.config.utility['insolvency'], reward=model.config.utility)
            rows.extend(selected)
            relative = f'validation-games/{i:03d}.jsonl.gz'
            path = stage / relative
            write_shard(path, [game], dict(provenance, seed=seed, sampling_seed=seed,
                                          role='validation', host_manifest_sha256=runtime_hash), cache=False)
            receipts.append(dict(path=relative, **io.receipt(path), game_id=game_id,
                                 setup_group=game['setup_group'], seed=seed, role='validation', complete=True,
                                 selected_steps=sorted(steps), decisions=len(game['records'])))
    require(len({r['setup_group'] for r in receipts}) == opt['games'], 'Validation setups overlap')
    return rows, receipts, dict(runtime_manifest_sha256=runtime_hash, timings=timings.report())


def _exporter():
    # Keep helper code outside the fingerprinted trainer modules.
    spec = importlib.util.spec_from_file_location('_nn_export_io', Path(__file__).with_name('export-test-nn.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root, source_bundle, values, label, *, part_bytes=48 * 1024 * 1024):
    opt = options(values)
    io = _exporter()
    root = Path(root).resolve()
    output = io.below(root, 'test_nn')
    source_bundle = io.below(output, source_bundle)
    parent = io.verify_bundle(source_bundle)
    source_hash = parent['checkpoint']['sha256']
    request = dict(contract=CONTRACT, parent_manifest_sha256=io.receipt(source_bundle / 'manifest.json')['sha256'],
                   options=opt, part_bytes=part_bytes, helper_sha256=io.receipt(Path(__file__))['sha256'])
    # Bind evidence to the sampler, encoder, targets and checkpoint code.
    request['code_sha256'] = {p.relative_to(root).as_posix(): io.receipt(p)['sha256']
                             for p in sorted((root / 'src/brass_ai').rglob('*.py'))}
    request_id = io.digest(io.encode(request))
    pointer_path = output / (label + '.json')
    if pointer_path.exists():
        pointer = io.read(pointer_path)
        require(pointer.get('format') == io.POINTER, 'Unknown selection pointer')
        candidate = io.below(output, pointer['bundle'])
        existing = io.verify_bundle(candidate)
        proof = existing.get('calibration') or {}
        if proof.get('request_id') == request_id and proof.get('ready') is True:
            require(io.receipt(candidate / 'manifest.json')['sha256'] == pointer['manifest_sha256'],
                    'Validated selection manifest differs')
            return candidate, existing['derived_checkpoint']
    if 'torch' not in sys.modules:
        # Keep CPU off training GPUs; explicit CUDA uses its first configured device.
        os.environ['CUDA_VISIBLE_DEVICES'] = ('' if opt['device'] == 'cpu' else
                                             os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    sys.path.insert(0, str(root / 'src'))
    try:
        import torch
        from brass_ai.learning.checkpoint import load_checkpoint
        from brass_ai.learning.batching import model_id
        from brass_ai.recovery.updates import Dataset
    except ImportError as error:
        raise ValueError('Utility validation needs the lab PyTorch environment. '
                         '--no-utility-validation makes a byte-copy export without a new readiness claim.') from error
    stage = Path(tempfile.mkdtemp(prefix='.staging-validation-', dir=output))
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(opt['threads'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(opt['seed'])
            input_path = stage / 'input.pt'
            with input_path.open('xb') as stream:
                for part in parent['parts']:
                    with io.below(source_bundle, part['path']).open('rb') as source:
                        shutil.copyfileobj(source, stream, io.CHUNK)
            require(io.receipt(input_path)['sha256'] == source_hash, 'Validation input hash differs')
            model, raw = load_checkpoint(input_path, device=opt['device'])
            require(getattr(model.config, 'utility', None) is not None, 'Selected checkpoint has no utility head')
            model.eval()
            config = io.read(source_bundle / 'config.json')
            rows, games, host = collect_data(root, stage, model, config, opt, source_hash)
            require(len(games) == opt['games'] and all(g.get('complete') is True for g in games),
                    'Incomplete validation games')
            require(len({g['game_id'] for g in games}) == len({g['setup_group'] for g in games}) == opt['games'],
                    'Validation game identities/setups overlap')
            require({row[2][0] for row in rows} == {g['game_id'] for g in games},
                    'Validation rows do not match complete-game evidence')
            dataset = Dataset(rows, opt['device'], policy=False)
            update = dict(microbatch_size=64, max_candidate_cells=131072, bucket=True)
            report = evaluate(model, dataset, update)
            require(model_id(model, contracts=raw['compatibility']) == raw['model_id'],
                    'Validation collection/evaluation changed model weights or config')
            for label_name, metrics in [('overall', report['metrics']['overall']),
                                        *report['metrics']['by_era'].items()]:
                print(f'Utility validation {label_name}: MSE={metrics["mse"]:.6f}, '
                      f'bias={metrics["bias"]:+.6f}, decisions={metrics["decisions"]}', flush=True)
            report.update(request_id=request_id, request=request, source_checkpoint_sha256=source_hash,
                          source_model_id=raw['model_id'], source_bundle=source_bundle.name,
                          original_generation=parent['recorded_checkpoint'].get('generation'),
                          games=games, host=host)
            report['calibration_receipt'].update(request_id=request_id, source_checkpoint_sha256=source_hash,
                                                 source_model_id=raw['model_id'])
            validated = stage / 'validated.pt'
            # The only payload change is the receipt. Preserve source tensors,
            # model ID, optimizer, RNG and all other training metadata verbatim.
            payload = dict(raw, training=dict(raw['training'], utility_calibration=report['calibration_receipt']))
            torch.save(payload, validated)
            checked, saved = load_checkpoint(validated, device='cpu')
            identity = saved['model_id']
            require(saved['training']['utility_calibration']['ready'] is True and
                    checked.config.utility == model.config.utility and identity == raw['model_id'] and
                    all(torch.equal(saved['state_dict'][name], tensor) for name, tensor in raw['state_dict'].items()),
                    'Saved utility receipt/reward/weights differ')
            report['exported_model_id'] = identity
            (stage / 'calibration.json').write_bytes(io.encode(report))
            (stage / 'model-config.json').write_bytes(io.encode(model.config.to_dict()))
            derived = dict(path='checkpoint.pt', generation=parent['recorded_checkpoint'].get('generation'),
                           model_id=identity, sha256=io.receipt(validated)['sha256'], derived=True)
            extras = {name: stage / name for name in ('calibration.json', 'model-config.json')}
            extras.update({g['path']: stage / g['path'] for g in games})
            metadata = dict(recorded_checkpoint=parent['recorded_checkpoint'], derived_checkpoint=derived,
                            calibration=dict(request_id=request_id, ready=True, mode='read-only-validation',
                                             weights_unchanged=True,
                                             source_checkpoint_sha256=source_hash,
                                             source_bundle=source_bundle.name, contract=CONTRACT))
            bundle = io.make_bundle(root, output, io.below(root, parent['source_run']),
                                    (source_bundle / 'state.json').read_bytes(),
                                    io.read(source_bundle / 'state.json'), derived, validated,
                                    part_bytes, extra_files=extras, provenance=metadata)
            return bundle, derived
    finally:
        torch.set_num_threads(old_threads)
        if stage.exists():
            require(stage.parent.resolve() == output.resolve() and stage.name.startswith('.staging-validation-')
                    and not stage.is_symlink(), 'Unsafe validation cleanup target')
            shutil.rmtree(stage)
