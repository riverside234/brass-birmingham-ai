"""Shared stdlib-only source inventory for training and machine transfer."""
from pathlib import Path

from .contracts import digest


def implementation_sources(root):
    root = Path(root)
    # Source-bound runs cannot silently resume with changed learning/search code.
    from brass_ai.learning.state_layout import LAYOUT_FILE
    from brass_ai.learning.contracts import FEATURE_SCHEMA
    base = root/'src/brass_ai'
    paths = [p for group in ('step5','recovery','search','learning','feature_engineering') for p in (base/group).glob('*.py')]
    paths += [root/'configs/learning'/FEATURE_SCHEMA]
    paths += [root/'configs/learning'/LAYOUT_FILE]
    paths += [root/'configs/learning/domino-descriptor-1.0.0.json',root/'src/BrassLabBootstrap/Bootstrap.cs']
    from brass_ai.learning.state_layout import MODEL_LAYOUT_FILE
    from brass_ai.learning.board_topology import TOPOLOGY_FILE,BOARD_FILE
    paths += [root/'configs/learning'/name for name in (MODEL_LAYOUT_FILE,TOPOLOGY_FILE,BOARD_FILE)]
    paths += [root/'configs/learning/architecture-source-equivalence.json']
    paths += [root/'configs/learning/v11-selector-source-equivalence.json']
    paths += [root/'configs/learning/reward-source-equivalence.json']
    paths += [root/'configs/learning/derived-source-equivalence.json']
    teacher_reuse=root/'configs/learning/collector-performance-teacher-reuse.json'
    if teacher_reuse.is_file():paths += [teacher_reuse]
    paths += [base / 'learning/runner.py', base / 'learning/batching.py']
    paths += [base/'searchlab/spiel.py',base/'beliefs.py',base/'runtime_contracts.py',base/'host.py',base/'bank.py',base/'legacy.py']
    # Native replay/admission and diagnostic resume are learning dependencies.
    paths += sorted((base/'rust_backend').glob('lab_*.py'))
    paths += [p for p in (root/'lab-native').rglob('*') if p.is_file() and
              p.suffix in ('.rs','.toml','.lock') and 'target' not in p.parts]
    paths += sorted((root/'src/BrassTrainingHost').glob('*.cs'))
    paths += [root/'scripts/build-training-host.py',root/'scripts/build-search-host.py']
    paths += sorted((root/'training/runtime/throughput').glob('*/runtime.json'))
    paths += sorted((root/'src/BrassLocalScoringHost').glob('*.cs'))
    paths += [root/'scripts/build-local-scoring-host.py']
    paths += sorted((root/'training/runtime/local-scoring').glob('*/runtime.json'))
    paths += sorted((root/'configs/contracts/masked-ismcts').glob('*.json'))
    paths += [root/'configs/references/masked-ismcts.lock.json']
    # Current heuristic opponents share game policy/search sources. A C# or
    # profile edit is a reviewed continuation, just like a Python search edit.
    from brass_ai.search.legacy_runtime import source_paths
    paths += source_paths(root)
    paths += sorted((root/'configs/legacy-ai').glob('*.json'))
    return {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8').replace('\r\n', '\n')
            for p in sorted(paths)}



def implementation_id(root):
    return digest(implementation_sources(root))
