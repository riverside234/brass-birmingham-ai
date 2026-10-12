"""Isolated P5 original-Hard evaluation host; no game/native training changes."""
from pathlib import Path
import os
import shutil
import subprocess

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, read_json, require
from brass_ai.step5.league import write_json
from .contracts import STOCK


def sources(root):
    return sorted((root/'src/BrassHost').glob('*.cs')) + sorted((root/'src/BrassPersonalityHost').glob('*.cs')) + [
        root/'src/BrassLegacyAI/HostExtensions.cs', Path(__file__).resolve()]


def pins(root):
    return {p.relative_to(root).as_posix(): digest(p.read_text(encoding='utf-8').replace('\r\n', '\n'))
            for p in sources(root)}


def runtime(root):
    root = Path(root).resolve(); out = root/'artifacts/personality-host/guarded'
    manifest = read_json(out/'runtime.json')
    require(manifest.get('p5_baseline') == STOCK and manifest['p5_sources'] == pins(root),
            'Build the current separate P5 baseline host first')
    # RulesHost additionally verifies every runtime binary against R1 pins.
    return out


def build(root, mono=None):
    root = Path(root).resolve(); base = root/'artifacts/r1-host/guarded'
    original = read_json(base/'runtime.json')
    expected = read_json(root/'configs/rules/runtime-dependencies-1.1.0.json')
    require({e['name']: e['sha256'] for e in original['files'] if e['name'] != 'BrassHost.exe'} ==
            {e['name']: e['sha256'] for e in expected['files']}, 'Baseline dependency pins differ')
    for e in original['files']:
        require(Path(e['name']).name == e['name'] and sha(base/e['name']) == e['sha256'], 'Baseline bytes changed')
    out = root/'artifacts/personality-host/guarded'; out.mkdir(parents=True, exist_ok=True)
    for e in expected['files']: shutil.copyfile(base/e['name'], out/e['name'])
    source_files = []
    for source in sources(root):
        if source.suffix != '.cs': continue
        text = source.read_text(encoding='utf-8')
        if source.name == 'Program.cs':
            changes = {
                'Host host=null;': 'Host host=null;BrassPersonalityHost.StockOpponent stock=null;',
                'host=new Host((int)req["seed"],req["setup"] as JObject);data=host.Context();':
                    'host=new Host((int)req["seed"],req["setup"] as JObject);'
                    'stock=new BrassPersonalityHost.StockOpponent((int)req["seed"]);data=host.Context();',
                'case "context":': 'case "stock_step":'
                    'if((string)req["game_id"]!=host.GameId || (int)req["revision"]!=host.Revision)'
                    'throw new InvalidOperationException("stale_stock_request");'
                    'var action=stock.Step(host);data=new{context=host.Context(),action};break;case "context":'}
            for before, after in changes.items():
                require(text.count(before) == 1, 'P5 baseline hook changed')
                text = text.replace(before, after)
        if source.name == 'HostExtensions.cs':
            # Trusted pre-settlement observation for the existing causal cost
            # ledger. It is never passed to another seat's policy or public UI.
            before = 'Game.TurnFlow.ConfirmCurrentAction();Revision++;Advance();'
            require(text.count(before) == 1, 'Stock settlement hook changed')
            text = text.replace(before, 'var beforeFinish=(JObject)Wire.Token(Observe(seat));'
                'beforeFinish["context"]["phase"]="continuation";' + before)
            text = text.replace('return new{kind,cards};', 'return new{kind,cards,before_finish=beforeFinish};')
            hook = 'var move=select();'
            require(text.count(hook)==1, 'Stock event boundary changed')
            text = text.replace(hook, 'int p5Revision=Game.Revision;' + hook)
            hook = 'publicHistory.Add(new{seat,era,choice=new Choice(kind,card:cards.FirstOrDefault(),cards:cards.Skip(1).ToArray())});'
            require(text.count(hook)==1, 'Stock public history hook changed')
            text = text.replace(hook, 'P5RecordStock(kind,cards,p5Revision);')
        dest = out/source.name; dest.write_text(text, encoding='utf-8', newline='\n'); source_files.append(dest)
    compiler = root/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    require(compiler.is_file(), 'Transfer the existing pinned C# compiler package before building')
    args = ['/nologo', '/target:exe', '/langversion:7.3', '/nostdlib+', '/deterministic+', '/optimize+',
            f'/out:"{out / "BrassHost.exe"}"']
    args += [f'/reference:"{out / e["name"]}"' for e in expected['files']]
    args += [f'"{p}"' for p in source_files]
    rsp = out/'build.rsp'; rsp.write_text('\n'.join(args), encoding='utf-8-sig')
    subprocess.run(([str(compiler)] if os.name == 'nt' else [mono or 'mono', str(compiler)]) + ['@'+str(rsp)], check=True)
    manifest = dict(original, p5_baseline=STOCK, p5_sources=pins(root),
                    files=expected['files'] + [dict(name='BrassHost.exe', sha256=sha(out/'BrassHost.exe'))])
    write_json(out/'runtime.json', manifest)
    return dict(path=str(runtime(root)), baseline=STOCK, installed_game_modified=False)
