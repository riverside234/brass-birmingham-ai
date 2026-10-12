"""Additive worker integrity, rollback and chance/restore regression checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from brass_ai.host import RulesHost, HostError
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.host_runtime import training_host


def source_sha(path):
    return hashlib.sha256(path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode()).hexdigest()


class NativeRuntimeReviewTests(unittest.TestCase):
    def test_search_transformation_is_pinned_and_changed_helper_is_rejected(self):
        source = SOURCE_ROOT / 'training/runtime/throughput/search'
        manifest = json.loads((source / 'runtime.json').read_text(encoding='utf-8'))
        relative = 'scripts/build-search-host.py'
        self.assertEqual(manifest['authored_sources'][relative], source_sha(SOURCE_ROOT / relative))
        # A minimal fixture reaches source verification before any private
        # dependency is copied or executable is launched.
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
            root = Path(directory)
            target = root / 'training/runtime/throughput/search'
            target.mkdir(parents=True)
            for name in ['scripts/build-training-host.py', *manifest['authored_sources']]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE_ROOT / name, path)
            base = root / 'artifacts/r1-host/guarded'
            base.mkdir(parents=True)
            (base / 'runtime.json').write_text('{}', encoding='utf-8')
            (target / 'runtime.json').write_text(json.dumps(manifest), encoding='utf-8')
            with (root / relative).open('a', encoding='utf-8') as stream:
                stream.write('\n# changed transformation\n')
            with self.assertRaisesRegex(ValueError, 'Training host source changed.*build-search-host'):
                training_host(root, 'search')

    def test_same_revision_restore_and_partial_chance_keep_cache_and_privacy(self):
        for kind in ('ordinary', 'search'):
            with self.subTest(kind=kind), RulesHost(training_host(SOURCE_ROOT, kind)) as host:
                context = host.reset(153)
                self.assertTrue(host.enable_fastpath())
                initial = host.call('checkpoint')
                loan = next(c for c in host.choices() if c['kind'] == 'loan')
                context = host.apply(context, loan)
                old_choices = host.choices()
                context = host.call('restore', checkpoint=initial)
                with self.assertRaisesRegex(HostError, 'stale'):
                    host.apply(dict(context, game_id=initial['game_id']), loan)
                develop = next(c for c in host.choices() if c['kind'] == 'develop')
                context = host.apply(context, develop)
                self.assertEqual(context['revision'], 1)
                self.assertNotEqual(host.choices(), old_choices)
                before = host.call('canonical_hash')
                checkpoint = host.call('checkpoint')
                context = host.call('restore', checkpoint=checkpoint)
                self.assertEqual(before, host.call('canonical_hash'))
                if kind == 'ordinary':
                    continue
                # Complete the first turn to expose an explicit private draw.
                while not context['chance']:
                    choices = host.choices()
                    choice = next((c for c in choices if c['kind'] == 'finish'), choices[0])
                    snapshot = host.call('search_step', game_id=context['game_id'],
                                         revision=context['revision'], choice=choice['id'])
                    context = snapshot['context']
                self.assertNotIn('views', snapshot)
                self.assertEqual(snapshot['choices'], [])
                with self.assertRaisesRegex(HostError, 'unresolved_private_chance'):
                    host.observe(0)
                checkpoint = snapshot['checkpoint']
                before = checkpoint['hash']
                context = host.call('restore', checkpoint=checkpoint)
                restored = host.call('search_snapshot')
                self.assertEqual(before, restored['checkpoint']['hash'])
                self.assertEqual(snapshot['chance'], restored['chance'])
                with self.assertRaisesRegex(HostError, 'illegal_chance'):
                    host.call('search_step', game_id=context['game_id'], revision=context['revision'], card='forged')
                self.assertEqual(before, host.call('canonical_hash'))

    def test_optimized_apply_rolls_back_after_engine_mutation(self):
        compiler = SOURCE_ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
        code = r'''
using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using BrassHost;
using BoardGameRules.Entities.EventSourcing.Aggregates;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.NumericalResources.Events;
class Check {
  static int Main() {
    var host=new Host(19);
    var chance=typeof(Host).GetField("ExplicitChance");
    if(chance!=null)chance.SetValue(host,true);
    var before=host.Checkpoint();
    var routes=(IDictionary)typeof(AggregateRoot).GetField("_applyEventMethods",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(host.Game);
    var registration=(IEventRegistration)routes[typeof(MoneyLoaned)];
    var original=registration.GetType().GetMethod("ApplyEvent");
    host.Game.UnregisterApplyEventRegistration(registration);
    host.Game.RegisterApplyEvent<MoneyLoaned>(e=>{original.Invoke(registration,new object[]{e});throw new InvalidOperationException("injected_after_mutation");});
    var loan=host.Choices().First(c=>c.kind=="loan");
    bool rejected=false;
    try{host.Apply(host.Revision,loan.id);}catch(InvalidOperationException e){rejected=e.Message=="injected_after_mutation";}
    if(!rejected || !Newtonsoft.Json.Linq.JToken.DeepEquals(before,host.Checkpoint()))throw new Exception("Failed transaction changed checkpoint");
    host.Apply(host.Revision,loan.id);
    if(host.Revision!=1)throw new Exception("Reconstructed host could not continue");
    Console.WriteLine("rollback passed");return 0;
  }
}'''
        for kind in ('ordinary', 'search'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as directory:
                directory = Path(directory)
                runtime = training_host(SOURCE_ROOT, kind)
                for path in runtime.glob('*.dll'):
                    shutil.copyfile(path, directory / path.name)
                shutil.copyfile(runtime / 'BrassHost.exe', directory / 'BrassHost.exe')
                source = directory / 'Check.cs'
                source.write_text(code, encoding='utf-8')
                executable = directory / 'Check.exe'
                command = [str(compiler)] if os.name == 'nt' else ['mono', str(compiler)]
                arguments = ['/nologo', '/target:exe', '/langversion:7.3', '/nostdlib+', '/out:' + str(executable)]
                arguments += ['/reference:' + str(p) for p in directory.glob('*.dll')]
                arguments += ['/reference:' + str(directory / 'BrassHost.exe'), str(source)]
                subprocess.run(command + arguments, check=True, capture_output=True, text=True)
                run = [str(executable)] if os.name == 'nt' else ['mono', str(executable)]
                result = subprocess.run(run, check=True, capture_output=True, text=True)
                self.assertIn('rollback passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
