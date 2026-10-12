"""Stage a game helper from an accepted source archive. Never installs it."""
from pathlib import Path
import os
import shutil
import subprocess

from brass_ai.bank import sha
from brass_ai.learning.contracts import require
from .contracts import read,write,seal,child,inventory,identity

BUILD='personality-game-build/1.0.0'


def edits(files):
    debt='src/BrassPatch/Liquidation/LiquidationFixes.cs'
    anchor='// Preserve the shipped AI\'s valuation, but never let it liquidate a human player\'s tile.'
    require(files[debt].count(anchor)==1,'Accepted liquidation hook anchor changed')
    files[debt]=files[debt].replace(anchor,'''// Neural seats use their own qualified policy for mandatory debt.
            var neural=Type.GetType("BrassLegacyAI.PersonalityTurnBridge, BrassLegacyAI",true);
            if((bool)neural.GetMethod("PlayDebt").Invoke(null,new object[]{useCase}))return;
            '''+anchor)
    def change(name,old,new):
        path='src/BrassLocalAI/'+name
        require(files[path].count(old)==1,'Accepted source anchor changed: '+path+' / '+old[:60])
        files[path]=files[path].replace(old,new)
    change('Policy/AiSelectionCatalog.cs','if (family == AiFamily.Gamma) return gamma;',
           'if (family == AiFamily.Gamma) return PersonalityPackage.Entries();')
    change('Policy/AiSelectionCatalog.cs',
           'Choose from available Gamma models at game creation. Models and the neural game backend are not installed yet. Search does not narrow Random.',
           'Choose from the 16 available Gamma styles at game creation. Each game varies within its selected style.')
    change('Policy/LocalGameState.cs','var state = SaveExtensions.Get(metadata);','var state = PersonalityState.Unwrap(SaveExtensions.Get(metadata));')
    change('Policy/LocalGameState.cs','SaveExtensions.Set(metadata, state);','PersonalityState.SetLegacy(metadata, state);')
    change('Unity/LocalTurnBridge.cs','return active != null && !active.stopped;','return PersonalityTurnBridge.IsThinking || active != null && !active.stopped;')
    change('Unity/LocalTurnBridge.cs','public static void PlayNow() { }','public static void PlayNow() { PersonalityTurnBridge.PlayNow(); }')
    change('Unity/LocalTurnBridge.cs','generation++;','generation++; PersonalityTurnBridge.CancelActive();')
    change('Unity/LocalTurnBridge.cs','public static void PlayAITurn(PlayAiTurnUseCase owner, bool endTurn)\n        {',
           'public static void PlayAITurn(PlayAiTurnUseCase owner, bool endTurn)\n        {\n            if (PersonalityTurnBridge.Dispatch(owner,endTurn)) return;')
    change('Unity/LocalMenu.cs','public static string Strategy(PlayerMetadata player)', '''public static void SelectNeural(PlayerMetadata player,AiChoice choice) {
            if(choice.Family!=AiFamily.Gamma || !AiSelectionCatalog.CanChoose(choice))throw new InvalidOperationException("Qualified neural style unavailable");
            var s=Selections.GetValue(player,p=>new Selection());s.Choice=choice;s.DisplayFamily="Gamma";
        }
        public static string Strategy(PlayerMetadata player)''')
    change('Unity/LocalMenu.cs','choice.Family != AiFamily.V4 || !AiSelectionCatalog.CanChoose(choice)', '!AiSelectionCatalog.CanChoose(choice)')
    change('Unity/LocalMenu.cs','Select(player, choice.EntryId); Refresh(', 'if(choice.Family==AiFamily.Gamma)SelectNeural(player,choice);else Select(player,choice.EntryId); Refresh(')
    change('Unity/LocalMenu.cs','return key == "random" ? "Random" : ProfileCatalog.Labels[ProfileCatalog.Index(key)];',
           'return key == "random" ? "Random" : Choices.Contains(key) ? ProfileCatalog.Labels[ProfileCatalog.Index(key)] : PersonalityPackage.Entries().Single(x=>x.Choice.EntryId==key).Name;')
    change('Unity/LocalMenu.cs',': ProfileCatalog.Description(key);', ': Choices.Contains(key) ? ProfileCatalog.Description(key) : PersonalityPackage.Entries().Single(x=>x.Choice.EntryId==key).Description;')
    change('Unity/LocalMenu.cs','(Choices.Contains(Strategy(p)) && DifficultyCatalog.Contains(Difficulty(p)))',
           '(AiSelectionCatalog.CanChoose(Choice(p)) && DifficultyCatalog.Contains(Difficulty(p)) && (Choice(p).Family!=AiFamily.Gamma || count==4 && !Metadata(owner).GameStartConfiguration.IsIntroductoryGame))')
    change('Unity/LocalMenu.cs','Select(player, player.AiType == AIType.Human ? choice : ResolveStrategy(choice));',
           'if(Choice(Metadata(owner).PlayerMetadata[i]).Family==AiFamily.Gamma)SelectNeural(player,Choice(Metadata(owner).PlayerMetadata[i]));\n                    else Select(player, player.AiType == AIType.Human ? choice : ResolveStrategy(choice));')
    change('Unity/LocalMenu.cs','LocalGameState.Create(metadata, Strategy, Difficulty)',
           'LocalGameState.Create(metadata, p=>Choice(p).Family==AiFamily.Gamma?"balanced":Strategy(p), Difficulty)')
    change('Unity/LocalMenu.cs','SaveExtensions.Set(app.Metadata, state);', 'PersonalityState.Attach(app.Metadata,state,Choice,Difficulty);')
    change('Unity/LocalMenu.cs','Select(p, (string)row["strategy"]);',
           '''var neural=PersonalityState.Assignment(app.Metadata,(int)p.PlayerStartConfiguration.Color);
                if(neural!=null){SelectNeural(p,new AiChoice(AiFamily.Gamma,(string)neural["archetype"]));SelectDifficulty(p,(string)neural["tier"]=="medium"?"mid":(string)neural["tier"]);continue;}
                Select(p, (string)row["strategy"]);''')
    change('Unity/LocalMenu.cs','draft.Page != PlayerSelectionPage.V4 || draft.Choice.Family != AiFamily.V4 ||',
           '(draft.Page != PlayerSelectionPage.V4 && draft.Page != PlayerSelectionPage.Gamma) ||\n                (draft.Page==PlayerSelectionPage.Gamma)!=(draft.Choice.Family==AiFamily.Gamma) ||')
    change('Unity/LocalMenu.cs','else { selection.Choice = draft.Choice; selection.Difficulty = draft.Difficulty; player.AiType = AIType.Hard; }',
           'else { selection.Choice = draft.Choice; selection.DisplayFamily=draft.Choice.Family.ToString(); selection.Difficulty = draft.Difficulty; player.AiType = AIType.Hard; }')
    change('Unity/AiSelectionPanel.cs','? PlayerSelectionPage.Human : PlayerSelectionPage.V4;',
           '? PlayerSelectionPage.Human : LocalMenu.Choice(row.Player).Family==AiFamily.Gamma ? PlayerSelectionPage.Gamma : PlayerSelectionPage.V4;')
    change('Unity/AiSelectionPanel.cs','CurrentPage == PlayerSelectionPage.V4 || fixture && catalogs[Family].Any(e => e.Available)',
           'catalogs[Family].Any(e => e.Available)')
    change('Unity/AiSelectionPanel.cs','if (CurrentPage == PlayerSelectionPage.Gamma && !fixture) return;',
           'if (CurrentPage == PlayerSelectionPage.Gamma && !fixture && !PersonalityPackage.Available) return;')
    change('Unity/AiSelectionPanel.cs','CurrentPage == PlayerSelectionPage.Gamma ? "Gamma is not available yet." : "Maximum thinking time"',
           'CurrentPage == PlayerSelectionPage.Gamma ? (PersonalityPackage.Available ? (PersonalityPackage.ReviewMode ? "Unreleased review · maximum thinking time" : "Qualified tier · maximum thinking time") : "Qualified model pending") : "Maximum thinking time"')
    change('Unity/AiSelectionPanel.cs','CurrentPage == PlayerSelectionPage.V4 && AiSelectionCatalog.CanChoose(selected)',
           'AiSelectionCatalog.CanChoose(selected)')
    change('Unity/AiSelectionPanel.cs','CurrentPage == PlayerSelectionPage.Gamma && !fixture ?',
           'CurrentPage == PlayerSelectionPage.Gamma && !fixture && !PersonalityPackage.Available ?')
    change('Unity/AiSelectionPanel.cs','"Available when trained Gamma models are installed."',
           '(PersonalityPackage.Available ? "Choose from all 16 styles at game start." : "Available after model qualification.")')
    change('Unity/AiSelectionPanel.cs','" models · "','" styles · "')
    return files


def stage(root,output,*,base_version='1.4.31'):
    root=Path(root).resolve();out=Path(output).resolve();base=root/'artifacts/patch-releases'/base_version
    require(os.name=='nt','Stage the Unity .NET helper on Windows')
    require(not out.exists(),'Use a new integration build directory')
    m=read(base/'build-manifest.json');require(read(base/'validation.json')['passed'],'Accepted baseline required')
    for pin in read(base/'archive-manifest.json')['files']:
        require(sha(child(base,pin['path'])).lower()==pin['sha256'].lower(),'Accepted archive changed')
    files={p['path']:child(base,p['path']).read_text(encoding='utf-8-sig').replace('\r\n','\n') for p in m['sourcePins'] if p['path'].endswith('.cs')}
    files=edits(files);out.mkdir(parents=True);sources=out/'sources';managed=out/'Managed';managed.mkdir()
    for pin in m['sourcePins']:
        src=child(base,pin['path']);require(sha(src).lower()==pin['sha256'].lower(),'Source pin differs')
        dst=child(sources,pin['path']);dst.parent.mkdir(parents=True,exist_ok=True)
        if pin['path'] in files:dst.write_text(files[pin['path']],encoding='utf-8',newline='\n')
        else:shutil.copyfile(src,dst)
    for src in (root/'src/BrassPersonalityGame').glob('*.cs'):
        dst=sources/'src/BrassPersonalityGame'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for name in ('Assembly-CSharp.dll','BrassPatch.dll'):shutil.copyfile(base/'Managed'/name,managed/name)
    compiler=root/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    references=[p for p in (root/'BrassBirminghamUnity_Data/Managed').glob('*.dll') if p.name not in ('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll') and 'Tests' not in p.name and 'Regression' not in p.name]
    references += [managed/'Assembly-CSharp.dll',managed/'BrassPatch.dll']
    patch_args=['/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',f'/out:"{managed}/BrassPatch.dll"']
    patch_args += [f'/reference:"{p}"' for p in references if p.name!='BrassPatch.dll']
    patch_args += [f'"{p}"' for p in (sources/'src/BrassPatch').rglob('*.cs')]
    patch_rsp=out/'patch.rsp';patch_rsp.write_text('\n'.join(patch_args),encoding='utf-8-sig')
    patch_result=subprocess.run([str(compiler),'@'+str(patch_rsp)],capture_output=True,text=True)
    (out/'patch-compile.log').write_text(patch_result.stdout+patch_result.stderr,encoding='utf-8')
    require(patch_result.returncode==0,'Liquidation dispatch compilation failed: '+patch_result.stdout+patch_result.stderr)
    inputs=list((sources/'src/BrassLocalAI').rglob('*.cs'))+list((sources/'src/BrassMaskedSearch').glob('*.cs'))+list((sources/'src/BrassPersonalityGame').glob('*.cs'))
    resources=[]
    for key in ('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'):
        for suffix in ('v2','v3','v3-masked','v3-masked-strength','v3-masked-era','v3-masked-industry','v3-masked-economy'):
            resources.append(f'/resource:"{sources}/configs/legacy-ai/{key}-{suffix}.json",BrassLegacyAI.Profiles.{key}-{suffix}.json')
        resources.append(f'/resource:"{sources}/configs/legacy-ai/archive/1.4.0/{key}-v3.json",BrassLegacyAI.Profiles.{key}-v3-1.4.0.json')
    args=['/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',f'/out:"{managed}/BrassLegacyAI.dll"']
    args += [f'/reference:"{p}"' for p in references]+[f'"{p}"' for p in inputs]+resources
    rsp=out/'build.rsp';rsp.write_text('\n'.join(args),encoding='utf-8-sig')
    result=subprocess.run([str(compiler),'@'+str(rsp)],capture_output=True,text=True)
    (out/'compile.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    require(result.returncode==0,'Integration compilation failed: '+result.stdout+result.stderr)
    report=seal(dict(format=BUILD,contract=identity(),base_version=base_version,base_files=m['files'],
        files=[dict(name=p.name,sha256=sha(p)) for p in sorted(managed.glob('*.dll'))],
        source_pins=inventory(sources),installed=False,qualified=False))
    write(out/'build.json',report);return report
