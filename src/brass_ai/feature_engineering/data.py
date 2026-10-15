"""Explicit heuristic raw-data re-audit; never a neural replay migration."""
import gzip
from pathlib import Path

from brass_ai.bank import sha
from brass_ai.learning.contracts import (CONFIG_DIR,FEATURE_VERSION,compatibility,decode_json,
    digest,project_path,read_json,require)

CONVERSION='heuristic-raw-to-features4/1.0.0'


def raw_compatible(actual):
    """Admission of complete raw wire facts, followed by a new native audit.

    Old encodings/architectures have no authority over regenerated tensors.
    Rules, wire, targets and reference/producer pins must still match exactly.
    """
    current=compatibility()
    require(type(actual) is dict,'Unsupported raw feature contract; regenerate demonstrations')
    generation={'four-seat-learning/1.0.0':1,'four-seat-learning/2.0.0':2,'four-seat-learning/3.0.0':3,'four-seat-learning/4.0.0':4}.get(actual.get('learning_contract'))
    require(generation is not None,'Unsupported raw feature contract; regenerate demonstrations')
    legacy=generation==1
    omitted={'feature_version','derived_source_sha256'} if legacy else set()
    require(set(actual)==set(current)-omitted,
            'Unsupported raw feature contract; regenerate demonstrations')
    schema=f'features-{generation}.0.0.json'
    require(actual['features_sha256']==digest(read_json(CONFIG_DIR/schema)),
            'Unknown raw layout; regenerate demonstrations')
    # This function grants permission for a full native re-audit of raw facts,
    # never permission to reuse old tensors, neural probabilities or teachers.
    # Complete older wire recordings contain the current hand and full legal
    # list; new dimensions alone do not require new expensive teacher games.
    fingerprints={'encoder_sha256','architecture_sha256'}|({'derived_source_sha256'} if not legacy else set())
    if not legacy:require(actual['feature_version']==f'permitted-features/{generation}.0.0','Unknown raw feature version')
    ignored={'learning_contract','features_sha256','feature_version'}|fingerprints|omitted
    require({k:v for k,v in actual.items() if k not in ignored}==
            {k:v for k,v in current.items() if k not in ignored},
            'Raw rules/wire/targets differ; regenerate demonstrations')
    for field in fingerprints:
        value=actual[field]
        require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),
                'Invalid raw source fingerprint')


def convert(root,manifest_path,output,*,mono=None):
    from brass_ai.recovery.demonstrations import _audit_path,_manifest
    from brass_ai.step5.league import write_json
    root=Path(root).resolve();source=project_path(root,manifest_path);out=project_path(root,output)
    require(not out.exists(),'Conversion requires a new output directory; old files are preserved')
    manifest=read_json(source)
    require(manifest['format']=='bc-dataset/1.0.0' and manifest['label_kind']=='qualified-v3-choice/1.0.0'
            and manifest['context']=='neutral-predecision/1.0.0','Unsupported demonstration manifest')
    raw_compatible(manifest['compatibility'])
    grouped={}
    for entry in manifest['entries']:
        require(entry['transitions_verified'],'Source demonstrations were not audited')
        grouped.setdefault(entry['path'],[]).append(entry)
    require(grouped,'Empty demonstration manifest')
    out.mkdir(parents=True);entries=[];originals=[];seen=set()
    for index,(name,selected) in enumerate(grouped.items()):
        path=project_path(root,name);before=sha(path)
        require(all(e['sha256']==before for e in selected),'Demonstration source changed')
        wanted={e['game_id']:e for e in selected}
        require(len(wanted)==len(selected),'Duplicate source game selection')
        target=out/'raw'/f'shard-{index:06d}.jsonl.gz';target.parent.mkdir(parents=True,exist_ok=True)
        found=set();labels=[]
        try:
            with gzip.open(path,'rt',encoding='utf-8') as incoming,gzip.open(target,'wt',encoding='utf-8') as outgoing:
                header=decode_json(next(incoming));raw_compatible(header['compatibility'])
                require(header['format']=='four-seat-replay/1.0.0' and header['provenance'].get('purpose')=='training',
                        'Only training heuristic replays can be converted')
                original_contract=header['compatibility'];header['compatibility']=compatibility()
                header['provenance']=dict(header['provenance'],feature_conversion=dict(
                    version=CONVERSION,source_sha256=before,
                    source_compatibility=original_contract,feature_version=FEATURE_VERSION))
                import json
                outgoing.write(json.dumps(header)+'\n')
                for line in incoming:
                    game=decode_json(line)
                    if game['game_id'] not in wanted:continue
                    require(game['game_id'] not in found,'Duplicate raw game')
                    require(game['split']==wanted[game['game_id']]['split'] and
                            game['setup_group']==wanted[game['game_id']]['setup_group'],'Source split/setup changed')
                    require(all(r['behavior']['kind']=='external' and r['teacher'] is None for r in game['records']),
                            'Neural behavior/search teachers cannot be converted or relabeled')
                    found.add(game['game_id']);labels.append(dict(game_id=game['game_id'],sha256=digest(game)))
                    outgoing.write(line if line.endswith('\n') else line+'\n')
            require(found==set(wanted),'Selected raw games missing')
            # This independently reconstructs every state/legal list/transition
            # and re-encodes all mandatory facts before committing the manifest.
            accepted,rejected=_audit_path((str(root),str(target),manifest['teachers'],mono,None))
            require(not rejected and len(accepted)==len(selected),
                    'Raw feature/native audit failed: '+str(rejected[:1]))
            for entry in accepted:
                require(entry['semantic_sha256']==wanted[entry['game_id']]['semantic_sha256'],
                        'Conversion changed decision/label alignment')
                require(entry['semantic_sha256'] not in seen,'Duplicate semantic source game')
                seen.add(entry['semantic_sha256'])
            entries.extend(accepted)
            require(sha(path)==before,'Source changed during conversion')
            originals.append(dict(path=name,sha256=before,labels=labels))
            print(f'feature conversion: {index+1}/{len(grouped)} shards audited',flush=True)
        except Exception as error:
            write_json(out/'failure.json',dict(source=name,error=str(error),committed=False,
                remedy='Required raw facts/rules are incompatible. Regenerate demonstrations in a new directory; do not zero-fill missing groups.'))
            raise ValueError(f'Feature conversion failed for {name}: {error}') from error
    result=_manifest(root,out/'manifest.json',entries,[],manifest['teachers'],manifest.get('calibration'),1)
    receipt=dict(format=CONVERSION,source_manifest=source.relative_to(root).as_posix(),
        source_manifest_sha256=sha(source),feature_version=FEATURE_VERSION,compatibility=compatibility(),
        inputs=originals,games=len(entries),partitions=result['partitions'],native_transitions_verified=True,
        preserved='Raw game/record objects, setup/splits, choices/IDs, chosen indices, behavior and provenance; only the new shard header describes re-encoding.',
        neural_or_search_label_reuse=False)
    write_json(out/'conversion.json',receipt);return receipt
