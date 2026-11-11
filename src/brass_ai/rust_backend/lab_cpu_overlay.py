"""Additive CPU-inference patches, applied only to a detached lab-native copy.

The qualified GPU binaries keep building from the untouched ``lab-native`` tree.
These exact-text patches add in-process ONNX Runtime evaluation to a separate
pair of binaries; each anchor must match once, so upstream drift fails loudly.
"""
import hashlib
from pathlib import Path
import shutil

from brass_ai.learning.contracts import SOURCE_ROOT, digest, require

CONTRACT = 'brass-lab-cpu-overlay/1.0.0'
OVERLAY_DIR = 'lab-native-cpu'
ORT = 'ort = { version = "=2.0.0-rc.10", default-features = false, features = ["std", "load-dynamic"] }'

PATCHES = {
    'Cargo.toml': (
        ('[dependencies]\nbrass-ai-core = { path = "../native/brass-ai-core" }\n',
         '[dependencies]\n' + ORT + '\nbrass-ai-core = { path = "../native/brass-ai-core" }\n'),
    ),
    'src/lib.rs': (
        ('pub mod scoring;\n', 'pub mod cpu_cotton;\npub mod cpu_eval;\npub mod scoring;\n'),
        ('impl Lane {\n    pub fn new(',
         'impl Lane {\n'
         '    /// Tensors of the last `feature_arrays` call, for in-process inference.\n'
         '    pub fn feature_tensors(&self) -> Result<&brass_ai_core::features::FeatureTensors> {\n'
         '        Ok(self.encoder.tensors().ok_or("missing tensors")?)\n'
         '    }\n'
         '    pub fn new('),
    ),
    'src/main.rs': (
        ('    let mut export_cursors = (0usize, 0usize);\n',
         '    let mut export_cursors = (0usize, 0usize);\n'
         '    let mut cpu = brass_lab_collector::cpu_eval::Cache::default();\n'),
        ('"performance_wire":"ordinary-collector-fast/1","export_pages":"ordinary-export-pages/1"}),',
         '"performance_wire":"ordinary-collector-fast/1","export_pages":"ordinary-export-pages/1",\n'
         '                        "cpu_inference":brass_lab_collector::cpu_eval::CONTRACT}),'),
        ('                "checkpoint" => {\n',
         '                "configure_cpu" => cpu.configure(&r),\n'
         '                "load_model" => cpu.load(&r),\n'
         '                "cpu_report" => Ok(cpu.report()),\n'
         '                "infer" => {\n'
         '                    check(revision <= u32::MAX as u64, "invalid revision")?;\n'
         '                    let context: brass_ai_core::features::FeatureContext =\n'
         '                        serde_json::from_value(r["context"].clone())?;\n'
         '                    let model = context.model_id.clone();\n'
         '                    let l = lane.as_mut().ok_or("missing lane")?;\n'
         '                    let (mut meta, _) = l.feature_arrays(token, revision as u32, context)?;\n'
         '                    let (bytes, forward_ns) = cpu.infer(&model, l.feature_tensors()?)?;\n'
         '                    meta["wire"] = json!("ordinary-infer/1");\n'
         '                    meta["bytes"] = json!(bytes.len());\n'
         '                    meta["forward_ns"] = json!(forward_ns);\n'
         '                    arrays = Some(vec![bytes]);\n'
         '                    Ok(meta)\n'
         '                }\n'
         '                "checkpoint" => {\n'),
    ),
    'src/teacher.rs': (
        # The g1190 restart schedule's Stage B moves w in .02 steps (fifty collections):
        # a 1/100 grid keeps every earlier 1/20 weight valid.
        ('            && (w * 20. - (w * 20.).round()).abs() < 1e-8\n',
         '            && (w * 100. - (w * 100.).round()).abs() < 1e-8\n'),
        ('        self.encoding_ns += start.elapsed().as_nanos() as u64;\n'
         '        let mut binding = json!(',
         '        self.encoding_ns += start.elapsed().as_nanos() as u64;\n'
         '        if crate::cpu_eval::teacher_enabled() {\n'
         '            let began = Instant::now();\n'
         '            let (logits, utility) =\n'
         '                crate::cpu_eval::teacher_evaluate(tensors, self.search_value.as_ref())?;\n'
         '            self.wait_ns += began.elapsed().as_nanos() as u64;\n'
         '            self.requests += 1;\n'
         '            check(self.clock.now_ns() < self.deadline, "expired/cancelled CPU reply")?;\n'
         '            check(logits.len() == info.actions.len(), "CPU legal width differs")?;\n'
         '            let stats = Statistics::new(logits, utility)?;\n'
         '            if self.root_watch.as_ref() == Some(info) { self.root_stats = Some(stats.clone()); }\n'
         '            return Ok(stats);\n'
         '        }\n'
         '        let mut binding = json!('),
        ('        let mut game = Brass::new(root.clone(), false)?;\n'
         '        let actual = game.root()?;\n',
         '        // The frozen importer refuses a legal mid-sale develop bonus that still owes beer,\n'
         '        // so search cannot rebuild that root. Build it from the true state; the policy\n'
         '        // then chooses, as it does for any unsearched decision.\n'
         '        let unsearchable = crate::cpu_eval::unimportable_bonus(&state);\n'
         '        let actual = if unsearchable {\n'
         '            crate::cpu_eval::direct_root(&state, &view, &root.recall, &root.context, &ledger)?\n'
         '        } else {\n'
         '            let mut game = Brass::new(root.clone(), false)?;\n'
         '            game.root()?\n'
         '        };\n'),
        ('        let choice = if let Some(owner) = &mut session {\n',
         '        let choice = if let Some(owner) = session.as_mut().filter(|_| !unsearchable) {\n'),
        ('    let calibration = request["op"] == "calibrate";\n',
         '    crate::cpu_eval::teacher_install(&request, &expected, reward.insolvency.max_cost)?;\n'
         '    let calibration = request["op"] == "calibrate";\n'),
        ('"verification_policy":"sampled-unbounded-verification/3"}),',
         '"verification_policy":"sampled-unbounded-verification/3",\n'
         '        "inference":crate::cpu_eval::teacher_receipt()}),'),
        ('                k: if turn_work.fast { 4 } else { 16 },\n',
         '                k: crate::cpu_eval::teacher_k(turn_work.fast, calibration, if turn_work.fast { 4 } else { 16 }),\n'),
        ('                simulations: 8,\n',
         '                simulations: crate::cpu_eval::teacher_simulations(turn_work.fast, calibration),\n'
         '                c2: crate::cpu_eval::teacher_c_scale(),\n'),
        ('    let target: Vec<f32> = e.policy()?.into_iter().map(|v| v as f32).collect();\n',
         '    let target: Vec<f32> = crate::cpu_eval::teacher_policy(&e.stats, e.info.actor)?\n'
         '        .into_iter()\n'
         '        .map(|v| v as f32)\n'
         '        .collect();\n'),
        ('"c1":50,"c2":0.1,"max_n":', '"c1":50,"c2":crate::cpu_eval::teacher_c_scale(),"max_n":'),
        ('                profile_budget(clock.now_ns(), turn_work.fast)?,\n',
         '                match crate::cpu_eval::teacher_turn_budget(clock.now_ns(), turn_work.fast, calibration)? {\n'
         '                    Some(budget) => budget,\n'
         '                    None => profile_budget(clock.now_ns(), turn_work.fast)?,\n'
         '                },\n'),
        ('            let mode = if calibration {\n                Mode::Count\n',
         '            let mode = if calibration || crate::cpu_eval::teacher_count_mode() {\n                Mode::Count\n'),
        ('                    if all_seat { row["teacher"]["search_value"] = job["search_value"].clone(); }\n',
         '                    if all_seat { row["teacher"]["search_value"] = job["search_value"].clone(); }\n'
         '                    if crate::cpu_eval::teacher_enabled() { row["teacher"]["inference_backend"] = json!("cpu-ort"); }\n'
         '                    crate::cpu_eval::teacher_mark_label(&mut row, turn_work.fast, explored)?;\n'),
        # g1491: a job with the reviewed revision labels its short searches too.
        ('        if session.is_some() && view.context.phase == Phase::Action && !turn_work.fast && !h2h && job["shadow_probe"] != true {\n',
         '        if session.is_some() && view.context.phase == Phase::Action\n'
         '            && (!turn_work.fast || crate::cpu_eval::teacher_short_labels())\n'
         '            && !h2h && job["shadow_probe"] != true {\n'),
        # g1491: the played action may be sampled from the decision root's visit counts.
        ('        let mut evidence = None;\n        let mut supported = false;\n',
         '        let cotton_limit = crate::cpu_eval::teacher_cotton_prepare(&view, &actual, turn_work.fast);\n'
         '        if let Some(owner) = &mut session { owner.set_teacher_cotton(&view.context, cotton_limit)?; }\n'
         '        let mut evidence = None;\n        let mut supported = false;\n        let mut explored = false;\n'),
        ('            supported = out.actions.first().is_some_and(|a| a.support >= 2);\n'
         '            out.action_ids\n'
         '                .first()\n'
         '                .and_then(|a| a.first())\n'
         '                .cloned()\n'
         '                .unwrap_or_else(|| actual.actions[0].clone())\n'
         '        } else if actual.actions.len() == 1 {\n',
         '            supported = out.actions.first().is_some_and(|a| a.support >= 2);\n'
         '            let planned = out.action_ids\n'
         '                .first()\n'
         '                .and_then(|a| a.first())\n'
         '                .cloned()\n'
         '                .unwrap_or_else(|| actual.actions[0].clone());\n'
         '            let (played, sampled) = crate::cpu_eval::teacher_explore(\n'
         '                planned,\n'
         '                evidence.as_ref().filter(|e| e.info == actual).map(|e| e.stats.count.as_slice()),\n'
         '                &actual,\n'
         '                &view,\n'
         '                turn_work.fast,\n'
         '                seed,\n'
         '                rows,\n'
         '            )?;\n'
         '            explored = sampled;\n'
         '            played\n'
         '        } else if actual.actions.len() == 1 {\n'),
        ('        state = ordinary_after(&state, &choice, &rail, &mut recorder)?;\n',
         '        state = ordinary_after(&state, &choice, &rail, &mut recorder)?;\n'
         '        crate::cpu_eval::teacher_cotton_commit(&view, selected, state.context().revision)?;\n'),
        ('"input_bytes":peer.bytes,"elapsed_ns":clock.now_ns(),"root_value_sources":root_value_sources},',
         '"input_bytes":peer.bytes,"elapsed_ns":clock.now_ns(),"root_value_sources":root_value_sources,\n'
         '            "cpu_inference":crate::cpu_eval::teacher_receipt()},'),
    ),
}


def _normalized(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def overlay_files(root=SOURCE_ROOT):
    base = Path(root)/OVERLAY_DIR
    return [p for p in sorted(base.rglob('*'))
            if p.is_file() and 'target' not in p.relative_to(base).parts]


def fingerprint(root=SOURCE_ROOT):
    """Everything that decides the CPU binaries, beyond the sealed native core."""
    root = Path(root)
    names = sorted(['Cargo.toml', 'Cargo.lock', 'src/lib.rs', 'src/main.rs', 'src/teacher.rs', 'src/bin/teacher.rs',
                    'src/collection_wire.rs', 'src/ordinary_delta.rs', 'src/scoring.rs', 'core-overlay/teacher.rs'])
    from .lab_cotton_overlay import PATCHES as cotton_patches
    return digest(dict(contract=CONTRACT, patches=PATCHES, cotton_patches=cotton_patches,
        base={name: _normalized(root/'lab-native'/name) for name in names},
        overlay={p.relative_to(root/OVERLAY_DIR).as_posix(): _normalized(p) for p in overlay_files(root)}))


def apply(directory, root=SOURCE_ROOT):
    """Copy the reviewed sources into ``directory`` and apply the exact patches."""
    directory = Path(directory).resolve()
    root = Path(root).resolve()
    require(directory != (root/'lab-native').resolve() and not directory.exists(),
            'CPU overlay requires a new detached build directory')
    shutil.copytree(root/'lab-native', directory, ignore=shutil.ignore_patterns('target'))
    for name, changes in PATCHES.items():
        path = directory/name
        text = path.read_bytes().replace(b'\r\n', b'\n').decode('utf-8')
        for before, after in changes:
            require(text.count(before) == 1, 'CPU overlay drift in '+name+': '+before[:60])
            text = text.replace(before, after)
        path.write_bytes(text.encode('utf-8'))
    shutil.copyfile(root/OVERLAY_DIR/'src/cpu_eval.rs', directory/'src/cpu_eval.rs')
    shutil.copyfile(root/OVERLAY_DIR/'src/cpu_cotton.rs', directory/'src/cpu_cotton.rs')
    shutil.copyfile(root/OVERLAY_DIR/'Cargo.lock', directory/'Cargo.lock')
    return dict(contract=CONTRACT, source=fingerprint(root))
