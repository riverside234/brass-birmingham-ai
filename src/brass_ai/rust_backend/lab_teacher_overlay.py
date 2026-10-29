"""B04 additive export hooks, applied only to a fresh copy of the sealed core.

The installed game and the B01/B02 archives are never rewritten. The resulting
core has its own real build fingerprint; the lab teacher producer pins both it
and this overlay. No selection, backup, quota or recommendation equation changes.
"""
from pathlib import Path
import hashlib

from brass_ai.learning.contracts import SOURCE_ROOT, digest, require
from .lab_contract import SOURCE, native_fingerprint

CONTRACT = 'brass-lab-teacher/1.0.0'


def source_fingerprint():
    paths = [Path(__file__), *sorted((SOURCE_ROOT / 'lab-native').rglob('*.rs')),
             SOURCE_ROOT / 'lab-native/Cargo.toml', SOURCE_ROOT / 'lab-native/Cargo.lock']
    return digest(dict(contract=CONTRACT, base=SOURCE, files={p.relative_to(SOURCE_ROOT).as_posix():
        hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in paths if 'target' not in p.parts}))


def apply(directory):
    directory = Path(directory).resolve()
    require(directory != (SOURCE_ROOT / 'native').resolve(), 'Overlay requires a detached build directory')
    require(native_fingerprint(directory) == SOURCE, 'B04 overlay requires the exact sealed B01 core')
    path = directory / 'brass-ai-core/src/search/session/mod.rs'
    source = path.read_text(encoding='utf-8')
    changes = [
        ('pub mod clock;', 'pub mod teacher;\npub mod clock;'),
        ('pub struct Session {', 'pub struct Session {\n    teacher: Option<teacher::Exports>,'),
        ('            retain_descendants: false,', '            teacher: None,\n            retain_descendants: false,'),
        ('    pub fn budget(&self) -> Budget {', '''    /// Explicit lab-only export. Baseline sessions allocate no export state.
    pub fn enable_teacher_exports(&mut self) -> Result<()> {
        check(self.request == 0 && !self.retain_descendants, "teacher export admission")?;
        self.teacher = Some(teacher::Exports::default());
        Ok(())
    }
    pub fn take_teacher_export(&mut self) -> Option<teacher::Evidence> {
        self.teacher.as_mut().and_then(|t| t.current.take())
    }
    pub fn budget(&self) -> Budget {'''),
        ('    pub fn retire(&mut self) {', '    pub fn retire(&mut self) {\n        self.teacher = None;'),
        ('        if let Some(previous) = self.continuation.take() {',
         '        if let Some(t) = &mut self.teacher { t.activate(&info); }\n        if let Some(previous) = self.continuation.take() {'),
        ('        if !self.budget.delivery_open(clock) {\n            return self.deliver(&game, out, clock);',
         '        if let Some(t) = &mut self.teacher { t.clear(); }\n        if !self.budget.delivery_open(clock) {\n            return self.deliver(&game, out, clock);'),
        ('            if self.retain_descendants && out.status == "policy_round_stop" {',
         '            if let Some(t) = &mut self.teacher { t.capture(&tree, root, &out)?; }\n            if self.retain_descendants && out.status == "policy_round_stop" {'),
    ]
    for before, after in changes:
        require(source.count(before) == 1, 'B04 export hook drift: ' + before[:60])
        source = source.replace(before, after)
    path.write_bytes(source.encode())
    (path.parent / 'teacher.rs').write_bytes((SOURCE_ROOT / 'lab-native/core-overlay/teacher.rs').read_bytes().replace(b'\r\n', b'\n'))
    return dict(contract=CONTRACT, base_source=SOURCE, producer=source_fingerprint(),
                core_source=native_fingerprint(directory), hooks=len(changes))
