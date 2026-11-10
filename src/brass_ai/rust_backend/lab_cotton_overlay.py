"""Small CPU-teacher-only extension of the detached search core.

The execution core gets its own fingerprint; the sealed GPU/game core is never
changed or relabelled. Coverage only touches candidate membership before the
first visit, and only for the exact acting root requested by the worker.
"""
from pathlib import Path
from brass_ai.learning.contracts import SOURCE_ROOT, require
from .lab_contract import native_fingerprint

PATCHES = {
    'brass-ai-core/src/search/session/mod.rs': (
        ('pub mod clock;', 'pub mod cotton;\npub mod clock;'),
        ('pub struct Session {', 'pub struct Session {\n    cotton: Option<cotton::Coverage>,'),
        ('            retain_descendants: false,', '            cotton: None,\n            retain_descendants: false,'),
        ('    pub fn budget(&self) -> Budget {',
         '    /// Root-only teacher exploration; no extra searches or invalidation of cached plans.\n'
         '    pub fn set_teacher_cotton(&mut self, context: &crate::state::observation::ViewContext, limit: usize) -> Result<()> {\n'
         '        check(limit <= 2, "cotton coverage limit")?;\n'
         '        self.cotton = (limit > 0).then(|| cotton::Coverage { context: context.clone(), limit });\n'
         '        Ok(())\n'
         '    }\n'
         '    pub fn budget(&self) -> Budget {'),
        ('            let (mut tree, root) = if let Some(ready) = self.promoted.take() {',
         '            if let Some(c) = &self.cotton {\n'
         '                if info.key.information.observation.view.as_ref().is_some_and(|v| v.context == c.context) {\n'
         '                    rounds.tournament.cover_cotton(&stats.logits, &info.key.choices, c.limit)?;\n'
         '                }\n'
         '            }\n'
         '            let (mut tree, root) = if let Some(ready) = self.promoted.take() {'),
    ),
    'brass-ai-core/src/search/halving.rs': (
        ('    pub fn next_edge(&mut self) -> Result<Option<usize>> {',
         '    pub fn cover_cotton(&mut self, logits: &[f64], choices: &[crate::state::dto::Choice], limit: usize) -> Result<()> {\n'
         '        check(self.completed == 0 && self.issued.is_empty(), "late cotton coverage")?;\n'
         '        self.survivors = super::session::cotton::candidates(logits, &self.gumbels, choices, &self.survivors, limit)?;\n'
         '        Ok(())\n'
         '    }\n'
         '    pub fn next_edge(&mut self) -> Result<Option<usize>> {'),
    ),
}


def apply(directory, root=SOURCE_ROOT):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    require(directory != root/'native', 'Cotton overlay requires a detached core')
    base = native_fingerprint(directory)
    for name, changes in PATCHES.items():
        path = directory/name
        text = path.read_text(encoding='utf-8')
        for before, after in changes:
            require(text.count(before) == 1, 'Cotton overlay drift: '+name+': '+before[:50])
            text = text.replace(before, after)
        path.write_bytes(text.encode())
    target = directory/'brass-ai-core/src/search/session/cotton.rs'
    target.write_bytes((root/'lab-native-cpu/core-overlay/cotton.rs').read_bytes().replace(b'\r\n', b'\n'))
    return dict(base_core=base, execution_core=native_fingerprint(directory))
