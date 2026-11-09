"""Exact source patches for search_v3 rolling halving quotas.

search_v3 changes timed scheduling only: between halving rounds it re-plans the
remaining rounds from the throughput measured at this root. Selection, backup,
Gumbel draws, plan support and every count-mode search are unchanged.
"""
from pathlib import Path

PATCHES = {
    'brass-ai-core/src/search/halving.rs': (
        ('    round_done: u64,\n    issued: std::collections::VecDeque<usize>,\n}\n',
         '    round_done: u64,\n    issued: std::collections::VecDeque<usize>,\n    rolling_quota: Option<u64>,\n}\n'),
        ('            issued: std::collections::VecDeque::new(),\n        })\n',
         '            issued: std::collections::VecDeque::new(),\n            rolling_quota: None,\n        })\n'),
        ('            let quota =\n                self.budget / (self.survivors.len() as u64 * self.effective_k.ilog2() as u64);\n'
         '            quota * self.survivors.len() as u64 - self.round_done\n',
         '            let quota = self.round_quota();\n            quota * self.survivors.len() as u64 - self.round_done\n'),
        ('        let quota = if self.survivors.len() > 1 {\n'
         '            self.budget / (self.survivors.len() as u64 * self.effective_k.ilog2() as u64)\n'
         '        } else {\n            0\n        };\n',
         '        let quota = if self.survivors.len() > 1 {\n            self.round_quota()\n'
         '        } else {\n            0\n        };\n'),
        ('    pub fn survivors(&self) -> &[usize] {\n',
         '    pub fn budget(&self) -> u64 {\n        self.budget\n    }\n'
         '    fn round_quota(&self) -> u64 {\n'
         '        self.rolling_quota.unwrap_or_else(|| {\n'
         '            self.budget / (self.survivors.len() as u64 * self.effective_k.ilog2() as u64)\n'
         '        })\n    }\n'
         '    /// search_v3 only: spread the measured remaining capacity evenly over the\n'
         '    /// rounds still to run, counting what this round already finished. It\n'
         '    /// always leaves this round reachable. Count-mode searches never call it.\n'
         '    pub fn replan(&mut self, remaining_simulations: u64) -> Result<()> {\n'
         '        check(self.issued.is_empty(), "replan only with no pending trajectory")?;\n'
         '        let m = self.survivors.len() as u64;\n'
         '        if m < 2 {\n            return Ok(());\n        }\n'
         '        let rounds = u64::from(m.ilog2());\n'
         '        let target = (self.round_done + remaining_simulations) / rounds / m;\n'
         '        let quota = target.max(self.round_done / m + 1);\n'
         '        self.rolling_quota = Some(quota);\n'
         '        self.budget = self.completed - self.round_done + quota * m * rounds;\n'
         '        Ok(())\n    }\n'
         '    pub fn survivors(&self) -> &[usize] {\n'),
    ),
    'brass-ai-core/src/search/session/tournament.rs': (
        ('    pub eligible: Option<Snapshot>,\n    budget: u64,\n}\n', '    pub eligible: Option<Snapshot>,\n}\n'),
        ('            eligible: None,\n            budget,\n        })\n', '            eligible: None,\n        })\n'),
        ('let reason = if t.completed == self.budget {', 'let reason = if t.completed == t.budget() {'),
    ),
    'brass-ai-core/src/search/session/clock.rs': (
        ('    /// Switch affordability only; the frozen baseline quota estimator is unchanged.\n',
         '    /// search_v3: mean of the recent completed trajectories.\n'
         '    pub fn mean(&self) -> Option<u64> {\n'
         '        (self.completed.len() >= 4)\n'
         '            .then(|| self.completed.iter().sum::<u64>() / self.completed.len() as u64)\n'
         '    }\n'
         '    /// Switch affordability only; the frozen baseline quota estimator is unchanged.\n'),
    ),
    'brass-ai-core/src/search/session/focus.rs': (
        ('    SearchV2FocusOnly,\n    SearchV2,\n}\n', '    SearchV2FocusOnly,\n    SearchV2,\n    SearchV3,\n}\n'),
    ),
    'brass-ai-core/src/search/session/mod.rs': (
        ('pub const IDENTITY: &str = "brass-gumbel-is-mcts/session-v1";\n',
         'pub const IDENTITY: &str = "brass-gumbel-is-mcts/session-v1";\n'
         'pub const IDENTITY_V3: &str = "brass-gumbel-is-mcts/session-v3";\n'),
        ('    pub minimum_plan_support: u64,\n    pub cache_entries: usize,\n}\n',
         '    pub minimum_plan_support: u64,\n    pub cache_entries: usize,\n'
         '    /// search_v3: re-plan halving rounds from measured throughput (timed only).\n'
         '    pub rolling_rounds: bool,\n}\n'),
        ('            minimum_plan_support: 2,\n            cache_entries: 128,\n        }\n',
         '            minimum_plan_support: 2,\n            cache_entries: 128,\n            rolling_rounds: false,\n        }\n'),
        ('            version: IDENTITY,\n',
         '            version: if self.settings.rolling_rounds {\n                IDENTITY_V3\n'
         '            } else {\n                IDENTITY\n            },\n'),
        ('                    &mut observer,\n                    self.settings.minimum_plan_support,\n                )?;\n',
         '                    &mut observer,\n                    self.settings.minimum_plan_support,\n'
         '                    self.settings.rolling_rounds && matches!(mode_kind, ModeKind::Timed),\n                )?;\n'),
        ('            let available = self.budget.remaining_work(clock.now_ns());\n            let quota = match mode {\n',
         '            let available = self.budget.remaining_work(clock.now_ns());\n'
         '            let mode_kind = match &mode {\n                Mode::Count => ModeKind::Count,\n'
         '                Mode::Timed { .. } => ModeKind::Timed,\n            };\n'
         '            let quota = match mode {\n'),
        ('struct CostBucket {\n',
         '#[derive(Clone, Copy)]\nenum ModeKind {\n    Count,\n    Timed,\n}\nstruct CostBucket {\n'),
    ),
    'brass-ai-core/src/search/session/cooperative.rs': (
        ('    observer: &mut Option<&mut dyn focus::Observer>,\n    minimum_support: u64,\n) -> Result<()> {\n',
         '    observer: &mut Option<&mut dyn focus::Observer>,\n    minimum_support: u64,\n    rolling: bool,\n) -> Result<()> {\n'),
        ('    let n = out.quota.as_ref().expect("frozen quota").simulations;\n'
         '    let mut published_round = 0;\n    while rounds.tournament.completed < n {\n',
         '    let mut published_round = 0;\n    while rounds.tournament.completed < rounds.tournament.budget() {\n'),
        ('        checkpoint(Stage::Traversal).or_else(|error| {\n',
         '        if rolling && rounds.tournament.pending() == 0 {\n'
         '            // Recent trajectories, because later rounds run deeper. The\n'
         '            // margin lets the last round finish inside the work deadline.\n'
         '            if let Some(cost) = costs.mean() {\n'
         '                let remaining = budget.remaining_work(clock.now_ns());\n'
         '                rounds.tournament.replan(remaining / 16 * 15 / cost.max(1))?;\n'
         '            }\n'
         '        }\n'
         '        checkpoint(Stage::Traversal).or_else(|error| {\n'),
    ),
    'brass-ai-worker/src/pair.rs': (
        ('        let (focus, mut progress) = if binding.method != session::focus::Method::SearchV1 {\n',
         '        let (focus, mut progress) = if !matches!(\n            binding.method,\n'
         '            session::focus::Method::SearchV1 | session::focus::Method::SearchV3\n        ) {\n'),
        ('            Settings {\n                search: config,\n                ..Default::default()\n            },\n',
         '            Settings {\n                search: config,\n'
         '                rolling_rounds: b.method == session::focus::Method::SearchV3,\n'
         '                ..Default::default()\n            },\n'),
    ),
}

TESTS = {
    'brass-ai-core/src/search/halving.rs': '''

#[cfg(test)]
mod rolling_tests {
    use super::*;

    fn stats(n: usize) -> Statistics {
        Statistics::new(vec![0.; n], [0.; 4]).unwrap()
    }

    fn tournament(k: usize, budget: u64) -> Tournament {
        let logits = vec![0.; 16];
        let uniforms: Vec<f64> = (0..16).map(|i| (i as f64 + 0.5) / 16.).collect();
        Tournament::new(&logits, &uniforms, k, budget).unwrap()
    }

    fn run_round(t: &mut Tournament) {
        let before = t.rounds.len();
        let mut s = stats(16);
        let p = Parameters { c1: 50., c2: 0.1 };
        while t.rounds.len() == before {
            let edge = t.next_edge().unwrap().unwrap();
            s.count[edge] += 1;
            t.completed_backup(edge, &s, 0, p).unwrap();
        }
    }

    #[test]
    fn without_a_replan_the_legacy_quotas_are_exact() {
        let mut t = tournament(16, 256);
        assert_eq!(t.round_quota(), 4);
        run_round(&mut t);
        assert_eq!((t.completed, t.survivors().len()), (64, 8));
        assert_eq!(t.round_quota(), 8);
        assert_eq!(t.budget(), 256);
    }

    #[test]
    fn replan_spreads_remaining_capacity_over_the_rounds_left() {
        let mut t = tournament(16, 256);
        run_round(&mut t);
        t.replan(600).unwrap();
        // Eight survivors and three rounds: 600 / 3 / 8 = 25 visits each.
        assert_eq!(t.round_quota(), 25);
        assert_eq!(t.budget(), 64 + 25 * 8 * 3);
        run_round(&mut t);
        assert_eq!((t.completed, t.survivors().len()), (264, 4));
    }

    #[test]
    fn a_replan_inside_a_round_keeps_its_equal_share_stable() {
        let mut t = tournament(16, 256);
        run_round(&mut t);
        let mut s = stats(16);
        let p = Parameters { c1: 50., c2: 0.1 };
        t.replan(600).unwrap();
        assert_eq!(t.round_quota(), 25);
        for done in 1..=40u64 {
            let edge = t.next_edge().unwrap().unwrap();
            s.count[edge] += 1;
            t.completed_backup(edge, &s, 0, p).unwrap();
            // Capacity falls as visits complete, so the round target must not move.
            t.replan(600 - done).unwrap();
            assert_eq!(t.round_quota(), 25, "after {done} visits");
        }
    }

    #[test]
    fn a_slow_machine_shrinks_later_rounds_but_keeps_one_visit_each() {
        let mut t = tournament(16, 256);
        run_round(&mut t);
        t.replan(3).unwrap();
        assert_eq!(t.round_quota(), 1);
        assert_eq!(t.budget(), 64 + 24);
    }

    #[test]
    fn replanning_the_last_round_leaves_no_remainder_for_the_winner() {
        let mut t = tournament(4, 64);
        run_round(&mut t);
        assert_eq!(t.survivors().len(), 2);
        t.replan(50).unwrap();
        assert_eq!(t.round_quota(), 25);
        run_round(&mut t);
        assert_eq!(t.survivors().len(), 1);
        assert_eq!(t.completed, t.budget());
        assert_eq!(t.next_edge().unwrap(), None);
    }

    #[test]
    fn a_running_round_can_shrink_but_always_stays_reachable() {
        let mut t = tournament(4, 64);
        let mut s = stats(16);
        let p = Parameters { c1: 50., c2: 0.1 };
        for _ in 0..3 {
            let edge = t.next_edge().unwrap().unwrap();
            s.count[edge] += 1;
            t.completed_backup(edge, &s, 0, p).unwrap();
        }
        // No capacity left: the round still ends after one more visit.
        t.replan(0).unwrap();
        assert_eq!(t.round_quota(), 1);
        assert_eq!(t.survivors().len(), 4);
        let edge = t.next_edge().unwrap().unwrap();
        s.count[edge] += 1;
        t.completed_backup(edge, &s, 0, p).unwrap();
        assert_eq!((t.survivors().len(), t.rounds.len()), (2, 1));
    }

    #[test]
    fn a_running_round_grows_when_time_allows() {
        let mut t = tournament(2, 8);
        assert_eq!(t.round_quota(), 4);
        let mut s = stats(16);
        let p = Parameters { c1: 50., c2: 0.1 };
        for _ in 0..5 {
            let edge = t.next_edge().unwrap().unwrap();
            s.count[edge] += 1;
            t.completed_backup(edge, &s, 0, p).unwrap();
        }
        t.replan(16).unwrap();
        assert_eq!(t.round_quota(), (5 + 16) / 2);
        run_round(&mut t);
        assert_eq!(t.completed, 20);
        assert_eq!(t.completed, t.budget());
    }

    #[test]
    fn replan_is_rejected_with_a_pending_trajectory() {
        let mut t = tournament(4, 64);
        let edge = t.next_edge().unwrap().unwrap();
        assert!(t.replan(100).is_err());
        let mut s = stats(16);
        s.count[edge] += 1;
        t.completed_backup(edge, &s, 0, Parameters { c1: 50., c2: 0.1 }).unwrap();
        assert!(t.replan(100).is_ok());
    }
}
''',
}


def apply(native):
    """Apply to a native tree; anchors must match once, so drift fails loudly."""
    native = Path(native)
    for name, changes in PATCHES.items():
        path = native/name
        text = path.read_bytes().replace(b'\r\n', b'\n').decode('utf-8')
        for before, after in changes:
            if text.count(before) != 1:
                raise ValueError(f'search_v3 patch drift in {name}: {before[:60]!r}')
            text = text.replace(before, after)
        text += TESTS.get(name, '')
        path.write_bytes(text.encode('utf-8'))
