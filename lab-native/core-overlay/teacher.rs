//! Read-only final-watermark snapshots; recommendations still use Rounds::eligible.
use super::{
    super::{
        brass::Key,
        math::{Parameters, Statistics},
        tree::{Handle, Tree},
        Info, Result,
    },
    Outcome,
};

#[derive(Clone)]
pub struct Evidence {
    pub work: Option<super::super::domino::Work>,
    pub info: Info<Key, String>,
    pub stats: Statistics,
    pub frozen_n: u64,
    pub completed: u64,
    pub rounds: usize,
    pub effective_k: usize,
    pub root_legal_count: usize,
    pub gumbels: Vec<f64>,
    pub continuation: bool,
}
impl Evidence {
    pub fn policy(&self) -> Result<Vec<f64>> {
        self.stats.policy(self.info.actor, Parameters::default())
    }
}
#[derive(Default)]
pub struct Exports {
    pub current: Option<Evidence>,
    continuation: Option<Evidence>,
}
impl Exports {
    pub fn clear(&mut self) {
        self.current = None;
        self.continuation = None;
    }
    pub fn activate(&mut self, actual: &Info<Key, String>) {
        self.current = None;
        // Typed equality includes observer, legal order, revision, recall,
        // public state, model/reward/context and causal insolvency ledger.
        if self
            .continuation
            .as_ref()
            .is_some_and(|e| e.info == *actual)
        {
            self.current = self.continuation.take();
        }
    }
    pub fn capture(&mut self, tree: &Tree<Key, String>, root: Handle, out: &Outcome) -> Result<()> {
        self.clear();
        let node = tree.get(root)?;
        let evidence = Evidence {
            work: out.teacher_work.clone(),
            info: node.info.clone(),
            stats: node.stats.clone(),
            frozen_n: out.quota.as_ref().map_or(0, |q| q.simulations),
            completed: out.completed,
            rounds: out.snapshot.as_ref().map_or(0, |s| s.completed_rounds),
            effective_k: out.effective_k,
            root_legal_count: node.info.actions.len(),
            gumbels: out.gumbels.clone(),
            continuation: false,
        };
        if let Some(step) = out.actions.get(1).and_then(|a| a.action.steps.first()) {
            if let Some(handle) = tree.find(&step.precondition)? {
                let node = tree.get(handle)?;
                self.continuation = Some(Evidence {
                    info: node.info.clone(),
                    stats: node.stats.clone(),
                    continuation: true,
                    ..evidence.clone()
                });
            }
        }
        self.current = Some(evidence);
        Ok(())
    }
}
