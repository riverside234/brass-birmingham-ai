//! V205: conservative first-expansion descendants from one eligible watermark.
//! No worlds, old plan support or old promoted-root counts cross this boundary.
use super::super::tree::{Handle, RetainedCounts};
use super::*;

pub const POLICY: &str = "descendant-reuse/1.0.0";
pub(super) struct Retired {
    pub tree: Tree<Key, String>,
    pub root: Handle,
    pub game: Brass,
    pub watermark: u64,
}
pub(super) struct Ready {
    pub tree: Tree<Key, String>,
    pub root: Handle,
}
#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub policy: &'static str,
    pub hit: bool,
    pub reason: String,
    pub matching_ns: u64,
    pub counts: RetainedCounts,
    pub diagnostic_max_depth: usize,
}
impl Metrics {
    pub fn miss(reason: &str, depth: usize) -> Self {
        Self {
            policy: POLICY,
            hit: false,
            reason: reason.into(),
            matching_ns: 0,
            counts: Default::default(),
            diagnostic_max_depth: depth,
        }
    }
}
impl Retired {
    pub fn promote(
        mut self,
        input: &Input,
        first: &plans::Action,
        budget: Budget,
        clock: &impl Clock,
    ) -> Result<(Ready, RetainedCounts)> {
        check(
            self.game.reusable_sampler() && input.shuffle_tape.is_none(),
            "reuse requires resampled hidden worlds",
        )?;
        // Repeat the certificate on THIS owner's permitted original root. This
        // verifies its conditional allocation law and node-relative value ledger,
        // not merely a matching hash or another owner's prepared input.
        let (certified, boundary) = self.game.project(first, budget, clock)?;
        check(
            certified.view == input.view
                && certified.recall == input.recall
                && certified.context == input.context,
            "reuse projected information/context mismatch",
        )?;
        check(
            certified.ledger.spent == input.ledger.spent
                && certified.ledger.settled == input.ledger.settled
                && certified.reward.identity()? == input.reward.identity()?,
            "reuse utility ledger mismatch",
        )?;
        // Independent ChaCha streams may have different seeds, but both roots
        // must use the same sampler law. Recorded/conditioned tapes are excluded.
        let prefix = first
            .steps
            .iter()
            .map(|s| (s.precondition.clone(), s.choice.clone()))
            .collect::<Vec<_>>();
        let (root, counts) =
            self.tree
                .promote(self.root, &prefix, &boundary, self.watermark, &mut || {
                    crate::work::checkpoint(crate::work::Stage::Traversal)?;
                    check(budget.work_open(clock), "reuse work cutoff")
                })?;
        Ok((
            Ready {
                tree: self.tree,
                root,
            },
            counts,
        ))
    }
}
