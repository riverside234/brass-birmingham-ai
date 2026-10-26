//! Bounded cooperative scheduling. Poll readiness while advancing other paths;
//! never sleep waiting for work that only this owning thread can produce.
use crate::{require, Input, Result};
use brass_ai_core::{
    evaluation::BatchPolicy,
    features::{ACTION_DIM, MAX_CANDIDATES, STATE_DIM},
};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub fn validate_policy(p: BatchPolicy) -> Result<()> {
    require(
        [1, 2, 4].contains(&p.cap)
            && (p.cap..=4).contains(&p.pending_limit)
            && (MAX_CANDIDATES..=MAX_CANDIDATES * 4).contains(&p.max_candidate_cells)
            && [0, 500, 1000, 2000].contains(&p.flush_us),
        "invalid local batch policy",
    )
}
pub struct Pending {
    pub slot: usize,
    pub candidates: usize,
    pub submitted: Instant,
    pub deadline: Instant,
}
pub struct Queue {
    policy: BatchPolicy,
    pending: VecDeque<Pending>,
}
impl Queue {
    pub fn new(policy: BatchPolicy) -> Result<Self> {
        validate_policy(policy)?;
        Ok(Self {
            policy,
            pending: VecDeque::with_capacity(policy.pending_limit),
        })
    }
    pub fn push(&mut self, pending: Pending) -> Result<()> {
        require(
            self.pending.len() < self.policy.pending_limit,
            "inference backpressure: pending queue full",
        )?;
        require(
            (1..=MAX_CANDIDATES).contains(&pending.candidates)
                && !self.pending.iter().any(|p| p.slot == pending.slot),
            "invalid/duplicate pending ticket",
        )?;
        self.pending.push_back(pending);
        Ok(())
    }
    pub fn take_ready(&mut self, now: Instant, producer_can_progress: bool) -> Vec<Pending> {
        let Some(first) = self.pending.front() else {
            return Vec::new();
        };
        let age = now.saturating_duration_since(first.submitted);
        let due = self.pending.iter().any(|p| {
            p.deadline.saturating_duration_since(now) <= Duration::from_micros(self.policy.flush_us)
        });
        if producer_can_progress
            && self.pending.len() < self.policy.cap
            && age < Duration::from_micros(self.policy.flush_us)
            && !due
        {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(self.policy.cap);
        let mut width = 0;
        while let Some(p) = self.pending.front() {
            let next = width.max(p.candidates);
            if result.len() == self.policy.cap
                || next * (result.len() + 1) > self.policy.max_candidate_cells
            {
                break;
            }
            width = next;
            result.push(self.pending.pop_front().expect("checked front"));
        }
        result
    }
}
#[derive(Default)]
pub struct Packed {
    pub states: Vec<f32>,
    pub actions: Vec<f32>,
    pub masks: Vec<bool>,
    pub mappings: Vec<i64>,
    pub remaining: Vec<f32>,
    pub descriptors: Vec<f32>,
}
impl Packed {
    pub fn pack(&mut self, inputs: &[Input<'_>], width: usize) -> usize {
        let b = inputs.len();
        self.states.resize(b * STATE_DIM, 0.);
        self.actions.resize(b * width * ACTION_DIM, 0.);
        self.actions.fill(0.);
        self.masks.resize(b * width, false);
        self.masks.fill(false);
        self.mappings.resize(b * 4, 0);
        self.remaining.resize(b * 4, 0.);
        self.descriptors.resize(b * 136, 0.);
        let mut copied = 0;
        for (i, input) in inputs.iter().enumerate() {
            let t = input.tensors;
            let n = input.choices.len();
            self.states[i * STATE_DIM..(i + 1) * STATE_DIM].copy_from_slice(&t.state);
            self.actions[i * width * ACTION_DIM..i * width * ACTION_DIM + n * ACTION_DIM]
                .copy_from_slice(&t.actions);
            self.masks[i * width..i * width + n].fill(true);
            self.mappings[i * 4..(i + 1) * 4].copy_from_slice(&t.stable_to_relative);
            self.remaining[i * 4..(i + 1) * 4].copy_from_slice(&t.remaining);
            self.descriptors[i * 136..(i + 1) * 136].copy_from_slice(&t.descriptors);
            copied += (STATE_DIM + n * ACTION_DIM + 4 + 136) * 4 + 4 * 8 + n;
        }
        copied
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cap_flush_backpressure_deadline_and_no_producer() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(1);
        let mut q = Queue::new(BatchPolicy {
            cap: 2,
            pending_limit: 2,
            flush_us: 1000,
            ..BatchPolicy::default()
        })
        .unwrap();
        q.push(Pending {
            slot: 0,
            candidates: 2,
            submitted: now,
            deadline,
        })
        .unwrap();
        assert!(q.take_ready(now, true).is_empty());
        assert_eq!(
            q.take_ready(now + Duration::from_micros(1000), true).len(),
            1
        );
        for slot in 0..2 {
            q.push(Pending {
                slot,
                candidates: 8192,
                submitted: now,
                deadline,
            })
            .unwrap();
        }
        assert!(q
            .push(Pending {
                slot: 3,
                candidates: 1,
                submitted: now,
                deadline
            })
            .is_err());
        assert_eq!(q.take_ready(now, true).len(), 1); // cell bound, never truncate
        assert_eq!(q.take_ready(now, false).len(), 1); // no sleep to fill
        q.push(Pending {
            slot: 0,
            candidates: 1,
            submitted: now,
            deadline: now,
        })
        .unwrap();
        assert_eq!(q.take_ready(now, true).len(), 1); // caller retires expired work
    }
}
