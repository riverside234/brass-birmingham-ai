//! Caller-owned monotonic clock; no receiving/retry path creates more time.
use super::super::{check, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

pub trait Clock {
    fn now_ns(&self) -> u64;
    fn cancelled(&self) -> bool;
}
#[derive(Clone)]
pub struct MonotonicClock {
    origin: Instant,
    pub cancelled: Arc<AtomicBool>,
}
impl MonotonicClock {
    pub fn instant(&self, ns: u64) -> Instant {
        self.origin + std::time::Duration::from_nanos(ns)
    }
    pub fn new(cancelled: Arc<AtomicBool>) -> Self {
        Self {
            origin: Instant::now(),
            cancelled,
        }
    }
}
impl Clock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Budget {
    pub started_ns: u64,
    pub deadline_ns: u64,
    pub work_deadline_ns: u64,
    pub reserve_ns: u64,
}
impl Budget {
    /// The personality SearchSession reference reserves min(1.2s, 10%).
    pub fn new(started_ns: u64, allowance_ns: u64) -> Result<Self> {
        check(
            allowance_ns > 0 && allowance_ns <= 120_000_000_000,
            "turn allowance bounds",
        )?;
        let deadline_ns = started_ns
            .checked_add(allowance_ns)
            .ok_or_else(|| super::super::invalid("turn clock overflow"))?;
        let reserve_ns = (allowance_ns / 10).min(1_200_000_000);
        Ok(Self {
            started_ns,
            deadline_ns,
            work_deadline_ns: deadline_ns - reserve_ns,
            reserve_ns,
        })
    }
    pub fn narrow(&mut self, now: u64, remaining: u64) {
        self.deadline_ns = self.deadline_ns.min(now.saturating_add(remaining));
        self.work_deadline_ns = self
            .work_deadline_ns
            .min(self.deadline_ns.saturating_sub(self.reserve_ns));
    }
    pub fn remaining_work(&self, now: u64) -> u64 {
        self.work_deadline_ns.saturating_sub(now)
    }
    pub fn work_open(&self, c: &impl Clock) -> bool {
        !c.cancelled() && c.now_ns() < self.work_deadline_ns
    }
    pub fn delivery_open(&self, c: &impl Clock) -> bool {
        !c.cancelled() && c.now_ns() < self.deadline_ns
    }
}

#[derive(Default)]
pub struct Costs {
    completed: VecDeque<u64>,
}
impl Costs {
    pub fn observe(&mut self, ns: u64) {
        if ns == 0 {
            return;
        }
        if self.completed.len() == 8 {
            self.completed.pop_front();
        }
        self.completed.push_back(ns);
    }
    pub fn estimate(&self) -> Option<u64> {
        if self.completed.len() < 4 {
            return None;
        }
        let mut values: Vec<_> = self.completed.iter().copied().collect();
        values.sort_unstable();
        Some(values[(values.len() - 1) / 4])
    }
    pub fn admit(&self, remaining: u64) -> bool {
        remaining > 0 && remaining >= self.estimate().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Estimate {
    pub nanoseconds: u64,
    pub source: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct FrozenQuota {
    pub simulations: u64,
    pub available_ns: u64,
    pub estimate: Option<Estimate>,
    pub mode: &'static str,
}
impl FrozenQuota {
    pub fn timed(available_ns: u64, estimate: Option<Estimate>) -> Result<Self> {
        let estimate = estimate.filter(|e| e.nanoseconds > 0 && !e.source.is_empty());
        let simulations = estimate
            .as_ref()
            .map_or(0, |e| available_ns / e.nanoseconds);
        check(
            simulations <= 1_000_000,
            "estimated quota exceeds search bound",
        )?;
        Ok(Self {
            simulations,
            available_ns,
            estimate,
            mode: "deadline",
        })
    }
    pub fn count(simulations: u64, available_ns: u64) -> Result<Self> {
        check(simulations <= 1_000_000, "count quota exceeds search bound")?;
        Ok(Self {
            simulations,
            available_ns,
            estimate: None,
            mode: "count",
        })
    }
}
