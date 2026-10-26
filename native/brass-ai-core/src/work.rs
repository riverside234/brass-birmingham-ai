//! Explicit owner-thread work scope. Deep rules calls poll without a global lock.
//! A scope is never transferred to another thread; unwinding restores its parent.
use crate::CoreError;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(usize)]
pub enum Stage {
    Generation,
    Sampling,
    Observation,
    Application,
    Traversal,
    Encoding,
    Expansion,
    Backup,
    Publication,
    Completion,
    Inference,
}
pub type Probe = Arc<dyn Fn(Stage) -> Result<(), CoreError> + Send + Sync>;
#[derive(Clone)]
pub struct Control {
    pub deadline: Instant,
    pub cancelled: Arc<AtomicBool>,
    pub probe: Option<Probe>,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct PollStats {
    pub calls: [u64; 11],
    pub max_unchecked_ns: u64,
    pub owned_copies: u64,
    pub owned_copy_ns: u64,
}
struct Active {
    control: Control,
    stats: PollStats,
    last: Instant,
}
thread_local! { static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) }; }
/// An active scope cannot move away from the thread whose deep calls it guards.
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<brass_ai_core::work::Scope>();
/// ```
pub struct Scope {
    previous: Option<Active>,
    owner: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Scope {
    pub fn narrow_remaining(remaining_ns: u64) -> Option<Self> {
        let control = ACTIVE.with(|a| {
            a.borrow().as_ref().map(|a| {
                let mut c = a.control.clone();
                c.deadline = c
                    .deadline
                    .min(Instant::now() + std::time::Duration::from_nanos(remaining_ns));
                c
            })
        });
        control.map(Self::enter)
    }
    pub fn enter(control: Control) -> Self {
        let previous = ACTIVE.with(|a| {
            a.replace(Some(Active {
                control,
                stats: PollStats::default(),
                last: Instant::now(),
            }))
        });
        Self {
            previous,
            owner: std::marker::PhantomData,
        }
    }
    pub fn stats(&self) -> PollStats {
        ACTIVE.with(|a| {
            a.borrow()
                .as_ref()
                .map_or_else(PollStats::default, |a| a.stats.clone())
        })
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|a| {
            let current = a.take();
            if let (Some(parent), Some(child)) = (&mut self.previous, current) {
                for (p, c) in parent.stats.calls.iter_mut().zip(child.stats.calls) {
                    *p += c;
                }
                parent.stats.max_unchecked_ns = parent
                    .stats
                    .max_unchecked_ns
                    .max(child.stats.max_unchecked_ns);
                parent.stats.owned_copies += child.stats.owned_copies;
                parent.stats.owned_copy_ns += child.stats.owned_copy_ns;
                parent.last = Instant::now();
            }
            a.replace(self.previous.take());
        });
    }
}
/// Every polling site is a bounded chunk. ORT is synchronous and separately timed;
/// its return must be checked before any cache, expansion or backup publication.
pub fn checkpoint(stage: Stage) -> Result<(), CoreError> {
    ACTIVE.with(|cell| {
        let mut active = cell.borrow_mut();
        let Some(a) = active.as_mut() else {
            return Ok(());
        };
        let now = Instant::now();
        a.stats.calls[stage as usize] += 1;
        a.stats.max_unchecked_ns = a
            .stats
            .max_unchecked_ns
            .max(now.duration_since(a.last).as_nanos().min(u64::MAX as u128) as u64);
        a.last = now;
        if let Some(probe) = &a.control.probe {
            probe(stage)?;
        }
        if a.control.cancelled.load(Ordering::Acquire) || Instant::now() >= a.control.deadline {
            return Err(CoreError::Invalid("work cancelled/deadline".into()));
        }
        Ok(())
    })
}

/// Exclude synchronous runtime time from the *CPU polling* interval, not from
/// the turn clock. The caller separately records runtime retirement latency.
pub fn runtime_returned() {
    ACTIVE.with(|a| {
        if let Some(a) = a.borrow_mut().as_mut() {
            a.last = Instant::now();
        }
    });
}
pub fn state_copied(start: Instant) {
    ACTIVE.with(|a| {
        if let Some(a) = a.borrow_mut().as_mut() {
            a.stats.owned_copies += 1;
            a.stats.owned_copy_ns += start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        }
    });
}
