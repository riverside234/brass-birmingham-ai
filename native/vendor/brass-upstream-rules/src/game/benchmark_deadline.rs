//! Optional benchmark-only cooperative deadline; ordinary builds are unchanged.
use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

pub(crate) const EXPIRED: &str = "benchmark wall-clock deadline reached";

pub(crate) struct Guard(Option<Instant>);

impl Guard {
    pub(crate) fn new(seconds: f64) -> Result<Self, String> {
        if !seconds.is_finite() || seconds <= 0.0 || seconds > 60.0 {
            return Err("benchmark deadline seconds must be in (0, 60]".to_string());
        }
        let target = Instant::now() + Duration::from_secs_f64(seconds);
        Ok(Self(DEADLINE.with(|slot| slot.replace(Some(target)))))
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        DEADLINE.with(|slot| slot.set(self.0));
    }
}

#[inline]
pub(crate) fn check() -> Result<(), String> {
    if DEADLINE.with(|slot| slot.get().is_some_and(|target| Instant::now() >= target)) {
        Err(EXPIRED.to_string())
    } else {
        Ok(())
    }
}
