//! g3471 expanded search rows (recovery/search_rows.py): extra decision states of a search-teacher table,
//! for the search-value head and the era heads. Off unless the job's teacher revision names them.
//!
//! Besides the 64 turn-start roots the table already reports, a table records:
//! - "second": the state before every played later main action of a searched turn (its second action);
//! - "sampled": in each era, a uniform sample of up to 24 of the other played decisions with two or more
//!   legal choices (a develop's second tile, a sale, a payment source, a bonus, a liquidation): the
//!   decisions with the smallest keys of a splitmix64 stream keyed by the job's sampling seed and the
//!   decision's revision. The stream is its own, so no search, policy or exploration draw changes, and each
//!   decision is seen once. Only decisions entering the sample are encoded.
//! Nothing new is searched: the extra rows' targets bootstrap from the table's existing turn-start search
//! values. The finish message carries every candidate revision, so admission recomputes the sample.
use crate::{check, cpu_eval::mix, Result};
use brass_ai_core::types::Era;
use serde_json::{json, Value};
use std::cell::RefCell;

pub const CONTRACT: &str = "expanded-search-rows/1.0.0";
pub const KEY: &str = "search_rows";
pub const PER_ERA: u64 = 24;
/// The sample stream's own domain, apart from the exploration stream (0x1491...).
const STREAM: u64 = 0x3471_5EA2_0000_0000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Second,
    Sampled { era: usize, key: u64 },
}

struct Capture {
    per_era: usize,
    seed: u64,
    second: Vec<Value>,
    sampled: [Vec<(u64, u32, Value)>; 2],
    candidates: [Vec<u32>; 2],
}

thread_local! {
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}

/// The reviewed job block's sample size; anything else is refused.
pub fn settings(block: &Value) -> Result<usize> {
    check(block == &json!({"contract": CONTRACT, "sampled_per_era": PER_ERA}), "unreviewed search-row capture")?;
    Ok(PER_ERA as usize)
}

/// Start a table: on with the job's sample size, or off.
pub fn install(per_era: Option<usize>, seed: u64) {
    CAPTURE.with(|cell| {
        *cell.borrow_mut() = per_era.map(|per_era| Capture {
            per_era,
            seed,
            second: Vec::new(),
            sampled: [Vec::new(), Vec::new()],
            candidates: [Vec::new(), Vec::new()],
        })
    });
}

/// A decision's sample key: its own stream, keyed by the job seed and the decision's revision.
pub fn key(seed: u64, revision: u32) -> u64 {
    mix(seed ^ mix(u64::from(revision) ^ STREAM))
}

/// What to record for this played decision; call once per decision, before it is applied.
/// `main`: an Action-phase decision of a searched turn; `first`: the turn's first one (its root, which
/// is already a value row); `turn`: the index of the turn's root (0 before the first root).
pub fn kind(era: Era, revision: u32, main: bool, first: bool, legal: usize, turn: u64) -> Result<Option<Kind>> {
    CAPTURE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(c) = guard.as_mut() else { return Ok(None) };
        if turn == 0 || (main && first) {
            return Ok(None);
        }
        if main {
            return Ok(Some(Kind::Second));
        }
        if legal < 2 {
            return Ok(None);
        }
        let era = usize::from(era == Era::RailEra);
        check(c.candidates[era].last().is_none_or(|&r| r < revision), "captured decisions must advance")?;
        c.candidates[era].push(revision);
        let key = key(c.seed, revision);
        let kept = &c.sampled[era];
        if kept.len() >= c.per_era && kept.iter().all(|&(k, r, _)| (key, revision) > (k, r)) {
            return Ok(None);
        }
        Ok(Some(Kind::Sampled { era, key }))
    })
}

/// Keep the encoded row of a decision `kind` selected; a full sample drops its largest key.
pub fn record(kind: Kind, mut row: Value) -> Result<()> {
    CAPTURE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let c = guard.as_mut().ok_or("search-row capture is off")?;
        match kind {
            Kind::Second => {
                row["kind"] = json!("second");
                c.second.push(row);
            }
            Kind::Sampled { era, key } => {
                row["kind"] = json!("sampled");
                let revision = u32::try_from(row["revision"].as_u64().ok_or("captured revision")?)?;
                let kept = &mut c.sampled[era];
                if kept.len() >= c.per_era {
                    let worst = (0..kept.len()).max_by_key(|&i| (kept[i].0, kept[i].1)).ok_or("empty decision sample")?;
                    kept.swap_remove(worst);
                }
                kept.push((key, revision, row));
            }
        }
        Ok(())
    })
}

/// The finish message's block (sampled rows in revision order), or Null when the capture is off.
pub fn finish() -> Value {
    CAPTURE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(c) = guard.as_mut() else { return Value::Null };
        let mut sampled: Vec<_> = c.sampled.iter_mut().flat_map(std::mem::take).collect();
        sampled.sort_by_key(|&(_, revision, _)| revision);
        json!({"contract": CONTRACT, "sampled_per_era": c.per_era,
            "second": std::mem::take(&mut c.second),
            "sampled": sampled.into_iter().map(|(_, _, row)| row).collect::<Vec<_>>(),
            "candidates": {"CanalEra": c.candidates[0], "RailEra": c.candidates[1]}})
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decide(revision: u32, era: Era, main: bool, first: bool, legal: usize, turn: u64) -> Option<Kind> {
        let kind = kind(era, revision, main, first, legal, turn).unwrap();
        if let Some(k) = kind {
            record(k, json!({"revision": revision})).unwrap();
        }
        kind
    }

    #[test]
    fn the_sample_is_the_smallest_keys_and_roots_and_single_choices_are_skipped() {
        install(Some(3), 77);
        assert_eq!(decide(1, Era::CanalEra, true, true, 5, 1), None); // a turn's root
        assert_eq!(decide(2, Era::CanalEra, true, false, 5, 1), Some(Kind::Second));
        assert_eq!(decide(3, Era::CanalEra, false, false, 1, 1), None); // forced
        for revision in 4..40 {
            decide(revision, if revision < 20 { Era::CanalEra } else { Era::RailEra }, false, false, 3, 1);
        }
        let out = finish();
        assert_eq!(out["second"].as_array().unwrap().len(), 1);
        assert_eq!(out["second"][0]["kind"], "second");
        for (era, range) in [("CanalEra", 4..20u32), ("RailEra", 20..40u32)] {
            let candidates: Vec<u32> = range.collect();
            assert_eq!(out["candidates"][era], json!(candidates));
            let mut expected = candidates.clone();
            expected.sort_by_key(|&r| (key(77, r), r));
            expected.truncate(3);
            expected.sort();
            let got: Vec<u32> = out["sampled"].as_array().unwrap().iter().map(|r| r["revision"].as_u64().unwrap() as u32)
                .filter(|r| candidates.contains(r)).collect();
            assert_eq!(got, expected);
        }
        install(None, 0);
        assert_eq!(decide(50, Era::RailEra, false, false, 3, 2), None);
        assert!(finish().is_null());
    }

    #[test]
    fn only_the_reviewed_block_is_accepted() {
        assert_eq!(settings(&json!({"contract": CONTRACT, "sampled_per_era": 24})).unwrap(), 24);
        assert!(settings(&json!({"contract": CONTRACT, "sampled_per_era": 12})).is_err());
        assert!(settings(&json!({"contract": CONTRACT})).is_err());
    }

    #[test]
    fn the_stream_matches_its_python_twin() {
        // recovery/search_rows.py sample_key(77, 1234) and sample_key(2**63 + 5, 9).
        assert_eq!(key(77, 1234), 0xc0c9_a807_0698_9395);
        assert_eq!(key((1 << 63) + 5, 9), 0x854a_a481_66fe_6ecf);
    }
}
