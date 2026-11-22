//! Local release 1.4.63: the timed search's startup estimate by era stage.
//!
//! A turn's first search is sized from this estimate: quota = available time / estimate, per lane. The cost of a
//! simulation depends on the position far more than on the engine, because the network scores every legal choice
//! at every leaf. So the wide early-Canal roots cost about twice what most later roots cost, and one estimate for
//! the whole game must be set by them. Four stages: rounds 1-4 and 5-8 of each era. NN seats play four-player games,
//! which have eight rounds per era.
use brass_ai_core::{search::session::clock::Estimate, types::Era};
use serde::{Deserialize, Serialize};

pub const RANGE: std::ops::RangeInclusive<u64> = 1_000_000..=100_000_000;
pub const STAGES: [&str; 4] = ["canal_early", "canal_late", "rail_early", "rail_late"];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StartupByStage {
    pub canal_early: u64,
    pub canal_late: u64,
    pub rail_early: u64,
    pub rail_late: u64,
}

impl StartupByStage {
    pub fn values(&self) -> [u64; 4] {
        [self.canal_early, self.canal_late, self.rail_early, self.rail_late]
    }

    /// Every stage in range and none above the package's single estimate, which stays the fallback.
    pub fn valid(&self, fallback_ns: u64) -> bool {
        self.values().iter().all(|ns| RANGE.contains(ns) && *ns <= fallback_ns)
    }

    /// The stage of a round: 1-4 early, 5-8 late; none outside a four-player era.
    pub fn stage(era: Era, round: u8) -> Option<usize> {
        match (era, round) {
            (Era::CanalEra, 1..=4) => Some(0),
            (Era::CanalEra, 5..=8) => Some(1),
            (Era::RailEra, 1..=4) => Some(2),
            (Era::RailEra, 5..=8) => Some(3),
            _ => None,
        }
    }
}

/// The estimate for a turn's first search: its stage's value when the package has a table, else the single one.
pub fn estimate(table: Option<&StartupByStage>, startup_ns: u64, source: &str, era: Era, round: u8) -> Estimate {
    match (table, StartupByStage::stage(era, round)) {
        (Some(table), Some(stage)) => Estimate {
            nanoseconds: table.values()[stage],
            source: format!("{source}; {}", STAGES[stage]),
        },
        _ => Estimate {
            nanoseconds: startup_ns,
            source: source.into(),
        },
    }
}

#[cfg(test)]
mod local_startup_tests {
    use super::*;

    fn table() -> StartupByStage {
        StartupByStage { canal_early: 6_000_000, canal_late: 4_000_000, rail_early: 5_000_000, rail_late: 3_000_000 }
    }

    #[test]
    fn rounds_one_to_four_are_early_and_five_to_eight_late_in_each_era() {
        let expected = [
            (Era::CanalEra, 1, Some(0)), (Era::CanalEra, 4, Some(0)), (Era::CanalEra, 5, Some(1)), (Era::CanalEra, 8, Some(1)),
            (Era::RailEra, 1, Some(2)), (Era::RailEra, 4, Some(2)), (Era::RailEra, 5, Some(3)), (Era::RailEra, 8, Some(3)),
            (Era::CanalEra, 0, None), (Era::RailEra, 9, None),
        ];
        for (era, round, stage) in expected {
            assert_eq!(StartupByStage::stage(era, round), stage, "{era:?} round {round}");
        }
    }

    #[test]
    fn each_stage_reads_its_own_estimate_and_others_fall_back() {
        let t = table();
        for (era, round, ns, name) in [
            (Era::CanalEra, 2, 6_000_000, "canal_early"), (Era::CanalEra, 6, 4_000_000, "canal_late"),
            (Era::RailEra, 3, 5_000_000, "rail_early"), (Era::RailEra, 7, 3_000_000, "rail_late"),
        ] {
            let e = estimate(Some(&t), 7_000_000, "calibration", era, round);
            assert_eq!((e.nanoseconds, e.source), (ns, format!("calibration; {name}")));
        }
        let e = estimate(None, 7_000_000, "calibration", Era::RailEra, 7);
        assert_eq!((e.nanoseconds, e.source.as_str()), (7_000_000, "calibration"));
        let e = estimate(Some(&t), 7_000_000, "calibration", Era::RailEra, 9);
        assert_eq!(e.nanoseconds, 7_000_000);
    }

    #[test]
    fn tables_out_of_range_above_the_fallback_or_incomplete_are_rejected() {
        let t = table();
        assert!(t.valid(6_000_000));
        assert!(!t.valid(5_999_999)); // a stage above the single estimate
        assert!(!StartupByStage { rail_late: 999_999, ..table() }.valid(7_000_000));
        assert!(!StartupByStage { canal_early: 100_000_001, ..table() }.valid(200_000_000));
        let parsed: StartupByStage = serde_json::from_str(
            r#"{"canal_early":6000000,"canal_late":4000000,"rail_early":5000000,"rail_late":3000000}"#).unwrap();
        assert_eq!(parsed, t);
        assert!(serde_json::from_str::<StartupByStage>(r#"{"canal_early":1,"canal_late":1,"rail_early":1}"#).is_err());
        assert!(serde_json::from_str::<StartupByStage>(
            r#"{"canal_early":1,"canal_late":1,"rail_early":1,"rail_late":1,"round_1":1}"#).is_err());
    }
}
