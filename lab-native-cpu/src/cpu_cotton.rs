//! Bounded, teacher-only cotton exploration. All counters stay in-process until
//! the table receipt; no hot-loop JSON, IPC, hashes or extra NN forwards.
use crate::{check, Result};
use brass_ai_core::{
    state::{
        dto::{Choice, Industry, Kind},
        observation::{Observation, ViewContext},
    },
    types::{Era, Phase},
};
use serde::Serialize;
use serde_json::{json, Value};

pub const CONTRACT: &str = "teacher-cotton-exploration/1.0.0";
const DEVELOP_CONTRACT: &str = "teacher-cotton-exploration/2.0.0";
const MIXED_CONTRACT: &str = "teacher-cotton-exploration/3.0.0";
const MIXED_25_CONTRACT: &str = "teacher-cotton-exploration/4.0.0";
const SEATED_CONTRACT: &str = "teacher-cotton-exploration/5.0.0";
pub const CAP: u8 = 3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Builds,
    Develop,
    Mixed,
    Mixed25,
    Seated,
}

impl Mode {
    pub fn develop(self) -> bool {
        self != Self::Builds
    }

    fn contract(self) -> &'static str {
        match self {
            Self::Builds => CONTRACT,
            Self::Develop => DEVELOP_CONTRACT,
            Self::Mixed => MIXED_CONTRACT,
            Self::Mixed25 => MIXED_25_CONTRACT,
            Self::Seated => SEATED_CONTRACT,
        }
    }

    fn rate(self) -> (u128, u128) {
        match self {
            Self::Builds => (1, 5),
            Self::Develop | Self::Mixed25 => (1, 4),
            Self::Mixed | Self::Seated => (3, 10),
        }
    }

    fn probability(self) -> f64 {
        match self {
            Self::Builds => 0.2,
            Self::Develop | Self::Mixed25 => 0.25,
            Self::Mixed | Self::Seated => 0.3,
        }
    }

    pub fn settings(self) -> Value {
        let scope = if matches!(self, Self::Mixed | Self::Mixed25 | Self::Seated) { "train-and-heldout" } else { "train" };
        json!({"contract":self.contract(),"probability":self.probability(),"max_per_seat_era":CAP,
            "long_candidates":2,"short_candidates":1,"pause_unflipped":true,"scope":scope})
    }
}

pub fn cotton(c: &Choice) -> bool {
    c.kind == Kind::Build && c.industry == Some(Industry::CottonMill)
}
fn develop_cotton(c: &Choice) -> bool {
    c.kind == Kind::Develop && c.industry == Some(Industry::CottonMill)
}
fn targeted(c: &Choice, develop: bool) -> bool {
    cotton(c) || develop && develop_cotton(c)
}
fn is_zero(n: &u64) -> bool {
    *n == 0
}

fn era(e: Era) -> usize {
    usize::from(e == Era::RailEra)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Binding {
    revision: u32,
    seat: usize,
    era: usize,
}
impl Binding {
    fn of(c: &ViewContext) -> Self {
        Self {
            revision: c.revision,
            seat: c.actor as usize,
            era: era(c.era),
        }
    }
}
#[derive(Default, Debug, Serialize)]
pub struct Counts {
    pub action_decisions: u64,
    pub legal_decisions: u64,
    pub eligible_decisions: u64,
    pub blocked_window: u64,
    pub blocked_cap: u64,
    pub blocked_unflipped: u64,
    pub covered_decisions: u64,
    pub covered_locations: u64,
    pub cotton_visits: u64,
    pub total_visits: u64,
    pub no_evidence: u64,
    pub no_cotton_visits: u64,
    pub triggers: u64,
    pub targeted_builds: u64,
    pub actual_builds: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub targeted_develops: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub actual_develops: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub continuation_targets: u64,
    pub base_cotton_probability_sum: f64,
    pub mixed_cotton_probability_sum: f64,
}
#[derive(Clone, Debug)]
pub struct Selection {
    contract: &'static str,
    pub targeted: bool,
    pub probability: f64,
    pub used_before: u8,
    pub cotton_visits: u64,
    pub total_visits: u64,
    pub covered_locations: usize,
    pub choice: String,
}
impl Selection {
    pub fn label(&self) -> Value {
        json!({"contract":self.contract,"targeted":self.targeted,"probability":self.probability,
            "used_before":self.used_before,"cotton_visits":self.cotton_visits,"total_visits":self.total_visits,
            "covered_locations":self.covered_locations})
    }
}
#[derive(Default, Debug)]
pub struct State {
    pub counts: [Counts; 2],
    pub used: [[u8; 4]; 2],
    pub built: [[u32; 4]; 2],
    developed: [[u32; 4]; 2],
    pub mode: Mode,
    pub seats: Option<u8>,
    current: Option<Binding>,
    committed: Option<Binding>,
    eligible: bool,
    profile: usize,
    pub selection: Option<Selection>,
    continuation_choice: Option<String>,
}
impl State {
    fn allowed_phase(&self, phase: Phase) -> bool {
        phase == Phase::Action || self.mode.develop() && phase == Phase::Continuation
    }

    /// Repeated preparation of one revision is idempotent. No allowance is
    /// spent here: a cancelled search or rejected action cannot consume it.
    pub fn prepare(
        &mut self,
        view: &Observation,
        choices: &[Choice],
        fast: bool,
        window: bool,
    ) -> usize {
        let phase = view.context.phase;
        if !self.allowed_phase(phase) {
            self.selection = None;
            self.eligible = false;
            return 0;
        }
        let binding = Binding::of(&view.context);
        if self.current == Some(binding) {
            return if self.eligible && phase == Phase::Action {
                if fast {
                    1
                } else {
                    2
                }
            } else {
                0
            };
        }
        self.current = Some(binding);
        self.selection = None;
        self.continuation_choice = None;
        self.eligible = false;
        self.profile = usize::from(fast);
        if self.seats.is_some_and(|mask| mask & (1 << binding.seat) == 0) {
            return 0;
        }
        let pause = || {
            view.slots.iter().any(|s| {
                s.tile.as_ref().is_some_and(|t| {
                    t.seat == view.seat && t.industry == Industry::CottonMill && !t.flipped
                })
            })
        };
        if phase == Phase::Continuation {
            self.eligible = window
                && self.used[binding.era][binding.seat] < CAP
                && choices.len() > 1
                && choices.iter().any(develop_cotton)
                && !pause();
            return 0;
        }
        let develop = self.mode.develop();
        let c = &mut self.counts[self.profile];
        c.action_decisions += 1;
        if choices.len() < 2 || !choices.iter().any(|c| targeted(c, develop)) {
            return 0;
        }
        c.legal_decisions += 1;
        if !window {
            c.blocked_window += 1;
            return 0;
        }
        if self.used[binding.era][binding.seat] >= CAP {
            c.blocked_cap += 1;
            return 0;
        }
        if pause() {
            c.blocked_unflipped += 1;
            return 0;
        }
        c.eligible_decisions += 1;
        self.eligible = true;
        if fast {
            1
        } else {
            2
        }
    }

    pub fn no_evidence(&mut self) {
        if self.eligible {
            self.counts[self.profile].no_evidence += 1;
        }
    }

    fn trigger(&self, seed: u64, decision: u64) -> bool {
        let (successes, trials) = self.mode.rate();
        u128::from(super::cpu_eval::mix(
            seed ^ super::cpu_eval::mix(decision ^ 0xc077_0a01),
        )) * trials
            < (successes << 64)
    }

    /// The second removal is a legal continuation choice within the same action.
    /// Use the cached plan as the base and spend no additional search budget.
    pub fn sample_continuation(
        &mut self,
        planned: String,
        choices: &[Choice],
        seed: u64,
        decision: u64,
    ) -> String {
        if self.eligible && self.trigger(seed, decision) {
            if let Some(choice) = choices.iter().find(|c| develop_cotton(c)) {
                self.continuation_choice = Some(choice.id.clone());
                return choice.id.clone();
            }
        }
        planned
    }

    /// Called after search, and AFTER candidate coverage. The mixture's exact
    /// behavior probability mixes the root visit weights with the cotton subset.
    /// Separate keyed streams leave search and the existing baseline draw intact.
    pub fn sample(
        &mut self,
        base: usize,
        choices: &[Choice],
        counts: &[u64],
        total: u64,
        tempered: Option<(&[f64], f64)>,
        seed: u64,
        decision: u64,
    ) -> Result<usize> {
        if !self.eligible {
            return Ok(base);
        }
        check(
            counts.len() == choices.len() && base < choices.len() && total > 0,
            "cotton sampling shape",
        )?;
        let develop = self.mode.develop();
        let mut cotton_total = 0;
        let mut cotton_mass = 0.;
        let mut locations = Vec::with_capacity(2);
        for (i, (choice, n)) in choices.iter().zip(counts).enumerate() {
            if *n > 0 && targeted(choice, develop) {
                cotton_total += n;
                if let Some((weights, _)) = tempered {
                    cotton_mass += weights[i];
                }
                let location = (choice.kind, choice.target.as_deref());
                if !locations.contains(&location) {
                    locations.push(location);
                }
            }
        }
        let trigger = self.trigger(seed, decision);
        let mixture = self.mode.probability();
        let c = &mut self.counts[self.profile];
        if cotton_total == 0 {
            c.no_cotton_visits += 1;
            return Ok(base);
        }
        c.covered_decisions += 1;
        c.covered_locations += locations.len() as u64;
        c.cotton_visits += cotton_total;
        c.total_visits += total;
        let base_p = tempered.map_or(cotton_total as f64 / total as f64, |(_, mass)| cotton_mass / mass);
        c.base_cotton_probability_sum += base_p;
        c.mixed_cotton_probability_sum += (1. - mixture) * base_p + mixture;
        let mut selected = base;
        if trigger {
            let z = super::cpu_eval::mix(seed ^ super::cpu_eval::mix(decision ^ 0xc077_0a02));
            if let Some((weights, _)) = tempered {
                let eligible = weights.iter().copied().enumerate().filter(|(i, _)| targeted(&choices[*i], develop));
                selected = super::cpu_eval::sample_weights(eligible, cotton_mass, z)?;
            } else {
                let draw = ((u128::from(z) * u128::from(cotton_total)) >> 64) as u64;
                let mut sum = 0;
                for (i, (choice, n)) in choices.iter().zip(counts).enumerate() {
                    if targeted(choice, develop) {
                        sum += n;
                        if draw < sum {
                            selected = i;
                            break;
                        }
                    }
                }
            }
            c.triggers += 1;
        }
        let binding = self.current.ok_or("cotton choice without preparation")?;
        let (selected_mass, full_mass, cotton_mass) = tempered.map_or(
            (counts[selected] as f64, total as f64, cotton_total as f64),
            |(weights, mass)| (weights[selected], mass, cotton_mass));
        let probability = (1. - mixture) * selected_mass / full_mass
            + if targeted(&choices[selected], develop) { mixture * selected_mass / cotton_mass } else { 0. };
        self.selection = Some(Selection {
            contract: self.mode.contract(),
            targeted: trigger,
            probability,
            used_before: self.used[binding.era][binding.seat],
            cotton_visits: cotton_total,
            total_visits: total,
            covered_locations: locations.len(),
            choice: choices[selected].id.clone(),
        });
        Ok(selected)
    }

    /// Only call after ordinary_after succeeded. Repeated acknowledgement is
    /// harmless; mismatched choice/revision cannot consume or evade the cap.
    pub fn commit(
        &mut self,
        view: &Observation,
        choice: &Choice,
        after_revision: u32,
    ) -> Result<()> {
        if !self.allowed_phase(view.context.phase) {
            return Ok(());
        }
        let binding = Binding::of(&view.context);
        if self.committed == Some(binding) {
            return Ok(());
        }
        check(
            after_revision > binding.revision && self.current == Some(binding),
            "stale cotton commit",
        )?;
        let mut trigger = self.continuation_choice.is_some();
        if let Some(selected) = &self.continuation_choice {
            check(*selected == choice.id, "cotton continuation choice differs")?;
        }
        if let Some(selection) = &self.selection {
            check(
                selection.choice == choice.id,
                "cotton committed choice differs",
            )?;
            trigger = selection.targeted;
        }
        if trigger {
            check(
                targeted(choice, self.mode.develop()) && self.used[binding.era][binding.seat] < CAP,
                "cotton allowance exceeded",
            )?;
            self.used[binding.era][binding.seat] += 1;
            if cotton(choice) {
                self.counts[self.profile].targeted_builds += 1;
            } else {
                self.counts[self.profile].targeted_develops += 1;
                self.counts[self.profile].continuation_targets +=
                    u64::from(self.continuation_choice.is_some());
            }
        }
        if cotton(choice) {
            self.built[binding.era][binding.seat] += 1;
            self.counts[self.profile].actual_builds += 1;
        }
        if self.mode.develop() && develop_cotton(choice) {
            self.developed[binding.era][binding.seat] += 1;
            self.counts[self.profile].actual_develops += 1;
        }
        self.committed = Some(binding);
        Ok(())
    }
    pub fn receipt(&self) -> Value {
        let mut receipt = json!({"contract":self.mode.contract(),"long":self.counts[0],"short":self.counts[1],
            "targeted_by_era_seat":self.used,"builds_by_era_seat":self.built});
        if self.mode.develop() {
            receipt["develops_by_era_seat"] = json!(self.developed);
        }
        receipt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brass_ai_core::{
        state::{dto::StateDto, NativeState},
        types::Seat,
    };
    fn view() -> Observation {
        let dto: StateDto = serde_json::from_str(include_str!(
            "../../native/brass-ai-core/tests/data/initial-state.json"
        ))
        .unwrap();
        let state = NativeState::import(&dto).unwrap();
        state
            .observation(Seat::try_from(state.context().actor as u8).unwrap())
            .unwrap()
    }
    fn choices() -> Vec<Choice> {
        (0..3)
            .map(|i| Choice {
                id: i.to_string(),
                kind: Kind::Build,
                card: None,
                target: Some(i.to_string()),
                industry: Some(if i == 0 {
                    Industry::CoalMine
                } else {
                    Industry::CottonMill
                }),
                merchant: None,
                cards: None,
            })
            .collect()
    }
    fn trigger() -> u64 {
        (0..100)
            .find(|d| {
                (u128::from(super::super::cpu_eval::mix(
                    7 ^ super::super::cpu_eval::mix(*d ^ 0xc077_0a01),
                )) * 5)
                    < (1u128 << 64)
            })
            .unwrap()
    }
    #[test]
    fn mixture_records_exact_probability_and_approximately_twenty_percent_triggers() {
        let v = view();
        let c = choices();
        let mut s = State::default();
        let mut picked = [0; 3];
        s.prepare(&v, &c, false, true);
        for d in 0..10_000 {
            let i = s.sample(0, &c, &[90, 3, 7], 100, None, 7, d).unwrap();
            picked[i] += 1;
            let e = s.selection.as_ref().unwrap();
            assert!((e.probability - ([0.72, 0.024 + 0.06, 0.056 + 0.14][i])).abs() < 1e-14);
            assert_eq!(e.targeted, i != 0);
        }
        assert!((1800..2200).contains(&(picked[1] + picked[2])));
        assert!((500..700).contains(&picked[1]));
        assert_eq!(s.used, [[0; 4]; 2]); // sampling alone never spends an allowance
    }
    #[test]
    fn tempered_opening_cotton_mixture_records_the_tempered_probability() {
        let v = view();
        let c = choices();
        let mut s = State { mode: Mode::Seated, ..State::default() };
        s.prepare(&v, &c, false, true);
        let weights = [9., 3., 4.];
        let decision = (0..100).find(|d| s.trigger(7, *d)).unwrap();
        let i = s.sample(0, &c, &[81, 9, 16], 106, Some((&weights, 16.)), 7, decision).unwrap();
        assert!(i == 1 || i == 2);
        let e = s.selection.as_ref().unwrap();
        assert_eq!((e.cotton_visits, e.total_visits), (25, 106));
        assert!((e.probability - (0.7 * weights[i] / 16. + 0.3 * weights[i] / 7.)).abs() < 1e-14);
    }
    #[test]
    fn commit_is_bound_idempotent_and_caps_each_seat_and_era_independently() {
        let mut v = view();
        let c = choices();
        let mut s = State::default();
        let seat = v.context.actor as usize;
        for n in 0..3 {
            v.context.revision += 1;
            assert_eq!(s.prepare(&v, &c, true, true), 1);
            assert_eq!(s.prepare(&v, &c, true, true), 1);
            let i = s.sample(0, &c, &[90, 3, 7], 100, None, 7, trigger()).unwrap();
            assert!(s.selection.as_ref().unwrap().targeted);
            assert!(s.commit(&v, &c[0], v.context.revision + 1).is_err());
            assert!(s.commit(&v, &c[i], v.context.revision).is_err());
            s.commit(&v, &c[i], v.context.revision + 1).unwrap();
            s.commit(&v, &c[i], v.context.revision + 1).unwrap();
            assert_eq!(s.used[0][seat], n + 1);
        }
        v.context.revision += 1;
        assert_eq!(s.prepare(&v, &c, true, true), 0);
        assert_eq!(s.counts[1].eligible_decisions, 3);
        // Natural choices still build cotton at the cap.
        s.commit(&v, &c[1], v.context.revision + 1).unwrap();
        assert_eq!(s.used[0][seat], 3);
        assert_eq!(s.built[0][seat], 4);
        v.context.revision += 1;
        v.context.era = Era::RailEra;
        assert_eq!(s.prepare(&v, &c, false, true), 2);
        assert_eq!(s.used[1][seat], 0);
        v.context.revision += 1;
        v.context.actor = (v.context.actor + 1) % 4;
        v.seat = Seat::try_from(v.context.actor as u8).unwrap();
        v.context.era = Era::CanalEra;
        assert_eq!(s.prepare(&v, &c, false, true), 2);
    }
    #[test]
    fn windows_unflipped_and_missing_evidence_do_not_spend_the_quota() {
        let mut v = view();
        let c = choices();
        let mut s = State::default();
        assert_eq!(s.prepare(&v, &c, false, false), 0);
        v.context.revision += 1;
        let mut tile = v.players[v.context.actor as usize]
            .mat
            .iter()
            .find(|m| m.industry == Industry::CottonMill)
            .unwrap()
            .tiles[0]
            .clone();
        tile.flipped = false;
        tile.seat = v.seat;
        v.slots[0].tile = Some(tile);
        assert_eq!(s.prepare(&v, &c, false, true), 0);
        v.slots[0].tile.as_mut().unwrap().flipped = true;
        v.context.revision += 1;
        assert_eq!(s.prepare(&v, &c, false, true), 2);
        assert_eq!(s.sample(0, &c, &[100, 0, 0], 100, None, 7, trigger()).unwrap(), 0);
        assert!(s.selection.is_none());
        assert_eq!(s.counts[0].no_cotton_visits, 1);
        // A cancelled/unpublished decision is retried from its frozen table seed;
        // neither this state nor a fresh table has consumed an allowance.
        assert_eq!(s.used, State::default().used);
        s.no_evidence();
        assert_eq!(s.counts[0].no_evidence, 1);
    }

    #[test]
    fn develop_mixtures_can_develop_twice_in_one_action() {
        for (mode, contract, triggers_expected, probability) in [
            (Mode::Develop, DEVELOP_CONTRACT, 2300..2700, 0.325),
            (Mode::Mixed, MIXED_CONTRACT, 2800..3200, 0.37),
            (Mode::Mixed25, MIXED_25_CONTRACT, 2300..2700, 0.325),
        ] {
            let mut v = view();
            let mut choices = choices();
            choices[1].kind = Kind::Develop;
            choices[1].target = None;
            choices.truncate(2);
            let mut s = State { mode, ..State::default() };
            assert_eq!(s.prepare(&v, &choices, false, true), 2);
            let triggers = (0..10_000).filter(|d| s.trigger(7, *d)).count();
            assert!(triggers_expected.contains(&triggers));
            let decision = (0..100).find(|d| s.trigger(7, *d)).unwrap();
            assert_eq!(s.sample(0, &choices, &[90, 10], 100, None, 7, decision).unwrap(), 1);
            let label = s.selection.as_ref().unwrap().label();
            assert_eq!(label["contract"], contract);
            assert!((label["probability"].as_f64().unwrap() - probability).abs() < 1e-14);
            s.commit(&v, &choices[1], v.context.revision + 1).unwrap();

            v.context.phase = Phase::Continuation;
            v.context.revision += 1;
            choices[0].kind = Kind::Finish;
            assert_eq!(s.prepare(&v, &choices, false, true), 0); // no new search coverage
            let picked = s.sample_continuation(choices[0].id.clone(), &choices, 7, decision);
            assert_eq!(picked, choices[1].id);
            s.commit(&v, &choices[1], v.context.revision + 1).unwrap();
            let seat = v.context.actor as usize;
            assert_eq!(s.used[0][seat], 2);
            assert_eq!(s.counts[0].targeted_develops, 2);
            assert_eq!(s.counts[0].continuation_targets, 1);
            assert_eq!(s.receipt()["develops_by_era_seat"][0][seat], 2);
        }
    }
}
