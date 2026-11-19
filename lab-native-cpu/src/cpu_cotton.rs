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
const SINGLE_SEAT_CONTRACT: &str = "teacher-cotton-exploration/6.0.0";
const NETWORK_CONTRACT: &str = "teacher-cotton-exploration/7.0.0";
/// g2654: the network recipe, with Rail-era cotton builds targeted only from level three.
const RAIL_NETWORK_CONTRACT: &str = "teacher-cotton-exploration/8.0.0";
/// g3471: the g2654 recipe, timed within each era (`timing`).
const TIMED_NETWORK_CONTRACT: &str = "teacher-cotton-exploration/9.0.0";
pub const CAP: u8 = 3;
/// g3471: a seat's main actions in the era (Canal 1..=15, Rail 1..=16) with no cotton exploration,
/// and those where only cotton builds are targeted (no cotton develop or cotton-city network).
pub const QUIET_ACTIONS: ([u32; 2], [u32; 3]) = ([1, 15], [1, 2, 16]);
pub const BUILD_ONLY_ACTIONS: ([u32; 2], [u32; 2]) = ([14, 15], [15, 16]);

/// The acting seat's 1-based main action in its era. Its payment and continuation decisions keep that
/// index: the engine counts actions down only when an action finishes.
pub fn era_action(c: &ViewContext) -> u32 {
    (2 * u32::from(c.round)).saturating_sub(u32::from(c.actions_remaining)) + u32::from(c.era == Era::RailEra)
}

/// g3471 (step5.policy.cotton_timing twin): (any cotton target, cotton develop and network targets).
pub fn timing(c: &ViewContext) -> (bool, bool) {
    let index = era_action(c);
    let (quiet, build_only): (&[u32], &[u32]) = if c.era == Era::RailEra {
        (&QUIET_ACTIONS.1, &BUILD_ONLY_ACTIONS.1)
    } else {
        (&QUIET_ACTIONS.0, &BUILD_ONLY_ACTIONS.0)
    };
    if quiet.contains(&index) { (false, false) } else { (true, !build_only.contains(&index)) }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Builds,
    Develop,
    Mixed,
    Mixed25,
    Seated,
    SingleSeat,
    Network,
    RailNetwork,
    TimedNetwork,
}

impl Mode {
    pub fn develop(self) -> bool {
        self != Self::Builds
    }

    /// The one-network-per-era cotton recipes (g2400, and g2654 with the Rail-era build level).
    pub fn network(self) -> bool {
        matches!(self, Self::Network | Self::RailNetwork | Self::TimedNetwork)
    }

    /// The single-seat recipes, which keep the search's planned action as the base.
    fn single(self) -> bool {
        matches!(self, Self::SingleSeat | Self::Network | Self::RailNetwork | Self::TimedNetwork)
    }

    /// (any cotton target, develop and network targets) at a decision; only the timed recipe gates.
    fn gate(self, c: &ViewContext) -> (bool, bool) {
        if self == Self::TimedNetwork { timing(c) } else { (true, true) }
    }

    /// Lowest targeted cotton build level; level one is never targeted, Rail-era level two only before g2654.
    fn minimum_build_level(self, era: usize) -> u8 {
        if matches!(self, Self::RailNetwork | Self::TimedNetwork) && era == 1 { 3 } else { 2 }
    }

    fn contract(self) -> &'static str {
        match self {
            Self::Builds => CONTRACT,
            Self::Develop => DEVELOP_CONTRACT,
            Self::Mixed => MIXED_CONTRACT,
            Self::Mixed25 => MIXED_25_CONTRACT,
            Self::Seated => SEATED_CONTRACT,
            Self::SingleSeat => SINGLE_SEAT_CONTRACT,
            Self::Network => NETWORK_CONTRACT,
            Self::RailNetwork => RAIL_NETWORK_CONTRACT,
            Self::TimedNetwork => TIMED_NETWORK_CONTRACT,
        }
    }

    fn rate(self) -> (u128, u128) {
        match self {
            Self::Builds => (1, 5),
            Self::Develop | Self::Mixed25 => (1, 4),
            Self::Mixed | Self::Seated | Self::SingleSeat | Self::Network | Self::RailNetwork | Self::TimedNetwork => (3, 10),
        }
    }

    fn probability(self) -> f64 {
        match self {
            Self::Builds => 0.2,
            Self::Develop | Self::Mixed25 => 0.25,
            Self::Mixed | Self::Seated | Self::SingleSeat | Self::Network | Self::RailNetwork | Self::TimedNetwork => 0.3,
        }
    }

    pub fn settings(self) -> Value {
        let scope = if matches!(self, Self::Mixed | Self::Mixed25 | Self::Seated) || self.single() { "train-and-heldout" } else { "train" };
        let mut settings = json!({"contract":self.contract(),"probability":self.probability(),"max_per_seat_era":CAP,
            "long_candidates":2,"short_candidates":1,"pause_unflipped":true,"scope":scope});
        if self.single() {
            settings["minimum_build_level"] = json!(2);
            settings["short_window"] = json!("all-actions");
        }
        if self.network() { settings["network_max_per_seat_era"] = json!(1); }
        if matches!(self, Self::RailNetwork | Self::TimedNetwork) { settings["rail_minimum_build_level"] = json!(3); }
        if self == Self::TimedNetwork {
            settings["quiet_actions"] = json!({"CanalEra": QUIET_ACTIONS.0, "RailEra": QUIET_ACTIONS.1});
            settings["build_only_actions"] = json!({"CanalEra": BUILD_ONLY_ACTIONS.0, "RailEra": BUILD_ONLY_ACTIONS.1});
        }
        settings
    }
}

pub fn cotton(c: &Choice) -> bool {
    c.kind == Kind::Build && c.industry == Some(Industry::CottonMill)
}
fn develop_cotton(c: &Choice) -> bool {
    c.kind == Kind::Develop && c.industry == Some(Industry::CottonMill)
}
fn targeted(c: &Choice, develop: bool, build: bool, networks: &[String]) -> bool {
    build && cotton(c) || develop && develop_cotton(c) ||
        c.kind == Kind::Network && c.target.as_ref().is_some_and(|t| networks.contains(t))
}

fn network_targets(view: &Observation, choices: &[Choice]) -> Vec<String> {
    if !choices.iter().any(|c| c.kind == Kind::Network) { return Vec::new(); }
    let cities: std::collections::HashSet<&str> = view.slots.iter()
        .filter(|s| s.tile.is_none() && s.types.contains(&Industry::CottonMill))
        .map(|s| s.id.split(':').next().unwrap()).collect();
    view.links.iter().filter(|l| l.regions.iter().any(|r| cities.contains(r.as_str())))
        .map(|l| l.id.clone()).collect()
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
    #[serde(skip_serializing_if = "is_zero")]
    pub targeted_networks: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub actual_networks: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub continuation_network_targets: u64,
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
    planned: Option<String>,
    network_used_before: Option<u8>,
}
impl Selection {
    pub fn label(&self) -> Value {
        let mut label = json!({"contract":self.contract,"targeted":self.targeted,"probability":self.probability,
            "used_before":self.used_before,"cotton_visits":self.cotton_visits,"total_visits":self.total_visits,
            "covered_locations":self.covered_locations});
        if let Some(planned) = &self.planned {
            label["planned_action"] = json!(planned);
        }
        if let Some(used) = self.network_used_before { label["network_used_before"] = json!(used); }
        label
    }
}
#[derive(Default, Debug)]
pub struct State {
    pub counts: [Counts; 2],
    pub used: [[u8; 4]; 2],
    pub built: [[u32; 4]; 2],
    developed: [[u32; 4]; 2],
    network_used: [[u8; 4]; 2],
    network_built: [[u32; 4]; 2],
    pub mode: Mode,
    pub seats: Option<u8>,
    current: Option<Binding>,
    committed: Option<Binding>,
    eligible: bool,
    builds: bool,
    develops: bool,
    networks: Vec<String>,
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
        self.networks.clear();
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
        let industry = !self.mode.network() || self.used[binding.era][binding.seat] < CAP;
        // g3471: a quiet action of the era explores no cotton; a build-only action targets cotton builds alone.
        let (timed, wide) = self.mode.gate(&view.context);
        let window = window && timed;
        self.develops = self.mode.develop() && industry && wide;
        let minimum = self.mode.minimum_build_level(binding.era);
        self.builds = industry && (!self.mode.single() || view.players[binding.seat].mat.iter()
            .find(|m| m.industry == Industry::CottonMill)
            .and_then(|m| m.tiles.first()).is_some_and(|tile| tile.level >= minimum));
        if self.mode.network() && wide && self.network_used[binding.era][binding.seat] == 0 {
            self.networks = network_targets(view, choices);
        }
        if phase == Phase::Continuation {
            self.eligible = window
                && (self.used[binding.era][binding.seat] < CAP || !self.networks.is_empty())
                && choices.len() > 1
                && choices.iter().any(|c| targeted(c, self.develops, false, &self.networks))
                && !pause();
            return 0;
        }
        let develop = self.develops;
        let c = &mut self.counts[self.profile];
        c.action_decisions += 1;
        if choices.len() < 2 || !choices.iter().any(|c| targeted(c, develop, self.builds, &self.networks)) {
            return 0;
        }
        c.legal_decisions += 1;
        if !window {
            c.blocked_window += 1;
            return 0;
        }
        if self.used[binding.era][binding.seat] >= CAP && self.networks.is_empty() {
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

    pub fn eligible(&self) -> bool {
        self.eligible
    }

    pub fn builds(&self) -> bool {
        self.builds
    }

    pub fn coverage(&self) -> (bool, bool, Vec<String>) {
        (self.develops, self.builds, self.networks.clone())
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
            let candidates: Vec<_> = choices.iter().filter(|c| targeted(c, self.develops, false, &self.networks)).collect();
            if !candidates.is_empty() {
                let index = if self.mode.network() {
                    ((u128::from(super::cpu_eval::mix(seed ^ super::cpu_eval::mix(decision ^ 0xc077_0a02))) * candidates.len() as u128) >> 64) as usize
                } else { 0 };
                let choice = candidates[index];
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
        let develop = self.develops;
        let builds = self.builds;
        let networks = &self.networks;
        let planned = self.mode.single();
        let mut cotton_total = 0;
        let mut cotton_mass = 0.;
        let mut locations = Vec::with_capacity(2);
        for (i, (choice, n)) in choices.iter().zip(counts).enumerate() {
            if *n > 0 && targeted(choice, develop, builds, networks) {
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
        let base_p = if planned { f64::from(u8::from(targeted(&choices[base], develop, builds, networks))) }
                     else { tempered.map_or(cotton_total as f64 / total as f64, |(_, mass)| cotton_mass / mass) };
        c.base_cotton_probability_sum += base_p;
        c.mixed_cotton_probability_sum += (1. - mixture) * base_p + mixture;
        let mut selected = base;
        if trigger {
            let z = super::cpu_eval::mix(seed ^ super::cpu_eval::mix(decision ^ 0xc077_0a02));
            if let Some((weights, _)) = tempered {
                let eligible = weights.iter().copied().enumerate().filter(|(i, _)| targeted(&choices[*i], develop, builds, networks));
                selected = super::cpu_eval::sample_weights(eligible, cotton_mass, z)?;
            } else {
                let draw = ((u128::from(z) * u128::from(cotton_total)) >> 64) as u64;
                let mut sum = 0;
                for (i, (choice, n)) in choices.iter().zip(counts).enumerate() {
                    if targeted(choice, develop, builds, networks) {
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
        let base_mass = if planned { f64::from(u8::from(selected == base)) } else { selected_mass / full_mass };
        let probability = (1. - mixture) * base_mass
            + if targeted(&choices[selected], develop, builds, networks) { mixture * selected_mass / cotton_mass } else { 0. };
        self.selection = Some(Selection {
            contract: self.mode.contract(),
            targeted: trigger,
            probability,
            used_before: self.used[binding.era][binding.seat],
            cotton_visits: cotton_total,
            total_visits: total,
            covered_locations: locations.len(),
            choice: choices[selected].id.clone(),
            planned: planned.then(|| choices[base].id.clone()),
            network_used_before: self.mode.network().then_some(self.network_used[binding.era][binding.seat]),
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
                targeted(choice, self.develops, self.builds, &self.networks) &&
                    (if choice.kind == Kind::Network { self.network_used[binding.era][binding.seat] < 1 }
                     else { self.used[binding.era][binding.seat] < CAP }),
                "cotton allowance exceeded",
            )?;
            if choice.kind == Kind::Network {
                self.network_used[binding.era][binding.seat] += 1;
                self.counts[self.profile].targeted_networks += 1;
                self.counts[self.profile].continuation_network_targets += u64::from(self.continuation_choice.is_some());
            } else if cotton(choice) {
                self.used[binding.era][binding.seat] += 1;
                self.counts[self.profile].targeted_builds += 1;
            } else {
                self.used[binding.era][binding.seat] += 1;
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
        if self.mode.network() && choice.kind == Kind::Network {
            self.network_built[binding.era][binding.seat] += 1;
            self.counts[self.profile].actual_networks += 1;
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
        if self.mode.network() {
            receipt["network_targeted_by_era_seat"] = json!(self.network_used);
            receipt["networks_by_era_seat"] = json!(self.network_built);
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

    #[test]
    fn single_seat_keeps_the_plan_and_uses_only_develop_visits_at_level_one() {
        let mut v = view();
        let mut choices = choices();
        let seat = v.seat as usize;
        choices[2].kind = Kind::Develop;
        let mut state = State { mode: Mode::SingleSeat, seats: Some(1 << seat), ..State::default() };
        assert_eq!(state.prepare(&v, &choices, true, true), 1);
        assert!(!state.builds());
        let hold = (0..100).find(|&d| !state.trigger(7, d)).unwrap();
        let trigger = (0..100).find(|&d| state.trigger(7, d)).unwrap();
        // Level-one build retains its planned probability and gets no cotton mass.
        assert_eq!(state.sample(1, &choices, &[0, 9, 1], 10, None, 7, hold).unwrap(), 1);
        let selection = state.selection.as_ref().unwrap();
        assert_eq!(selection.cotton_visits, 1);
        assert_eq!(selection.probability, 0.7);
        assert_eq!(state.sample(1, &choices, &[0, 9, 1], 10, None, 7, trigger).unwrap(), 2);
        assert_eq!(state.selection.as_ref().unwrap().probability, 0.3);
        state.commit(&v, &choices[2], v.context.revision + 1).unwrap();
        assert_eq!(state.used[0][seat], 1);
        let mat = v.players[seat].mat.iter_mut().find(|m| m.industry == Industry::CottonMill).unwrap();
        while mat.tiles.first().is_some_and(|t| t.level == 1) { mat.tiles.remove(0); }
        v.context.revision += 1;
        assert_eq!(state.prepare(&v, &choices, true, true), 1);
        assert!(state.builds());
    }

    #[test]
    fn rail_network_targets_rail_builds_only_from_level_three() {
        let mut v = view();
        let seat = v.seat as usize;
        let mut choices = choices();
        choices[2].kind = Kind::Develop;
        let mat = v.players[seat].mat.iter_mut().find(|m| m.industry == Industry::CottonMill).unwrap();
        while mat.tiles.first().is_some_and(|t| t.level < 2) { mat.tiles.remove(0); }
        assert_eq!(mat.tiles.first().unwrap().level, 2);
        for (mode, era, builds) in [(Mode::Network, Era::RailEra, true), (Mode::RailNetwork, Era::CanalEra, true),
                                    (Mode::RailNetwork, Era::RailEra, false)] {
            let mut state = State { mode, seats: Some(1 << seat), ..State::default() };
            v.context.era = era;
            v.context.revision += 1;
            assert_eq!(state.prepare(&v, &choices, true, true), 1);
            assert_eq!(state.builds(), builds, "{mode:?} {era:?}");
        }
        let mat = v.players[seat].mat.iter_mut().find(|m| m.industry == Industry::CottonMill).unwrap();
        while mat.tiles.first().is_some_and(|t| t.level < 3) { mat.tiles.remove(0); }
        let mut state = State { mode: Mode::RailNetwork, seats: Some(1 << seat), ..State::default() };
        v.context.revision += 1;
        assert_eq!(state.prepare(&v, &choices, true, true), 1);
        assert!(state.builds());
        let settings = Mode::RailNetwork.settings();
        assert_eq!(settings["contract"], RAIL_NETWORK_CONTRACT);
        assert_eq!((settings["minimum_build_level"].as_u64(), settings["rail_minimum_build_level"].as_u64()), (Some(2), Some(3)));
        let mut network = Mode::Network.settings();
        network["contract"] = json!(RAIL_NETWORK_CONTRACT);
        network["rail_minimum_build_level"] = json!(3);
        assert_eq!(network, settings);
    }

    #[test]
    fn network_has_a_separate_era_quota_and_continuations_recheck_it() {
        let mut v = view();
        let seat = v.seat as usize;
        let mut choices = choices();
        choices[1].kind = Kind::Network;
        choices[1].industry = None;
        let target = network_targets(&v, &choices).first().unwrap().clone();
        choices[1].target = Some(target.clone());
        choices[2].kind = Kind::Develop;
        let mut state = State { mode: Mode::Network, seats: Some(1 << seat), ..State::default() };
        state.used[0][seat] = CAP;
        assert_eq!(state.prepare(&v, &choices, true, true), 1);
        assert!(!state.builds && !state.develops);
        assert_eq!(state.prepare(&v, &choices, true, true), 1);
        let trigger = (0..100).find(|&d| state.trigger(7,d)).unwrap();
        assert_eq!(state.sample(0,&choices,&[9,1,0],10,None,7,trigger).unwrap(),1);
        let label = state.selection.as_ref().unwrap().label();
        assert_eq!(label["used_before"],3);
        assert_eq!(label["network_used_before"],0);
        assert_eq!(label["probability"],0.3);
        state.commit(&v,&choices[1],v.context.revision+1).unwrap();
        assert_eq!(state.used[0][seat],3);
        assert_eq!(state.network_used[0][seat],1);
        v.context.revision += 1;
        v.context.phase = Phase::Continuation;
        assert_eq!(state.prepare(&v,&choices,true,true),0);
        assert!(!state.eligible());
        v.context.revision += 1;
        v.context.era = Era::RailEra;
        state.prepare(&v,&choices,true,true);
        let picked = state.sample_continuation(choices[0].id.clone(), &choices[..2],7,trigger);
        assert_eq!(picked,choices[1].id);
        state.commit(&v,&choices[1],v.context.revision+1).unwrap();
        assert_eq!(state.network_used[1][seat],1);
        assert_eq!(state.used[1][seat],0);
        assert_eq!(state.counts[1].continuation_network_targets,1);
        // Occupying all cotton slots removes network targets, including mixed slots.
        v.slots.retain(|s| !s.types.contains(&Industry::CottonMill));
        assert!(network_targets(&v,&choices).is_empty());
    }

    #[test]
    fn the_timed_recipe_is_quiet_at_era_edges_and_builds_only_at_their_ends() {
        let mut v = view();
        let seat = v.seat as usize;
        let mut choices = choices();
        choices[1].kind = Kind::Network;
        choices[1].industry = None;
        let target = network_targets(&v, &choices).first().unwrap().clone();
        choices[1].target = Some(target);
        choices[2].kind = Kind::Develop;
        let mut build = choices[2].clone();
        build.id = "3".into();
        build.kind = Kind::Build;
        choices.push(build); // a cotton build: the one target of a build-only action
        let mat = v.players[seat].mat.iter_mut().find(|m| m.industry == Industry::CottonMill).unwrap();
        while mat.tiles.first().is_some_and(|t| t.level < 3) { mat.tiles.remove(0); }
        // (era, round, actions remaining) -> (the seat's action in the era, explores, develop and network too)
        for (era, round, left, index, explores, wide) in [
            (Era::CanalEra, 1, 1, 1, false, false), (Era::CanalEra, 2, 2, 2, true, true), (Era::CanalEra, 7, 1, 13, true, true),
            (Era::CanalEra, 8, 2, 14, true, false), (Era::CanalEra, 8, 1, 15, false, false),
            (Era::RailEra, 1, 2, 1, false, false), (Era::RailEra, 1, 1, 2, false, false), (Era::RailEra, 2, 2, 3, true, true),
            (Era::RailEra, 7, 1, 14, true, true), (Era::RailEra, 8, 2, 15, true, false), (Era::RailEra, 8, 1, 16, false, false),
        ] {
            v.context.era = era;
            v.context.round = round;
            v.context.actions_remaining = left;
            v.context.phase = Phase::Action;
            v.context.revision += 1;
            assert_eq!(era_action(&v.context), index);
            assert_eq!(timing(&v.context), (explores, wide), "{era:?} {round} {left}");
            let mut state = State { mode: Mode::TimedNetwork, seats: Some(1 << seat), ..State::default() };
            assert_eq!(state.prepare(&v, &choices, true, true), usize::from(explores), "{era:?} {round} {left}");
            assert_eq!(state.eligible(), explores);
            let (develops, builds, networks) = state.coverage();
            assert_eq!((develops, !networks.is_empty()), (wide, wide), "{era:?} {round} {left}");
            assert!(builds);
            // A continuation (a develop's second tile) keeps its action's index and rule.
            v.context.phase = Phase::Continuation;
            v.context.revision += 1;
            state.prepare(&v, &choices, true, true);
            assert_eq!(state.eligible(), explores && wide, "{era:?} {round} {left}");
            // The untimed recipe explores at every action.
            let mut plain = State { mode: Mode::RailNetwork, seats: Some(1 << seat), ..State::default() };
            v.context.phase = Phase::Action;
            v.context.revision += 1;
            assert_eq!(plain.prepare(&v, &choices, true, true), 1);
        }
        let settings = Mode::TimedNetwork.settings();
        let mut rail = Mode::RailNetwork.settings();
        rail["contract"] = json!(TIMED_NETWORK_CONTRACT);
        rail["quiet_actions"] = json!({"CanalEra": [1, 15], "RailEra": [1, 2, 16]});
        rail["build_only_actions"] = json!({"CanalEra": [14, 15], "RailEra": [15, 16]});
        assert_eq!(settings, rail);
    }
}
