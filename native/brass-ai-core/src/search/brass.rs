//! Native staged Brass adapter. Policy inputs never contain a determinized world.
use super::{
    reward::{Ledger, LedgerState, Reward},
    *,
};
use crate::{
    features::FeatureContext,
    state::{
        belief::{PreparedRoot, SamplingDraws},
        dto::Choice,
        observation::{InformationState, Observation, ObservationMemory, Recall},
        NativeState,
    },
    types::Seat,
};
use std::hash::Hasher;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Key {
    pub information: InformationState,
    pub choices: Vec<Choice>,
    pub context: FeatureContext,
    pub settled: Vec<(crate::types::Era, u8)>,
}
impl Eq for Key {}
fn bits(x: f64) -> u64 {
    if x == 0. {
        0
    } else {
        x.to_bits()
    }
}
impl Hash for Key {
    fn hash<H: Hasher>(&self, h: &mut H) {
        // A cheap subset of the fields Eq compares, so equal keys hash equally.
        // Typed equality still decides every candidate; nothing here is a digest.
        let o = &self.information.observation;
        o.chance.hash(h);
        o.setup.hash(h);
        if let Some(v) = &o.view {
            v.seat.hash(h);
            let c = &v.context;
            (c.revision, c.actor, c.phase, c.actions_remaining, c.round).hash(h);
            v.own_hand.hash(h);
            v.history.len().hash(h);
            v.history.last().hash(h);
            for p in &v.players {
                (p.cash, p.vp, p.spent, p.income_marker, p.debt, p.hand_count).hash(h);
            }
            v.pending.hash(h);
            (v.coal_market, v.iron_market, v.deck_count).hash(h);
        }
        let r = &self.information.recall;
        r.len().hash(h);
        r.last().map(|e| e.revision).hash(h);
        self.choices.len().hash(h);
        self.choices.first().map(|c| &c.id).hash(h);
        self.settled.hash(h);
        for x in &self.context.remaining {
            bits(*x).hash(h);
        }
    }
}
/// Multiply-rotate hasher for the owner-local decision cache, whose keys are
/// exact identity bytes compared in full on every hit.
#[derive(Default)]
struct Fx(u64);
const FX: u64 = 0x517c_c1b7_2722_0a95;
impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let word = u64::from_le_bytes(c.try_into().expect("eight bytes"));
            self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(FX);
        }
        for b in chunks.remainder() {
            self.0 = (self.0.rotate_left(5) ^ u64::from(*b)).wrapping_mul(FX);
        }
    }
    fn write_usize(&mut self, x: usize) {
        self.0 = (self.0.rotate_left(5) ^ x as u64).wrapping_mul(FX);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}
/// Decisions this root has already built, keyed by an exact identity that
/// refines the permitted view, recall, legal set and ledger. A hit returns
/// the same shared key, so tree and cache equality short-circuit on the
/// pointer, and no view, legal set or key hash is rebuilt.
#[derive(Default)]
struct Known {
    infos: std::collections::HashMap<Box<[u8]>, Info<Key, String>, std::hash::BuildHasherDefault<Fx>>,
    scratch: Vec<u8>,
    bytes: usize,
    // Test controls: no reuse, or rebuild and compare every reused decision.
    disabled: bool,
    checked: bool,
    checked_hits: u64,
}
// Identity bytes retained per root; beyond this, decisions are rebuilt.
const KNOWN_BYTES: usize = 64 << 20;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub view: Observation,
    pub recall: Recall,
    pub context: FeatureContext,
    pub reward: Reward,
    pub ledger: LedgerState,
    pub shuffle_tape: Option<Vec<Vec<u32>>>,
    pub sample_seed: u64,
}
pub struct Brass {
    prepared: PreparedRoot,
    root_info: Info<Key, String>,
    root_ledger: Ledger,
    context: FeatureContext,
    reward: Reward,
    shuffle_tape: Option<Vec<Vec<u32>>>,
    sample_seed: u64,
    samples: usize,
    pub sampled_indices: Vec<Vec<u32>>,
    record: bool,
    known: std::cell::RefCell<Known>,
    root_seat: Seat,
    // The root seat's recall prefix is shared by every world of this root.
    root_recall: usize,
}
pub struct World {
    state: NativeState,
    memory: ObservationMemory,
    ledger: Ledger,
    // The immutable key already owns the permitted view and legal choices.
    // Keep that exact decision alive through step instead of cloning both.
    decision: Option<std::sync::Arc<Key>>,
}
impl Brass {
    pub(crate) fn reusable_sampler(&self) -> bool {
        self.shuffle_tape.is_none()
    }
    /// Deterministic permitted completion. Never consumes a search world's stream.
    pub fn completion_world(&self) -> Result<World> {
        let (state, memory, _) = self.prepared.sample(SamplingDraws::Recorded {
            indices: vec![0; self.prepared.draw_count()],
        })?;
        Ok(World {
            state,
            memory,
            ledger: self.root_ledger.clone(),
            decision: None,
        })
    }
    /// Certified ordinary own-action transition. Every supported rule below
    /// reads only the public board and actor's hand; Finish with two actions
    /// remaining cannot draw, settle income or change actor. We also check the
    /// complete semantic path, ledger and hidden-zone allocation law. No sampled
    /// opponent cards/deck are copied into the successor Input.
    pub fn project(
        &self,
        action: &session::plans::Action,
        budget: session::clock::Budget,
        clock: &impl session::clock::Clock,
    ) -> Result<(Input, Info<Key, String>)> {
        use crate::{state::dto::Kind, types::Phase};
        let start = &self.root_info;
        let view = start.key.information.observation.view.as_ref().unwrap();
        check(
            view.context.phase == Phase::Action && view.context.actions_remaining == 2,
            "projection requires two ordinary actions",
        )?;
        check(
            !action.steps.is_empty() && action.steps.len() <= 96,
            "projection step bound",
        )?;
        let turn = session::TurnKey::of(start)?;
        let mut world = self.completion_world()?;
        for (i, step) in action.steps.iter().enumerate() {
            check(budget.work_open(clock), "projection cutoff")?;
            let Position::Decision(info) = self.inspect(&mut world)? else {
                return Err(invalid("projection chance/terminal"));
            };
            check(
                info == step.precondition && session::TurnKey::of(&info)? == turn,
                "projection precondition/actor changed",
            )?;
            let choice = info
                .key
                .choices
                .iter()
                .find(|c| c.id == step.choice)
                .ok_or_else(|| invalid("projection illegal step"))?;
            check(
                matches!(
                    choice.kind,
                    Kind::Build
                        | Kind::Network
                        | Kind::Sell
                        | Kind::Develop
                        | Kind::Loan
                        | Kind::Pass
                        | Kind::Finish
                        | Kind::Coal
                        | Kind::Iron
                        | Kind::Beer
                        | Kind::BonusDevelop
                        | Kind::BonusSkip
                ),
                "uncertified projection action",
            )?;
            check(
                (choice.kind == Kind::Finish) == (i + 1 == action.steps.len()),
                "projection incomplete action",
            )?;
            self.step(&mut world, &step.choice)?;
        }
        let Position::Decision(boundary) = self.inspect(&mut world)? else {
            return Err(invalid("projection information transition"));
        };
        let next = boundary.key.information.observation.view.as_ref().unwrap();
        check(
            session::TurnKey::of(&boundary)? == turn
                && next.context.phase == Phase::Action
                && next.context.actions_remaining == 1
                && next.context.revision > view.context.revision
                && next.deck_count == view.deck_count,
            "projection boundary changed",
        )?;
        let input = Input {
            view: next.clone(),
            recall: boundary.key.information.recall.clone(),
            context: boundary.key.context.clone(),
            reward: self.reward.clone(),
            ledger: world.ledger.state(),
            shuffle_tape: None,
            sample_seed: self.sample_seed ^ 0xa0761d6478bd642f,
        };
        let projected = Brass::new(input.clone(), false)?;
        check(
            projected.root_info == boundary,
            "projected observation/legal/ledger mismatch",
        )?;
        check(
            self.prepared.same_hidden_allocation(&projected.prepared),
            "conditional sampler mismatch",
        )?;
        check(budget.work_open(clock), "projection cutoff")?;
        Ok((input, boundary))
    }
    pub fn new(input: Input, record: bool) -> Result<Self> {
        input.context.validate()?;
        input.reward.validate()?;
        check(
            input.context.reward_id == input.reward.identity()?,
            "reward identity mismatch",
        )?;
        check(
            input.context.stable_descriptors.is_none(),
            "search uses unknown critic descriptors",
        )?;
        check(
            input.context.max_cost == input.reward.insolvency.max_cost,
            "model/reward budget mismatch",
        )?;
        check(
            input.view.context.actor == i8::try_from(u8::from(input.view.seat)).expect("seat"),
            "root must be the acting permitted view",
        )?;
        let mut ledger = Ledger::new(input.reward.clone(), input.ledger)?;
        ledger.observe(&input.view)?;
        check(
            input.context.remaining == ledger.remaining(),
            "root remaining budget is not the causal ledger",
        )?;
        let prepared = PreparedRoot::new(&input.view, input.recall.clone())?;
        // Legal semantics depend only on permitted root facts, verified per sample.
        let (state, memory, _) = prepared.sample(SamplingDraws::Recorded {
            indices: vec![0; prepared.draw_count()],
        })?;
        let root_info = Self::information(&state, &memory, &ledger, &input.context)?;
        check(
            root_info.key.information.observation.view.as_ref() == Some(&input.view),
            "root reconstruction changed permitted view",
        )?;
        let root_seat = input.view.seat;
        let root_recall = memory.recall_len(root_seat);
        let mut known = Known::default();
        // Every sampled root resolves to this exact shared root key.
        let mut id = Vec::new();
        Self::identify(&state, &memory, &ledger, root_seat, root_seat, root_recall, &mut id);
        known.bytes = id.len();
        known.infos.insert(id.into_boxed_slice(), root_info.clone());
        Ok(Self {
            prepared,
            root_info,
            root_ledger: ledger,
            context: input.context,
            reward: input.reward,
            shuffle_tape: input.shuffle_tape,
            sample_seed: input.sample_seed,
            samples: 0,
            sampled_indices: vec![],
            record,
            known: std::cell::RefCell::new(known),
            root_seat,
            root_recall,
        })
    }
    fn identify(
        state: &NativeState,
        memory: &ObservationMemory,
        ledger: &Ledger,
        seat: Seat,
        root_seat: Seat,
        root_recall: usize,
        out: &mut Vec<u8>,
    ) {
        out.clear();
        state.visible_identity(seat, out);
        let skip = if seat == root_seat { root_recall } else { 0 };
        memory.recall_identity(seat, skip, out);
        ledger.identity(out);
    }
    /// The decision at `w`: an identical decision this root already built, or a new one.
    fn decision(&self, w: &World) -> Result<Info<Key, String>> {
        let actor =
            usize::try_from(w.state.context().actor).map_err(|_| invalid("invalid acting seat"))?;
        let seat = Seat::try_from(actor as u8).map_err(|_| invalid("invalid acting seat"))?;
        let mut known = self.known.borrow_mut();
        let known = &mut *known;
        let mut id = std::mem::take(&mut known.scratch);
        Self::identify(&w.state, &w.memory, &w.ledger, seat, self.root_seat, self.root_recall, &mut id);
        let reused = if known.disabled {
            None
        } else {
            known.infos.get(id.as_slice()).cloned()
        };
        let info = match reused {
            Some(info) => {
                if known.checked {
                    let fresh = Self::information(&w.state, &w.memory, &w.ledger, &self.context)?;
                    check(fresh == info, "decision identity is not a refinement")?;
                    known.checked_hits += 1;
                }
                info
            }
            None => {
                let info = Self::information(&w.state, &w.memory, &w.ledger, &self.context)?;
                if !known.disabled && known.bytes + id.len() <= KNOWN_BYTES {
                    known.bytes += id.len();
                    known.infos.insert(id.as_slice().into(), info.clone());
                }
                info
            }
        };
        known.scratch = id;
        Ok(info)
    }
    fn information(
        state: &NativeState,
        memory: &ObservationMemory,
        ledger: &Ledger,
        context: &FeatureContext,
    ) -> Result<Info<Key, String>> {
        let actor =
            usize::try_from(state.context().actor).map_err(|_| invalid("invalid acting seat"))?;
        let seat = Seat::try_from(actor as u8).map_err(|_| invalid("invalid acting seat"))?;
        let information = memory.information_at(state, seat)?;
        check(
            information
                .observation
                .view
                .as_ref()
                .is_some_and(|v| v.seat == seat && v.context.actor == actor as i8)
                && information.observation.chance.is_none(),
            "unsupported actor view",
        )?;
        let choices = state.legal_choices()?;
        let actions = choices
            .iter()
            .map(|c| c.id.clone())
            .collect::<Vec<_>>()
            .into();
        let mut context = context.clone();
        context.remaining = ledger.remaining();
        Ok(Info {
            key: std::sync::Arc::new(Key {
                information,
                choices,
                context,
                settled: ledger.state().settled,
            }),
            actor,
            actions,
        })
    }
    pub fn finish(&self) -> Result<()> {
        check(
            self.shuffle_tape
                .as_ref()
                .is_none_or(|s| s.len() == self.samples),
            "unused sampled worlds",
        )
    }
}
impl Game for Brass {
    type Key = Key;
    type Action = String;
    type World = World;
    fn root(&mut self) -> Result<Info<Key, String>> {
        Ok(self.root_info.clone())
    }
    fn sample(&mut self, _: &mut Draws) -> Result<World> {
        let random = if let Some(tape) = &self.shuffle_tape {
            SamplingDraws::Recorded {
                indices: tape
                    .get(self.samples)
                    .ok_or_else(|| invalid("sample tape exhausted"))?
                    .clone(),
            }
        } else {
            SamplingDraws::Chacha12 {
                seed: self.sample_seed,
                stream: self.samples as u64,
            }
        };
        let (state, memory, indices) = self.prepared.sample(random)?;
        self.samples += 1;
        if self.record {
            self.sampled_indices.push(indices);
        }
        Ok(World {
            state,
            memory,
            ledger: self.root_ledger.clone(),
            decision: None,
        })
    }
    fn inspect(&self, w: &mut World) -> Result<Position<Key, String>> {
        if let Some(chance) = w.state.chance_outcomes() {
            let mut outcomes: Vec<_> = chance
                .outcomes
                .into_iter()
                .map(|o| (format!("chance:card:{}", o.card), o.probability))
                .collect();
            outcomes.sort_by(|a, b| a.0.cmp(&b.0));
            return Ok(Position::Chance(outcomes));
        }
        // Public debt accounting is observer independent and runs at reveal boundaries.
        w.ledger.observe_accounts(
            &w.memory
                .accounts_at(&w.state)
                .ok_or_else(|| invalid("missing published accounting view"))?,
        )?;
        if let Some(result) = w.state.result() {
            return Ok(Position::Terminal(self.reward.terminal(&result)?));
        }
        let info = self.decision(w)?;
        w.decision = Some(info.key.clone());
        Ok(Position::Decision(info))
    }
    /// The world's own state moves to the successor in place: no copy of the state
    /// and no old state to drop. A failed step leaves the world partly updated;
    /// every caller (traversal, projection, plan replay) then drops it.
    fn step(&self, w: &mut World, action: &String) -> Result<()> {
        let revision = w.state.context().revision;
        if let Some(card) = action.strip_prefix("chance:card:") {
            w.state.resolve_chance_in_place(revision, card)?;
        } else {
            let decision = w
                .decision
                .take()
                .ok_or_else(|| invalid("step without decision"))?;
            let view = decision
                .information
                .observation
                .view
                .as_ref()
                .ok_or_else(|| invalid("missing actor view"))?;
            let choice = decision
                .choices
                .iter()
                .find(|c| &c.id == action)
                .ok_or_else(|| invalid("selected action outside legal set"))?;
            w.state.apply_legal_in_place(revision, choice)?;
            w.ledger.chosen(view, choice);
        }
        w.memory.refresh_recall(&w.state)?;
        Ok(())
    }
    fn spent(&self, w: &World) -> [f64; 4] {
        w.ledger.spent
    }
}

#[cfg(test)]
mod decision_cache_tests {
    use super::*;
    use crate::search::{math::Statistics, reward::Insolvency};
    use crate::state::{
        dto::{Kind, StateDto},
        observation::{Accounts, RecallEntry},
    };
    use crate::types::Phase;

    fn reward() -> Reward {
        Reward {
            version: "brass-combined-return/2.0.0".into(),
            win: 1.,
            placement: 0.1,
            vp: 0.001,
            vp_cap: Some(180),
            insolvency: Insolvency {
                event_cost: 0.0025,
                max_cost: 0.025,
            },
        }
    }
    fn mix(text: &str, salt: u64) -> u64 {
        text.bytes().fold(0xcbf2_9ce4_8422_2325 ^ salt, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        })
    }
    struct Prior;
    impl LeafEvaluator<Key, String> for Prior {
        fn evaluate(&mut self, info: &Info<Key, String>) -> Result<Statistics> {
            let logits = info
                .actions
                .iter()
                .map(|a| (mix(a, 7) % 1000) as f64 / 500. - 1.)
                .collect();
            Statistics::new(logits, [0.25; 4])
        }
    }
    fn rank(kind: Kind) -> u8 {
        match kind {
            Kind::Build => 6,
            Kind::Network => 5,
            Kind::Sell => 4,
            Kind::Develop => 3,
            Kind::Loan => 2,
            Kind::Pass => 0,
            _ => 1,
        }
    }
    /// One greedy four-seat game; `visit` sees every decision with the acting
    /// seat's search inputs. Returns every state, chance states included.
    fn play(mut visit: impl FnMut(&NativeState, Input)) -> Vec<NativeState> {
        let dto: StateDto =
            serde_json::from_str(include_str!("../../tests/data/initial-state.json")).unwrap();
        let mut state = NativeState::import(&dto).unwrap();
        let mut ledger = Ledger::new(
            reward(),
            LedgerState {
                spent: [0.; 4],
                settled: vec![],
            },
        )
        .unwrap();
        let mut recalls: [Recall; 4] = Default::default();
        let mut states = vec![state.clone()];
        let mut step = 0u64;
        while !state.context().terminal {
            step += 1;
            let revision = state.context().revision;
            if let Some(chance) = state.chance_outcomes() {
                let pick = (mix("chance", step) % chance.outcomes.len() as u64) as usize;
                state = state.after_chance(revision, &chance.outcomes[pick].card).unwrap();
                states.push(state.clone());
                continue;
            }
            let seat = Seat::try_from(state.context().actor as u8).unwrap();
            let view = state.observation(seat).unwrap();
            ledger.observe(&view).unwrap();
            let recall = &mut recalls[usize::from(u8::from(seat))];
            if recall.last().is_none_or(|r| r.hand != view.own_hand) {
                recall.push(RecallEntry {
                    revision,
                    hand: view.own_hand.clone(),
                });
            }
            let context = FeatureContext {
                model_id: "frozen".into(),
                reward_id: reward().identity().unwrap(),
                profile_id: "brass-lab-teacher/1.1.0".into(),
                style: [0.; 4],
                intent: [0.; 8],
                remaining: ledger.remaining(),
                max_cost: 0.025,
                stable_descriptors: None,
            };
            visit(
                &state,
                Input {
                    view: view.clone(),
                    recall: recall.clone(),
                    context,
                    reward: reward(),
                    ledger: ledger.state(),
                    shuffle_tape: None,
                    sample_seed: step,
                },
            );
            let choices = state.legal_choices().unwrap();
            let pick = choices
                .iter()
                .max_by_key(|c| (rank(c.kind), mix(&c.id, step)))
                .unwrap()
                .clone();
            ledger.chosen(&view, &pick);
            state = state.after_choice(revision, &pick.id).unwrap();
            states.push(state.clone());
        }
        states
    }

    #[test]
    fn reused_decisions_equal_rebuilt_ones_and_leave_search_unchanged() {
        let (mut searched, mut checked) = (0, 0);
        play(|state, input| {
            if state.context().phase != Phase::Action || input.sample_seed % 9 != 0 || searched >= 8 {
                return;
            }
            searched += 1;
            let config = Config {
                simulations: 48,
                k: 8,
                seed: input.sample_seed,
                trace: true,
                ..Config::default()
            };
            let mut fast = Brass::new(input.clone(), false).unwrap();
            fast.known.get_mut().checked = true;
            let a = run(&mut fast, &mut Prior, config.clone(), None).unwrap();
            checked += fast.known.get_mut().checked_hits;
            let mut slow = Brass::new(input, false).unwrap();
            slow.known.get_mut().disabled = true;
            let b = run(&mut slow, &mut Prior, config, None).unwrap();
            assert_eq!(a.selected_action, b.selected_action);
            assert_eq!(
                serde_json::to_value(&a.nodes).unwrap(),
                serde_json::to_value(&b.nodes).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&a.trajectories).unwrap(),
                serde_json::to_value(&b.trajectories).unwrap()
            );
            assert_eq!(
                (a.metrics.completed, a.metrics.nn_leaves, a.metrics.nodes, a.metrics.edges),
                (b.metrics.completed, b.metrics.nn_leaves, b.metrics.nodes, b.metrics.edges)
            );
        });
        assert_eq!(searched, 8);
        assert!(checked > 100, "decision cache barely used: {checked}");
    }

    #[test]
    fn sampled_worlds_of_one_root_share_the_root_key() {
        let mut root = None;
        play(|state, input| {
            if root.is_none() && state.context().phase == Phase::Action && state.context().round >= 3 {
                root = Some(input);
            }
        });
        let mut game = Brass::new(root.unwrap(), false).unwrap();
        let info = game.root().unwrap();
        let mut draws = Draws::new(1, None, false);
        for _ in 0..4 {
            let mut world = game.sample(&mut draws).unwrap();
            let Position::Decision(sampled) = game.inspect(&mut world).unwrap() else {
                panic!("sampled root is not a decision");
            };
            assert!(std::sync::Arc::ptr_eq(&sampled.key, &info.key));
        }
    }

    #[test]
    fn native_accounts_and_lazy_recall_match_the_published_projection() {
        let states = play(|_, _| {});
        let mut published = ObservationMemory::new(&states[0]);
        let mut lazy = ObservationMemory::new(&states[0]);
        for state in &states[1..] {
            published.refresh_checked(state).unwrap();
            lazy.refresh_recall(state).unwrap();
            if state.chance_outcomes().is_some() {
                continue;
            }
            let view = state.observation(Seat::Zero).unwrap();
            assert_eq!(state.accounts(), Accounts::from_view(&view));
            assert_eq!(lazy.accounts_at(state), published.accounts_at(state));
            if !state.context().terminal {
                let seat = Seat::try_from(state.context().actor as u8).unwrap();
                assert_eq!(lazy.information_at(state, seat).unwrap(), published.information(seat));
            }
            for s in 0..4u8 {
                let seat = Seat::try_from(s).unwrap();
                assert_eq!(lazy.recall_len(seat), published.recall_len(seat));
            }
        }
        assert!(states.len() > 300);
    }
}
