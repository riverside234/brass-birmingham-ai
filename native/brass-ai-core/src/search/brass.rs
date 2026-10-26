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
        self.information.hash(h);
        self.choices.hash(h);
        self.settled.hash(h);
        let c = &self.context;
        c.model_id.hash(h);
        c.reward_id.hash(h);
        c.profile_id.hash(h);
        for x in c
            .style
            .iter()
            .chain(&c.intent)
            .chain(&c.remaining)
            .chain([&c.max_cost])
        {
            bits(*x).hash(h);
        }
        // Only unknown deployment descriptors are admitted in this initial search.
    }
}
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
    root_seat: Seat,
    shuffle_tape: Option<Vec<Vec<u32>>>,
    sample_seed: u64,
    samples: usize,
    pub sampled_indices: Vec<Vec<u32>>,
    record: bool,
}
pub struct World {
    state: NativeState,
    memory: ObservationMemory,
    ledger: Ledger,
    decision: Option<(Observation, Vec<Choice>)>,
}
impl Brass {
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
        Ok(Self {
            prepared,
            root_info,
            root_ledger: ledger,
            context: input.context,
            reward: input.reward,
            root_seat: input.view.seat,
            shuffle_tape: input.shuffle_tape,
            sample_seed: input.sample_seed,
            samples: 0,
            sampled_indices: vec![],
            record,
        })
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
        let information = memory.information(seat);
        check(
            information
                .observation
                .view
                .as_ref()
                .is_some_and(|v| v.seat == seat && v.context.actor == actor as i8)
                && information.observation.chance.is_none(),
            "unsupported actor view",
        )?;
        let mut choices = state.legal_choices()?;
        choices.sort_by(|a, b| a.id.cmp(&b.id));
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
        w.ledger.observe(&w.state.observation(self.root_seat)?)?;
        if let Some(result) = w.state.result() {
            return Ok(Position::Terminal(self.reward.terminal(&result)?));
        }
        let info = Self::information(&w.state, &w.memory, &w.ledger, &self.context)?;
        w.decision = Some((
            info.key
                .information
                .observation
                .view
                .as_ref()
                .ok_or_else(|| invalid("missing actor view"))?
                .clone(),
            info.key.choices.clone(),
        ));
        Ok(Position::Decision(info))
    }
    fn step(&self, w: &mut World, action: &String) -> Result<()> {
        let revision = w.state.context().revision;
        let next = if let Some(card) = action.strip_prefix("chance:card:") {
            w.state.after_chance(revision, card)?
        } else {
            let (view, choices) = w
                .decision
                .take()
                .ok_or_else(|| invalid("step without decision"))?;
            let choice = choices
                .iter()
                .find(|c| &c.id == action)
                .ok_or_else(|| invalid("selected action outside legal set"))?;
            let next = w.state.after_legal_choice(revision, choice)?;
            w.ledger.chosen(&view, choice);
            next
        };
        w.memory.refresh_checked(&next)?;
        w.state = next;
        Ok(())
    }
    fn spent(&self, w: &World) -> [f64; 4] {
        w.ledger.spent
    }
}
