//! Bounded diagnostic simulator objects. Handles never reuse an identity.
//! Rules and cloning stay native; JSON exists only at the diagnostic boundary.
use super::{
    belief::{PreparedRoot, SamplingDraws},
    initial::InitialSetup,
    observation::{InformationState, Observation, ObservationEnvelope, ObservationMemory, Recall},
    *,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

pub const SIMULATION_FORMAT: &str = "brass-native-simulation/1.0.0";
pub const MAX_STATES: usize = 256;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Setup {
    picked: [Vec<String>; 3],
}
impl Setup {
    fn stage(&self) -> usize {
        if self.picked[0].len() < 4 {
            0
        } else if self.picked[1].len() < 9 {
            1
        } else {
            2
        }
    }
    fn pool(&self) -> BTreeMap<String, usize> {
        let mut pool = BTreeMap::new();
        match self.stage() {
            0 => {
                for n in 0..4 {
                    pool.insert(n.to_string(), 1);
                }
            }
            1 => {
                for types in &CATALOG.merchant_types {
                    let name = serde_json::to_string(types)
                        .expect("catalog")
                        .replace(',', ", ");
                    *pool.entry(name).or_default() += 1;
                }
            }
            _ => {
                for (c, n) in CATALOG.card_names.iter().zip(&CATALOG.card_counts) {
                    if !c.starts_with("wild:") {
                        pool.insert(c.clone(), *n as usize);
                    }
                }
            }
        }
        for picked in &self.picked[self.stage()] {
            *pool.get_mut(picked).expect("validated setup") -= 1;
        }
        pool
    }
    fn outcomes(&self) -> Vec<(String, f64)> {
        let pool = self.pool();
        let total: usize = pool.values().sum();
        let stage = ["order", "merchants", "cards"][self.stage()];
        pool.into_iter()
            .filter(|(_, n)| *n > 0)
            .map(|(x, n)| (format!("chance:{stage}:{x}"), n as f64 / total as f64))
            .collect()
    }
    fn after(&self, semantic: &str) -> Result<World, CoreError> {
        require(
            self.outcomes().iter().any(|(s, _)| s == semantic),
            "illegal setup choice",
        )?;
        let mut next = self.clone();
        let item = semantic
            .splitn(3, ':')
            .nth(2)
            .expect("validated choice")
            .to_owned();
        next.picked[self.stage()].push(item);
        if next.picked[2].len() == 64 {
            let order: Vec<Seat> = next.picked[0]
                .iter()
                .map(|s| Seat::try_from(s.parse::<u8>().expect("seat")).expect("seat"))
                .collect();
            let setup = InitialSetup {
                order: order.try_into().expect("four seats"),
                merchants: next.picked[1]
                    .iter()
                    .map(|x| serde_json::from_str(x).expect("catalog types"))
                    .collect(),
                deck: next.picked[2].clone(),
            };
            Ok(World::playing(NativeState::initial(setup)?))
        } else {
            Ok(World::Setup(next))
        }
    }
    fn validate(&self) -> Result<(), CoreError> {
        require(
            self.picked[0].len() <= 4 && self.picked[1].len() <= 9 && self.picked[2].len() < 64,
            "setup checkpoint size",
        )?;
        let mut current = World::Setup(Self::default());
        for (stage, items) in self.picked.iter().enumerate() {
            for item in items {
                let World::Setup(ref setup) = current else {
                    return Err(CoreError::Invalid("completed setup checkpoint".into()));
                };
                require(stage == setup.stage(), "setup checkpoint stage")?;
                current = setup.after(&format!(
                    "chance:{}:{item}",
                    ["order", "merchants", "cards"][stage]
                ))?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum World {
    Setup(Setup),
    Playing {
        state: Box<NativeState>,
        memory: ObservationMemory,
    },
}
impl World {
    pub fn playing(state: NativeState) -> Self {
        let memory = ObservationMemory::new(&state);
        Self::Playing {
            state: Box::new(state),
            memory,
        }
    }
    pub fn snapshot(&self) -> Result<SimulationSnapshot, CoreError> {
        match self {
            Self::Setup(setup) => Ok(SimulationSnapshot {
                setup_stage: Some(["order", "merchants", "cards"][setup.stage()].into()),
                setup_outcomes: setup.outcomes(),
                context: None,
                choices: vec![],
                chance: None,
                result: None,
                information: std::array::from_fn(|_| InformationState {
                    observation: ObservationEnvelope {
                        view: None,
                        setup: true,
                        chance: None,
                    },
                    recall: vec![],
                }),
            }),
            Self::Playing { state, memory } => Ok(SimulationSnapshot {
                setup_stage: None,
                setup_outcomes: vec![],
                context: Some(state.context.clone()),
                choices: state.legal_choices()?,
                chance: state.chance_outcomes(),
                result: state.result(),
                information: std::array::from_fn(|s| {
                    memory.information(Seat::try_from(s as u8).expect("seat"))
                }),
            }),
        }
    }
    pub fn after(&self, revision: Option<u32>, semantic: &str) -> Result<Self, CoreError> {
        match self {
            Self::Setup(setup) => {
                require(revision.is_none(), "setup has no revision")?;
                setup.after(semantic)
            }
            Self::Playing { state, memory } => {
                let revision =
                    revision.ok_or_else(|| CoreError::Invalid("missing revision".into()))?;
                let next = if let Some(card) = semantic.strip_prefix("chance:card:") {
                    state.after_chance(revision, card)?
                } else {
                    state.after_choice(revision, semantic)?
                };
                let mut memory = memory.clone();
                memory.refresh(&next);
                Ok(Self::Playing {
                    state: Box::new(next),
                    memory,
                })
            }
        }
    }
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            format: SIMULATION_FORMAT.into(),
            world: match self {
                Self::Setup(s) => SavedWorld::Setup { setup: s.clone() },
                Self::Playing { state, memory } => SavedWorld::Playing {
                    state: Box::new(state.export()),
                    information: Box::new(std::array::from_fn(|s| {
                        memory.information(Seat::try_from(s as u8).expect("seat"))
                    })),
                },
            },
        }
    }
}
#[derive(Debug, Serialize)]
pub struct SimulationSnapshot {
    pub setup_stage: Option<String>,
    pub setup_outcomes: Vec<(String, f64)>,
    pub context: Option<Context>,
    pub choices: Vec<Choice>,
    pub chance: Option<ChanceOutcomes>,
    pub result: Option<ResultDto>,
    pub information: [InformationState; 4],
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    format: String,
    world: SavedWorld,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum SavedWorld {
    Setup {
        setup: Setup,
    },
    Playing {
        state: Box<StateDto>,
        information: Box<[InformationState; 4]>,
    },
}
impl Checkpoint {
    fn restore(self) -> Result<World, CoreError> {
        require(
            self.format == SIMULATION_FORMAT,
            "simulation checkpoint contract",
        )?;
        match self.world {
            SavedWorld::Setup { setup } => {
                setup.validate()?;
                Ok(World::Setup(setup))
            }
            SavedWorld::Playing { state, information } => {
                let state = NativeState::import(&state)?;
                let memory = ObservationMemory::restore_checkpoint(&state, *information)?;
                Ok(World::Playing {
                    state: Box::new(state),
                    memory,
                })
            }
        }
    }
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SimulationCommand {
    New,
    Import {
        state: Box<StateDto>,
    },
    Deserialize {
        checkpoint: Box<Checkpoint>,
    },
    Prepare {
        view: Box<Observation>,
        recall: Recall,
    },
    Sample {
        random: SamplingDraws,
    },
    Snapshot {
        handle: u64,
    },
    Step {
        handle: u64,
        revision: Option<u32>,
        semantic: String,
    },
    Clone {
        handle: u64,
    },
    Restore {
        handle: u64,
        source: u64,
    },
    Release {
        handle: u64,
    },
    Export {
        handle: u64,
    },
}
#[derive(Debug, Serialize, Default)]
pub struct SimulationReply {
    pub handle: Option<u64>,
    pub snapshot: Option<SimulationSnapshot>,
    pub checkpoint: Option<Checkpoint>,
    pub draw_count: Option<usize>,
    pub live_states: usize,
}
#[derive(Default)]
pub struct Simulator {
    worlds: BTreeMap<u64, Arc<World>>,
    sequence: u64,
    prepared: Option<PreparedRoot>,
}
impl Simulator {
    /// New actor view and complete current legal set, entirely inside Rust.
    pub fn feature_input(
        &self,
        handle: u64,
        revision: u32,
    ) -> Result<(Observation, Vec<Choice>), CoreError> {
        let World::Playing { state, .. } = self.get(handle)?.as_ref() else {
            return Err(CoreError::Invalid("setup has no neural features".into()));
        };
        require(
            state.context.revision == revision && !state.context.terminal && state.chance.is_none(),
            "stale or non-policy feature state",
        )?;
        let actor = Seat::try_from(state.context.actor as u8)
            .map_err(|_| CoreError::Invalid("feature actor".into()))?;
        Ok((state.observation(actor)?, state.legal_choices()?))
    }
    fn get(&self, handle: u64) -> Result<&Arc<World>, CoreError> {
        self.worlds
            .get(&handle)
            .ok_or_else(|| CoreError::Invalid("stale or foreign simulation handle".into()))
    }
    fn insert(&mut self, world: Arc<World>, session: u8) -> Result<SimulationReply, CoreError> {
        require(self.worlds.len() < MAX_STATES, "simulation state limit")?;
        let sequence = self
            .sequence
            .checked_add(1)
            .filter(|s| *s < u64::MAX / 2)
            .ok_or_else(|| CoreError::Invalid("handle space exhausted".into()))?;
        let handle = sequence * 2 + u64::from(session);
        let snapshot = world.snapshot()?;
        self.worlds.insert(handle, world);
        self.sequence = sequence;
        Ok(SimulationReply {
            handle: Some(handle),
            snapshot: Some(snapshot),
            ..Default::default()
        })
    }
    pub fn execute(
        &mut self,
        command: SimulationCommand,
        session: u8,
    ) -> Result<SimulationReply, CoreError> {
        require(session < 2, "simulation owner must be 0 or 1")?;
        use SimulationCommand::*;
        let mut reply = match command {
            New => self.insert(Arc::new(World::Setup(Setup::default())), session)?,
            Import { state } => self.insert(
                Arc::new(World::playing(NativeState::import(&state)?)),
                session,
            )?,
            Deserialize { checkpoint } => self.insert(Arc::new(checkpoint.restore()?), session)?,
            Prepare { view, recall } => {
                let root = PreparedRoot::new(&view, recall)?;
                let draw_count = Some(root.draw_count());
                self.prepared = Some(root);
                SimulationReply {
                    draw_count,
                    ..Default::default()
                }
            }
            Sample { random } => {
                let (state, memory, _) = self
                    .prepared
                    .as_ref()
                    .ok_or(CoreError::NoGame)?
                    .sample(random)?;
                self.insert(
                    Arc::new(World::Playing {
                        state: Box::new(state),
                        memory,
                    }),
                    session,
                )?
            }
            Clone { handle } => {
                require(self.worlds.len() < MAX_STATES, "simulation state limit")?;
                let world = Arc::clone(self.get(handle)?);
                let sequence = self
                    .sequence
                    .checked_add(1)
                    .filter(|s| *s < u64::MAX / 2)
                    .ok_or_else(|| CoreError::Invalid("handle space exhausted".into()))?;
                let handle = sequence * 2 + u64::from(session);
                self.worlds.insert(handle, world);
                self.sequence = sequence;
                SimulationReply {
                    handle: Some(handle),
                    ..Default::default()
                }
            }
            Snapshot { handle } => SimulationReply {
                handle: Some(handle),
                snapshot: Some(self.get(handle)?.snapshot()?),
                ..Default::default()
            },
            Step {
                handle,
                revision,
                semantic,
            } => {
                let next = Arc::new(self.get(handle)?.after(revision, &semantic)?);
                let snapshot = next.snapshot()?;
                self.worlds.insert(handle, next);
                SimulationReply {
                    handle: Some(handle),
                    snapshot: Some(snapshot),
                    ..Default::default()
                }
            }
            Restore { handle, source } => {
                self.get(handle)?;
                let world = Arc::clone(self.get(source)?);
                self.worlds.insert(handle, world);
                SimulationReply {
                    handle: Some(handle),
                    ..Default::default()
                }
            }
            Release { handle } => {
                self.get(handle)?;
                self.worlds.remove(&handle);
                SimulationReply::default()
            }
            Export { handle } => SimulationReply {
                handle: Some(handle),
                checkpoint: Some(self.get(handle)?.checkpoint()),
                ..Default::default()
            },
        };
        reply.live_states = self.worlds.len();
        Ok(reply)
    }
}
