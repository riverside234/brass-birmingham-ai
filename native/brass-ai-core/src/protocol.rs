use crate::state::{
    belief::SamplingDraws,
    dto::{Choice, StateDto, STATE_FORMAT},
    observation::{InformationState, Observation, Recall},
    ChanceOutcomes, IndexSnapshot,
};
use crate::{
    diagnostic::DiagnosticSnapshot, CoreError, CONTROL_PROTOCOL, DIAGNOSTIC_RULES, TARGET_RULES,
    UPSTREAM_COMMIT,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub protocol: String,
    pub request_id: String,
    pub request: Operation,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Inference {
        session: u8,
        command: crate::evaluation::InferenceCommand,
    },
    Features {
        session: u8,
        command: crate::features::session::FeatureCommand,
    },
    Hello {
        sessions: u8,
        threads_per_session: u8,
        required_capabilities: Vec<String>,
    },
    Simulation {
        session: u8,
        command: crate::state::simulation::SimulationCommand,
    },
    DiagnosticNew {
        session: u8,
        players: u8,
        seed: u64,
    },
    DiagnosticSnapshot {
        session: u8,
    },
    DiagnosticCloneCheck {
        session: u8,
    },
    StateImport {
        session: u8,
        state: Box<StateDto>,
    },
    StateExport {
        session: u8,
    },
    StateSave {
        session: u8,
    },
    StateRestore {
        session: u8,
    },
    StateCloneCheck {
        session: u8,
    },
    RulesSnapshot {
        session: u8,
    },
    RulesApply {
        session: u8,
        revision: u32,
        choice: String,
    },
    RulesChance {
        session: u8,
        revision: u32,
        card: String,
    },
    ObservationSnapshot {
        session: u8,
        seat: crate::types::Seat,
    },
    BeliefPrepare {
        session: u8,
        view: Box<Observation>,
        recall: Recall,
    },
    BeliefSample {
        session: u8,
        random: SamplingDraws,
    },
    ImportReference {
        contract: String,
    },
    Search {
        contract: String,
    },
    Shutdown,
}

impl Request {
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.protocol != CONTROL_PROTOCOL {
            return Err(CoreError::UnsupportedContract(self.protocol.clone()));
        }
        if self.request_id.is_empty()
            || self.request_id.len() > 128
            || self.request_id.chars().any(char::is_control)
        {
            return Err(CoreError::Invalid(
                "request_id must have 1..128 bytes without control characters".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct BuildIdentity {
    pub version: &'static str,
    pub source_sha256: &'static str,
    pub cargo_lock_sha256: &'static str,
    pub rustc: &'static str,
    pub target: &'static str,
    pub profile: &'static str,
    pub opt_level: &'static str,
    pub debug: &'static str,
    pub rustflags: &'static str,
    pub linker: &'static str,
    pub upstream_commit: &'static str,
}
impl BuildIdentity {
    pub fn current() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            source_sha256: env!("BRASS_SOURCE_SHA256"),
            cargo_lock_sha256: env!("BRASS_LOCK_SHA256"),
            rustc: env!("BRASS_RUSTC"),
            target: env!("BRASS_TARGET"),
            profile: env!("BRASS_PROFILE"),
            opt_level: env!("BRASS_OPT_LEVEL"),
            debug: env!("BRASS_DEBUG"),
            rustflags: env!("BRASS_CARGO_ENCODED_RUSTFLAGS"),
            linker: env!("BRASS_LINKER"),
            upstream_commit: UPSTREAM_COMMIT,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Reply {
    pub protocol: &'static str,
    pub request_id: Option<String>,
    pub response: Response,
}
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Response {
    Inference {
        session: u8,
        data: Box<crate::evaluation::InferenceResponse>,
    },
    Features {
        session: u8,
        data: Box<crate::features::session::FeatureResponse>,
    },
    Simulation {
        session: u8,
        data: Box<crate::state::simulation::SimulationReply>,
    },
    Hello {
        build: BuildIdentity,
        capabilities: Vec<&'static str>,
        sessions: u8,
        threads_per_session: u8,
        diagnostic_rules: &'static str,
        target_rules: &'static str,
        target_rules_qualified: bool,
        state_format: &'static str,
        max_frame_bytes: usize,
    },
    Diagnostic {
        session: u8,
        owner: String,
        state: DiagnosticSnapshot,
        clone_isolated: Option<bool>,
    },
    State {
        session: u8,
        owner: String,
        state: Box<StateDto>,
        indexes: Box<IndexSnapshot>,
        clone_isolated: Option<bool>,
        saved: bool,
    },
    Error {
        code: &'static str,
        message: String,
    },
    Rules {
        session: u8,
        owner: String,
        state: Box<StateDto>,
        choices: Vec<Choice>,
        chance: Option<ChanceOutcomes>,
    },
    Observation {
        session: u8,
        owner: String,
        information: Box<InformationState>,
        reference_digest: String,
    },
    Prepared {
        session: u8,
        owner: String,
        sampler: &'static str,
        unseen_cards: usize,
        draw_count: usize,
    },
    Sampled {
        session: u8,
        owner: String,
        sampler: &'static str,
        draws: Vec<u32>,
        state: Box<StateDto>,
        choices: Vec<Choice>,
    },
    Shutdown,
}
pub const CAPABILITIES: &[&str] = &[
    "inference",
    "features",
    "simulation",
    "diagnostic_new",
    "diagnostic_snapshot",
    "diagnostic_clone_check",
    "state_import",
    "state_export",
    "state_save",
    "state_restore",
    "state_clone_check",
    "rules_snapshot",
    "rules_apply",
    "rules_chance",
    "observation_snapshot",
    "belief_prepare",
    "belief_sample",
];
impl Reply {
    pub fn new(id: Option<String>, response: Response) -> Self {
        Self {
            protocol: CONTROL_PROTOCOL,
            request_id: id,
            response,
        }
    }
    pub fn error(id: Option<String>, error: CoreError) -> Self {
        Self::new(
            id,
            Response::Error {
                code: error.code(),
                message: error.to_string(),
            },
        )
    }
    pub fn hello(id: String) -> Self {
        Self::new(
            Some(id),
            Response::Hello {
                build: BuildIdentity::current(),
                capabilities: CAPABILITIES.to_vec(),
                sessions: 2,
                threads_per_session: 1,
                diagnostic_rules: DIAGNOSTIC_RULES,
                target_rules: TARGET_RULES,
                target_rules_qualified: true,
                state_format: STATE_FORMAT,
                max_frame_bytes: crate::framing::MAX_FRAME_BYTES,
            },
        )
    }
}
