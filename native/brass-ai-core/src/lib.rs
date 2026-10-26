//! Native rules, permitted observations and detached belief worlds.
pub mod diagnostic;
pub mod evaluation;
pub mod features;
pub mod framing;
pub mod protocol;
pub mod search;
pub mod state;
pub mod types;
pub mod work;

pub const CONTROL_PROTOCOL: &str = "brass-native-control/0.1.0";
pub const DIAGNOSTIC_RULES: &str = "daiyu-upstream-diagnostic/0.1.0";
pub const TARGET_RULES: &str = "steam-brass-patch-1.1.0";
pub const UPSTREAM_COMMIT: &str = "f8f0246718d6d3fb3f301742d4154811c90d5813";

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("unsupported contract: {0}")]
    UnsupportedContract(String),
    #[error("capability is not qualified: {0}")]
    Unavailable(&'static str),
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("session has no state for the requested capability")]
    NoGame,
    #[error("native session failed and was retired")]
    SessionFailed,
}

impl CoreError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedContract(_) => "unsupported_contract",
            Self::Unavailable(_) => "unavailable_capability",
            Self::Invalid(_) => "invalid_request",
            Self::NoGame => "no_game",
            Self::SessionFailed => "session_failed",
        }
    }
}
