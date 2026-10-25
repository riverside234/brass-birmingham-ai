use crate::CoreError;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, IntoPrimitive, TryFromPrimitive, Serialize, Deserialize,
)]
#[repr(u8)]
#[serde(try_from = "u8", into = "u8")]
pub enum Seat {
    Zero,
    One,
    Two,
    Three,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Era {
    CanalEra,
    RailEra,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Action,
    Payment,
    Continuation,
    Bonus,
    Debt,
    Chance,
    Terminal,
}

/// Compact context value, independent of host UUIDs and JSON containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateContext {
    pub actor: Seat,
    pub era: Era,
    pub phase: Phase,
    pub round: u32,
    pub turn: u32,
    pub actions_remaining: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ChoiceId(String);

impl TryFrom<String> for ChoiceId {
    type Error = CoreError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
            return Err(CoreError::Invalid(
                "invalid semantic choice identity".into(),
            ));
        }
        Ok(Self(value))
    }
}
impl From<ChoiceId> for String {
    fn from(value: ChoiceId) -> Self {
        value.0
    }
}

/// Later staged adapters implement this interface after L04/L05 qualification.
/// No upstream atomic-action type is silently treated as a C# staged choice.
pub trait StagedSimulation: Clone {
    fn context(&self) -> StateContext;
    fn legal_choices(&self, output: &mut Vec<ChoiceId>) -> Result<(), CoreError>;
    fn apply(&mut self, choice: &ChoiceId) -> Result<(), CoreError>;
}
