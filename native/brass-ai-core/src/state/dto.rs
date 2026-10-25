//! Versioned, trusted simulation-state boundary. Never a policy observation.
use crate::types::{Era, Phase, Seat};
use serde::{Deserialize, Serialize};

pub const STATE_FORMAT: &str = "brass-native-state/1.0.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Industry {
    Brewery,
    CoalMine,
    CottonMill,
    IronWorks,
    Manufacturer,
    Pottery,
}
impl Industry {
    pub const ALL: [Self; 6] = [
        Self::Brewery,
        Self::CoalMine,
        Self::CottonMill,
        Self::IronWorks,
        Self::Manufacturer,
        Self::Pottery,
    ];
    pub fn index(self) -> usize {
        self as usize
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkEra {
    Canal,
    Rail,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bonus {
    Develop,
    VP,
    Income,
    Money,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Build,
    Network,
    Sell,
    Develop,
    Loan,
    Pass,
    Scout,
    Finish,
    Coal,
    Iron,
    Beer,
    BonusDevelop,
    BonusSkip,
    Liquidate,
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Network => "network",
            Self::Sell => "sell",
            Self::Develop => "develop",
            Self::Loan => "loan",
            Self::Pass => "pass",
            Self::Scout => "scout",
            Self::Finish => "finish",
            Self::Coal => "coal",
            Self::Iron => "iron",
            Self::Beer => "beer",
            Self::BonusDevelop => "bonus_develop",
            Self::BonusSkip => "bonus_skip",
            Self::Liquidate => "liquidate",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub protocol: String,
    pub rules: String,
    pub revision: u32,
    pub actor: i8,
    pub turn_owner: Seat,
    pub phase: Phase,
    pub actions_remaining: u8,
    pub era: Era,
    pub round: u8,
    pub turn: u8,
    pub terminal: bool,
    pub chance: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tile {
    pub id: String,
    pub seat: Seat,
    #[serde(rename = "type")]
    pub industry: Industry,
    pub level: u8,
    pub flipped: bool,
    pub resource: u8,
    pub vp: u8,
    pub link_vp: u8,
    pub income: u8,
    pub cost: u8,
    pub coal: u8,
    pub iron: u8,
    pub beer: u8,
    pub developable: bool,
    pub canal_only: bool,
    pub rail_only: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mat {
    #[serde(rename = "type")]
    pub industry: Industry,
    pub tiles: Vec<Tile>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Player {
    pub seat: Seat,
    pub cash: i32,
    pub income_marker: u8,
    pub income: i8,
    pub vp: i32,
    pub bonus_vp: i32,
    pub link_tiles: u8,
    pub spent: i32,
    pub mat: [Mat; 6],
    pub debt: i32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Slot {
    pub id: String,
    pub types: Vec<Industry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tile: Option<Tile>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Link {
    pub id: String,
    pub regions: Vec<String>,
    pub era: LinkEra,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<Seat>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Merchant {
    pub id: String,
    pub types: Vec<Industry>,
    pub beer: u8,
    pub bonus: Bonus,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cards<C> {
    pub hands: [Vec<C>; 4],
    pub discards: [Vec<C>; 4],
    pub discard_starts: [usize; 4],
    pub deck: Vec<C>,
    pub wild_industry: Vec<C>,
    pub wild_location: Vec<C>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pending {
    pub kind: String,
    pub target: String,
    pub merchant: String,
    pub parts: u8,
    pub coal: u8,
    pub iron: u8,
    pub beer: u8,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChanceKind {
    Draw,
    RailShuffle,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chance<C> {
    pub kind: ChanceKind,
    pub needed: u8,
    pub pool: Vec<C>,
    pub picked: Vec<C>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Choice {
    pub id: String,
    pub kind: Kind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub industry: Option<Industry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merchant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cards: Option<Vec<String>>,
}
impl Choice {
    pub fn semantic_id(&self) -> String {
        let industry = self.industry.map(|i| format!("{i:?}")).unwrap_or_default();
        let cards = self.cards.as_ref().map(|x| x.join(",")).unwrap_or_default();
        [
            self.kind.name(),
            self.target.as_deref().unwrap_or(""),
            &industry,
            self.card.as_deref().unwrap_or(""),
            self.merchant.as_deref().unwrap_or(""),
            &cards,
        ]
        .join("|")
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct History {
    pub seat: Seat,
    pub era: Era,
    pub choice: Choice,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultSeat {
    pub seat: Seat,
    pub vp: i32,
    pub income: i8,
    pub cash: i32,
    pub rank: u8,
    pub utility: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultDto {
    pub objective: String,
    pub seats: [ResultSeat; 4],
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateDto {
    pub format: String,
    pub rules: String,
    pub context: Context,
    pub current_player: Seat,
    pub round_pending: bool,
    pub debt_pending: bool,
    pub settlement_debts: [i32; 4],
    pub order: [Seat; 4],
    pub players: [Player; 4],
    pub slots: Vec<Slot>,
    pub links: Vec<Link>,
    pub merchants: Vec<Merchant>,
    pub removed: Vec<String>,
    pub retired: Vec<String>,
    pub cards: Cards<String>,
    pub coal_market: u8,
    pub iron_market: u8,
    pub pending: Pending,
    pub history: Vec<History>,
    pub chance: Option<Chance<String>>,
    pub result: Option<ResultDto>,
}
