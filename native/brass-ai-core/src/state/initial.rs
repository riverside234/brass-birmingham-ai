//! Explicit four-player setup, using the immutable admitted catalog.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitialSetup {
    pub order: [Seat; 4],
    pub merchants: Vec<Vec<Industry>>,
    pub deck: Vec<String>,
}

impl NativeState {
    pub fn initial(setup: InitialSetup) -> Result<Self, CoreError> {
        require(setup.deck.len() == 64, "setup requires 64 normal cards")?;
        require(setup.merchants.len() == 9, "setup requires nine merchants")?;
        let players = std::array::from_fn(|s| Player {
            seat: Seat::try_from(s as u8).expect("four seats"),
            cash: 17,
            income_marker: 10,
            income: 0,
            vp: 0,
            bonus_vp: 0,
            link_tiles: 14,
            spent: 0,
            debt: 0,
            mat: Industry::ALL.map(|industry| Mat {
                industry,
                tiles: CATALOG
                    .tiles
                    .iter()
                    .filter(|t| seat(t.seat) == s && t.industry == industry)
                    .cloned()
                    .collect(),
            }),
        });
        let first = setup.order[0];
        Self::import(&StateDto {
            format: STATE_FORMAT.into(),
            rules: TARGET_RULES.into(),
            context: Context {
                protocol: "0.2.0".into(),
                rules: TARGET_RULES.into(),
                revision: 0,
                actor: u8::from(first) as i8,
                turn_owner: first,
                phase: Phase::Action,
                actions_remaining: 1,
                era: Era::CanalEra,
                round: 1,
                turn: 1,
                terminal: false,
                chance: false,
            },
            current_player: first,
            round_pending: false,
            debt_pending: false,
            settlement_debts: [0; 4],
            order: setup.order,
            players,
            slots: CATALOG.slots.clone(),
            links: CATALOG.links.clone(),
            merchants: CATALOG
                .merchants
                .iter()
                .zip(setup.merchants)
                .map(|(m, types)| Merchant {
                    id: m.id.clone(),
                    beer: u8::from(!types.is_empty()),
                    types: Some(types),
                    bonus: m.bonus,
                })
                .collect(),
            removed: vec![],
            retired: vec![],
            cards: Cards {
                hands: std::array::from_fn(|s| setup.deck[s * 8..s * 8 + 8].to_vec()),
                discards: std::array::from_fn(|s| vec![setup.deck[32 + s].clone()]),
                discard_starts: [0; 4],
                deck: setup.deck[36..].to_vec(),
                wild_industry: vec!["wild:industry".into(); 4],
                wild_location: vec!["wild:location".into(); 4],
            },
            coal_market: 13,
            iron_market: 8,
            pending: Pending {
                kind: String::new(),
                target: String::new(),
                merchant: String::new(),
                parts: 0,
                coal: 0,
                iron: 0,
                beer: 0,
            },
            history: vec![],
            chance: None,
            result: None,
        })
    }
}
