use crate::{
    types::{Era, Phase, Seat, StateContext},
    CoreError, DIAGNOSTIC_RULES,
};
use brass_upstream_rules::GameRunner;
use serde::Serialize;

/// Owned native instance; Clone performs upstream's deep mutable-state copy.
/// This wrapper is only accessible through explicitly diagnostic operations.
#[derive(Clone)]
pub struct DiagnosticGame {
    runner: GameRunner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticSnapshot {
    pub rules: &'static str,
    pub seed: u64,
    pub context: StateContext,
    pub cash: [u16; 4],
    pub hand_counts: [usize; 4],
    pub deck_count: usize,
}

impl DiagnosticGame {
    pub fn new(players: u8, seed: u64) -> Result<Self, CoreError> {
        if players != 4 {
            return Err(CoreError::Invalid(
                "only four-player diagnostics are enabled".into(),
            ));
        }
        let mut runner = GameRunner::new(4, Some(seed));
        runner.start_turn();
        Ok(Self { runner })
    }

    pub fn snapshot(&self) -> DiagnosticSnapshot {
        let state = &self.runner.framework.board.state;
        // Construction admits four players and the wrapper cannot mutate seats.
        let actor = match self.runner.framework.current_player {
            0 => Seat::Zero,
            1 => Seat::One,
            2 => Seat::Two,
            _ => Seat::Three,
        };
        DiagnosticSnapshot {
            rules: DIAGNOSTIC_RULES,
            seed: state.seed,
            context: StateContext {
                actor,
                era: Era::CanalEra,
                phase: Phase::Action,
                round: self.runner.round_in_phase,
                turn: self.runner.turn_count,
                actions_remaining: self.runner.actions_remaining_in_turn,
            },
            cash: std::array::from_fn(|i| state.players[i].money),
            hand_counts: std::array::from_fn(|i| state.players[i].hand.cards.len()),
            deck_count: state.deck.cards.len(),
        }
    }

    pub fn root_action_count(&mut self) -> usize {
        self.runner.start_turn().len()
    }

    /// Mutation of a private fork exercises Vec/card/board ownership.
    pub fn clone_isolated(&self) -> bool {
        let before = self.snapshot();
        let mut fork = self.clone();
        fork.runner.framework.board.state.players[0].money = 0;
        fork.runner.framework.board.state.players[0]
            .hand
            .cards
            .clear();
        fork.runner.framework.board.state.deck.cards.clear();
        fork.runner.framework.board.state.visible_vps[0] = 99;
        self.snapshot() == before
            && fork.snapshot() != before
            && self.runner.framework.board.state.visible_vps[0] != 99
    }

    pub fn shallow_size_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn private_deals_and_rng_are_owned(seed in any::<u64>()) {
            let game = DiagnosticGame::new(4, seed).unwrap();
            let control = DiagnosticGame::new(4, seed).unwrap();
            let original = &game.runner.framework.board.state;
            let reference = &control.runner.framework.board.state;
            prop_assert_eq!(&original.deck.cards, &reference.deck.cards);
            for seat in 0..4 {
                prop_assert_eq!(&original.players[seat].hand.cards, &reference.players[seat].hand.cards);
            }
            let mut fork = game.clone();
            fork.runner.framework.board.state.deck.shuffle();
            fork.runner.framework.board.state.deck.draw();
            fork.runner.framework.board.state.trade_post_beer.clear();
            fork.runner.framework.board.state.players[0].hand.cards.clear();
            prop_assert_eq!(&original.deck.cards, &reference.deck.cards);
            prop_assert_eq!(&original.trade_post_beer, &reference.trade_post_beer);
            prop_assert_eq!(&original.players[0].hand.cards, &reference.players[0].hand.cards);
            let mut next = game.clone();
            let mut expected = control.clone();
            next.runner.framework.board.state.deck.shuffle();
            expected.runner.framework.board.state.deck.shuffle();
            prop_assert_eq!(next.runner.framework.board.state.deck.cards, expected.runner.framework.board.state.deck.cards);
        }
    }
}
