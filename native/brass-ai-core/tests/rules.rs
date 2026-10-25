use brass_ai_core::state::{
    dto::{Kind, StateDto},
    NativeState,
};
use brass_upstream_rules::market::pricing::MarketPricing;
use proptest::prelude::*;

fn initial() -> StateDto {
    serde_json::from_str(include_str!("data/initial-state.json")).unwrap()
}

#[test]
fn complete_pass_game_has_two_eras_no_final_income_and_shared_win() {
    let mut state = NativeState::import(&initial()).unwrap();
    let mut steps = 0;
    let mut rail_shuffles = 0;
    loop {
        let dto = state.export();
        assert_eq!(NativeState::import(&dto).unwrap().export(), dto);
        if dto.context.terminal {
            assert_eq!(dto.context.round, 8);
            assert_eq!(dto.context.turn, 4);
            for result in dto.result.unwrap().seats {
                assert_eq!(result.cash, 17);
                assert_eq!(result.vp, 0);
                assert_eq!(result.rank, 1);
                assert_eq!(result.utility, 0.25);
            }
            break;
        }
        if let Some(chance) = state.chance_outcomes() {
            if chance.remaining == 64 {
                rail_shuffles += 1;
            }
            assert!(
                (chance.outcomes.iter().map(|x| x.probability).sum::<f64>() - 1.0).abs() < 1e-12
            );
            state
                .resolve_chance(dto.context.revision, &chance.outcomes[0].card)
                .unwrap();
        } else {
            let choices = state.legal_choices().unwrap();
            let c = choices
                .iter()
                .find(|c| c.kind == Kind::Pass)
                .or_else(|| choices.iter().find(|c| c.kind == Kind::Finish))
                .unwrap();
            state.apply_choice(dto.context.revision, &c.id).unwrap();
        }
        steps += 1;
        assert!(steps < 600);
    }
    assert_eq!(rail_shuffles, 1);
}

#[test]
fn overflow_after_discard_rolls_back_current_state_and_save() {
    let mut dto = initial();
    let owner = dto.context.actor as usize;
    dto.players[owner].cash = i32::MAX;
    let mut state = NativeState::import(&dto).unwrap();
    let saved = state.snapshot();
    let loan = state
        .legal_choices()
        .unwrap()
        .into_iter()
        .find(|c| c.kind == Kind::Loan)
        .unwrap();
    assert!(state.apply_choice(dto.context.revision, &loan.id).is_err());
    assert_eq!(state.export(), dto);
    state.restore(&saved);
    assert_eq!(state.export(), dto);
}

#[test]
fn upstream_market_prices_match_target_for_all_bounded_supplies() {
    for maximum in [10u8, 14] {
        for amount in 0..=maximum {
            for needed in 0..=2 {
                let expected: u16 = (0..needed)
                    .map(|i| {
                        if amount > i {
                            u16::from((maximum - amount + i) / 2 + 1)
                        } else {
                            u16::from(maximum / 2 + 1)
                        }
                    })
                    .sum();
                let actual = if maximum == 14 {
                    MarketPricing::coal_price(amount, needed)
                } else {
                    MarketPricing::iron_price(amount, needed)
                };
                assert_eq!(actual, expected);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn lawful_staged_prefixes_preserve_import_and_forks(selections in prop::collection::vec(any::<u16>(),1..40)) {
        let mut state=NativeState::import(&initial()).unwrap();
        for selection in selections {
            let before=state.export();let saved=state.snapshot();let mut fork=state.clone();
            if let Some(chance)=fork.chance_outcomes() {
                fork.resolve_chance(before.context.revision,&chance.outcomes[usize::from(selection)%chance.outcomes.len()].card).unwrap();
            } else {
                let choices=fork.legal_choices().unwrap();if choices.is_empty(){break;}
                fork.apply_choice(before.context.revision,&choices[usize::from(selection)%choices.len()].id).unwrap();
            }
            prop_assert_eq!(state.export(),before.clone());
            let after=fork.export();prop_assert_eq!(NativeState::import(&after).unwrap().export(),after.clone());
            fork.restore(&saved);prop_assert_eq!(fork.export(),before);
            state=NativeState::import(&after).unwrap();
        }
    }
}
