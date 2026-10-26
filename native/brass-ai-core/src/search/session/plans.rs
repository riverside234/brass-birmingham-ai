//! Complete-action proofs and a separate, watermark-filtered support ledger.
use super::super::{
    brass::{Brass, Key},
    check, invalid,
    tree::{PathEdge, Tree},
    Game, Info, Position, Result,
};
use super::{
    clock::{Budget, Clock},
    TurnKey,
};
use crate::{state::dto::Kind, types::Phase};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Step {
    pub precondition: Info<Key, String>,
    pub choice: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Action {
    pub steps: Vec<Step>,
}
impl Action {
    pub fn ids(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.choice.clone()).collect()
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct SupportedAction {
    pub action: Action,
    /// Support for the complete prefix through this action, never just an edge.
    pub support: u64,
    pub returns: [f64; 4],
}

pub fn completed_prefixes(tree: &Tree<Key, String>, path: &[PathEdge]) -> Result<Vec<Action>> {
    let Some(first) = path.first() else {
        return Ok(vec![]);
    };
    let root = &tree.get(first.node)?.info;
    let turn = TurnKey::of(root)?;
    let mut actions = Vec::new();
    let mut steps = Vec::new();
    for edge in path {
        let info = &tree.get(edge.node)?.info;
        if info.actor != root.actor || TurnKey::of(info)? != turn {
            break;
        }
        let choice = info
            .key
            .choices
            .get(edge.edge)
            .ok_or_else(|| invalid("plan edge bounds"))?;
        if choice.kind == Kind::Liquidate {
            break;
        }
        steps.push(Step {
            precondition: info.clone(),
            choice: choice.id.clone(),
        });
        if choice.kind == Kind::Finish {
            actions.push(Action {
                steps: std::mem::take(&mut steps),
            });
            if actions.len() == 2 {
                break;
            }
        }
    }
    // A new NN leaf during payment/bonus/continuation does not close an action.
    Ok(actions)
}

struct Contribution {
    serial: u64,
    // Cumulative sum makes a historical watermark query logarithmic.
    value: [f64; 4],
}
struct Entry {
    prefix: Vec<Action>,
    contributions: Vec<Contribution>,
}
#[derive(Default)]
pub struct Ledger {
    entries: Vec<Entry>,
    events: usize,
    steps: usize,
}
impl Ledger {
    pub fn record(&mut self, actions: &[Action], serial: u64, value: [f64; 4]) -> Result<()> {
        check(value.iter().all(|x| x.is_finite()), "nonfinite plan return")?;
        check(
            actions.len() <= 2
                && actions.iter().all(|a| {
                    !a.steps.is_empty()
                        && a.steps.len() <= 96
                        && a.steps
                            .last()
                            .is_some_and(|s| s.choice.starts_with("finish|"))
                }),
            "incomplete plan contribution",
        )?;
        check(
            self.events + actions.len() <= 2_000_000,
            "plan contribution cap",
        )?;
        let added_steps: usize = (1..=actions.len())
            .filter(|n| !self.entries.iter().any(|e| e.prefix == actions[..*n]))
            .map(|n| actions[..n].iter().map(|a| a.steps.len()).sum::<usize>())
            .sum();
        check(self.steps + added_steps <= 65536, "plan proof memory cap")?;
        let mut staged = Vec::new();
        for n in 1..=actions.len() {
            let index = self.entries.iter().position(|e| e.prefix == actions[..n]);
            let last = index.and_then(|i| self.entries[i].contributions.last());
            check(
                last.is_none_or(|c| c.serial < serial),
                "duplicate/out-of-order plan contribution",
            )?;
            let sum = std::array::from_fn(|s| last.map_or(0., |c| c.value[s]) + value[s]);
            check(sum.iter().all(|x| x.is_finite()), "plan return overflow")?;
            staged.push((n, index, sum));
        }
        // All fallible checks precede publication of either prefix.
        for (n, index, value) in staged {
            let index = if let Some(i) = index {
                i
            } else {
                self.entries.push(Entry {
                    prefix: actions[..n].to_vec(),
                    contributions: vec![],
                });
                self.entries.len() - 1
            };
            let entry = &mut self.entries[index];
            entry.contributions.push(Contribution { serial, value });
            self.events += 1;
        }
        self.steps += added_steps;
        Ok(())
    }
    pub fn events(&self) -> usize {
        self.events
    }
    pub fn select(
        &self,
        first: &str,
        watermark: u64,
        minimum: u64,
        actor: usize,
    ) -> Vec<SupportedAction> {
        let mut chosen = Vec::<Action>::new();
        let mut result = Vec::new();
        for depth in 1..=2 {
            let mut eligible: Vec<_> = self
                .entries
                .iter()
                .filter_map(|e| {
                    if e.prefix.len() != depth
                        || e.prefix[..depth - 1] != chosen
                        || e.prefix[0].steps[0].choice != first
                    {
                        return None;
                    }
                    let count = e.contributions.partition_point(|c| c.serial <= watermark) as u64;
                    let returns = if count == 0 {
                        [0.; 4]
                    } else {
                        e.contributions[count as usize - 1].value
                    };
                    (count >= minimum).then_some((e, count, returns))
                })
                .collect();
            eligible.sort_by(|(a, an, ar), (b, bn, br)| {
                bn.cmp(an)
                    .then_with(|| (br[actor] / *bn as f64).total_cmp(&(ar[actor] / *an as f64)))
                    .then_with(|| {
                        a.prefix
                            .iter()
                            .flat_map(Action::ids)
                            .cmp(b.prefix.iter().flat_map(Action::ids))
                    })
            });
            let Some((entry, count, returns)) = eligible.first() else {
                break;
            };
            chosen = entry.prefix.clone();
            result.push(SupportedAction {
                action: chosen[depth - 1].clone(),
                support: *count,
                returns: *returns,
            });
        }
        result
    }
}

fn current(game: &Brass, world: &mut super::super::brass::World) -> Result<Info<Key, String>> {
    match game.inspect(world)? {
        Position::Decision(info) => Ok(info),
        _ => Err(invalid("plan crosses chance/terminal boundary")),
    }
}
fn open(budget: Budget, clock: &impl Clock) -> Result<()> {
    crate::work::checkpoint(crate::work::Stage::Completion)?;
    check(
        budget.delivery_open(clock),
        "completion deadline/cancellation",
    )
}

/// Replay only on an isolated permitted reconstruction. The live host must still
/// validate and commit each returned action separately when L13 integrates it.
pub fn validate(
    game: &Brass,
    actions: &[Action],
    budget: Budget,
    clock: &impl Clock,
) -> Result<Option<Info<Key, String>>> {
    check(
        !actions.is_empty() && actions.len() <= 2,
        "complete action count",
    )?;
    open(budget, clock)?;
    let mut world = game.completion_world()?;
    let first = current(game, &mut world)?;
    let turn = TurnKey::of(&first)?;
    for action in actions {
        check(
            !action.steps.is_empty() && action.steps.len() <= 96,
            "complete action step bound",
        )?;
        for (i, step) in action.steps.iter().enumerate() {
            open(budget, clock)?;
            let info = current(game, &mut world)?;
            check(
                info == step.precondition && TurnKey::of(&info)? == turn,
                "continuation precondition changed",
            )?;
            let choice = info
                .key
                .choices
                .iter()
                .find(|c| c.id == step.choice)
                .ok_or_else(|| invalid("illegal plan choice"))?;
            let closes =
                choice.kind == Kind::Finish || (turn.debt && choice.kind == Kind::Liquidate);
            check(
                closes == (i + 1 == action.steps.len()),
                "incomplete or multiple-action trace",
            )?;
            game.step(&mut world, &step.choice)?;
            open(budget, clock)?;
        }
        if turn.debt {
            check(
                actions.len() == 1,
                "debt uses a single isolated liquidation",
            )?;
        }
    }
    open(budget, clock)?;
    let next = match game.inspect(&mut world)? {
        Position::Decision(info) if TurnKey::of(&info)? == turn => Some(info),
        _ => None,
    };
    open(budget, clock)?;
    Ok(next)
}

/// Full legal deterministic completion, with no NN calls and no tree evidence.
pub fn complete(game: &Brass, first: &str, budget: Budget, clock: &impl Clock) -> Result<Action> {
    open(budget, clock)?;
    let mut world = game.completion_world()?;
    let initial = current(game, &mut world)?;
    let turn = TurnKey::of(&initial)?;
    let mut steps = Vec::new();
    for n in 0..96 {
        open(budget, clock)?;
        let info = current(game, &mut world)?;
        check(
            TurnKey::of(&info)? == turn,
            "completion left current action",
        )?;
        let choice = if n == 0 {
            info.key
                .choices
                .iter()
                .find(|c| c.id == first)
                .ok_or_else(|| invalid("fallback root legality"))?
        } else {
            info.key
                .choices
                .iter()
                .max_by_key(|c| {
                    (
                        c.kind == Kind::Finish,
                        c.kind == Kind::Beer && c.merchant.is_some(),
                        std::cmp::Reverse(&c.id),
                    )
                })
                .ok_or_else(|| invalid("empty completion legality"))?
        };
        let closes = choice.kind == Kind::Finish || (turn.debt && choice.kind == Kind::Liquidate);
        let id = choice.id.clone();
        steps.push(Step {
            precondition: info,
            choice: id.clone(),
        });
        game.step(&mut world, &id)?;
        open(budget, clock)?;
        if closes {
            return Ok(Action { steps });
        }
    }
    Err(invalid("bounded completion step cap"))
}

pub fn phase(info: &Info<Key, String>) -> Result<Phase> {
    Ok(info
        .key
        .information
        .observation
        .view
        .as_ref()
        .ok_or_else(|| invalid("missing permitted view"))?
        .context
        .phase)
}
