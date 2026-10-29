//! Experimental adaptive-focus policy. Pure gates never manufacture time,
//! evidence or confidence from correlated trajectories.
use super::*;

pub const POLICY: &str = "adaptive-focus/2.0.0";
pub const PREPARATION_NS: u64 = 20_000_000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    #[default]
    SearchV1,
    SearchV2NoSwitch,
    SearchV2FocusOnly,
    SearchV2,
}

#[derive(Clone, Serialize)]
pub struct Progress {
    pub outcome: Outcome,
    pub cost_ns: u64,
    pub at_ns: u64,
    pub extraction_ns: u64,
}
pub trait Observer {
    fn retirement_requested(&mut self) -> bool {
        false
    }
    /// Called with completed-round evidence after pending trajectories drain.
    /// Later partial work may exist, but is excluded by the supplied watermark.
    /// True ends this tournament; false resumes the identical state/draws.
    fn checkpoint(&mut self, game: &Brass, progress: Progress) -> Result<bool>;
}

#[derive(Clone)]
pub struct Prepared {
    pub input: super::super::brass::Input,
    pub boundary: Info<Key, String>,
    pub incumbent: Vec<SupportedAction>,
}

/// One bounded preparation attempt, on the deterministic completion stream.
/// Missing support for action two is labelled legal completion, never a visit.
pub fn prepare(
    game: &Brass,
    actions: &[SupportedAction],
    budget: Budget,
    clock: &impl Clock,
) -> Result<Prepared> {
    check(budget.work_open(clock), "projection work cutoff")?;
    check(
        !actions.is_empty() && actions.len() <= 2 && actions[0].support >= 2,
        "projection requires supported complete first action",
    )?;
    let (input, boundary) = game.project(&actions[0].action, budget, clock)?;
    let mut incumbent = actions.to_vec();
    if incumbent.len() == 1 {
        let projected = Brass::new(input.clone(), false)?;
        incumbent.push(SupportedAction {
            action: plans::complete(&projected, &boundary.actions[0], budget, clock)?,
            support: 0,
            returns: [0.; 4],
        });
    }
    plans::validate(
        game,
        &incumbent
            .iter()
            .map(|a| a.action.clone())
            .collect::<Vec<_>>(),
        budget,
        clock,
    )?;
    check(budget.work_open(clock), "projection work cutoff")?;
    Ok(Prepared {
        input,
        boundary,
        incumbent,
    })
}

#[derive(Default)]
pub struct Stability {
    previous: Option<plans::Action>,
    watermarks: [u64; 2],
}
impl Stability {
    pub fn observe(&mut self, path: Option<&plans::Action>, watermarks: [u64; 2]) -> bool {
        if watermarks
            .iter()
            .zip(self.watermarks)
            .any(|(new, old)| *new <= old)
        {
            return false;
        }
        let stable = path.is_some() && self.previous.as_ref() == path;
        self.previous = path.cloned();
        self.watermarks = watermarks;
        stable
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Affordability {
    pub remaining_ns: u64,
    pub cost_ns: u64,
    pub quota: u64,
    pub effective_k: usize,
    pub useful: bool,
}
pub fn affordable(
    remaining_ns: u64,
    cost_ns: u64,
    requested: usize,
    legal: usize,
    preparation_ns: u64,
) -> Result<Affordability> {
    let quota = if cost_ns == 0 {
        0
    } else {
        remaining_ns.saturating_sub(preparation_ns) / cost_ns
    };
    let k = super::super::halving::effective_k(requested, legal, quota)?;
    Ok(Affordability {
        remaining_ns,
        cost_ns,
        quota,
        effective_k: k,
        useful: k >= 2 && quota >= (k * k.ilog2() as usize) as u64,
    })
}
