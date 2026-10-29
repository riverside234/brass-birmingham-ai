//! Bounded owner-to-coordinator checkpoint transport for the isolated study.
use brass_ai_core::search::{
    self,
    brass::Brass,
    session::{
        clock::{Budget, Clock, MonotonicClock},
        cooperative::SharedRoot,
        focus::{self, Method, Prepared, Progress, Stability},
        merge,
        plans::SupportedAction,
    },
    Config, Result,
};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;

pub struct Observation {
    pub shared: Arc<SharedRoot>,
    pub progress: Progress,
}
pub enum Message {
    Progress(Observation),
    Parked(Observation),
    Prepared(Result<Box<Prepared>>, u64),
}
pub enum Control {
    Pause,
    Prepare(Vec<SupportedAction>),
    Resume,
    Switch,
}
pub struct Transport {
    tx: Sender<Message>,
    rx: Receiver<Control>,
}
pub struct Monitor {
    pub shared: Arc<SharedRoot>,
    pub transport: Transport,
    pub clock: MonotonicClock,
    pub budget: Budget,
    pub pause_pending: bool,
}
impl focus::Observer for Monitor {
    fn retirement_requested(&mut self) -> bool {
        if matches!(self.transport.rx.try_recv(), Ok(Control::Pause)) {
            self.pause_pending = true;
        }
        self.pause_pending
    }
    fn checkpoint(&mut self, game: &Brass, progress: Progress) -> Result<bool> {
        // At most log2(K) snapshots per owner; bounded independently of visits.
        self.transport
            .tx
            .try_send(Message::Progress(Observation {
                shared: self.shared.clone(),
                progress: progress.clone(),
            }))
            .map_err(|_| search::invalid("checkpoint channel unavailable"))?;
        if !self.pause_pending {
            return Ok(false);
        }
        self.pause_pending = false;
        self.transport
            .tx
            .try_send(Message::Parked(Observation {
                shared: self.shared.clone(),
                progress,
            }))
            .map_err(|_| search::invalid("retirement channel unavailable"))?;
        while self.budget.work_open(&self.clock) {
            match self.transport.rx.recv_timeout(Duration::from_millis(1)) {
                Ok(Control::Resume) => return Ok(false),
                Ok(Control::Switch) => return Ok(true),
                Ok(Control::Prepare(actions)) => {
                    let began = self.clock.now_ns();
                    let prepared =
                        focus::prepare(game, &actions, self.budget, &self.clock).map(Box::new);
                    self.transport
                        .tx
                        .try_send(Message::Prepared(
                            prepared,
                            self.clock.now_ns().saturating_sub(began),
                        ))
                        .map_err(|_| search::invalid("projection reply unavailable"))?;
                }
                Ok(Control::Pause) => return Err(search::invalid("duplicate retirement")),
                Err(RecvTimeoutError::Timeout) => {}
                Err(_) => return Err(search::invalid("coordinator disconnected")),
            }
        }
        // This is a work cutoff, not external cancellation. The original
        // completed-round recommendation remains eligible for finalization.
        Ok(false)
    }
}
#[derive(Serialize)]
pub struct GateEvent {
    pub at_ns: u64,
    pub watermarks: [u64; 2],
    pub rounds: [usize; 2],
    pub first_path: Vec<String>,
    pub supported_actions: usize,
    pub stable: bool,
    pub affordability: Vec<focus::Affordability>,
    pub reason: String,
}
#[derive(Serialize)]
pub struct Metrics {
    pub method: Method,
    pub policy: &'static str,
    pub events: Vec<GateEvent>,
    pub extraction_ns: [u64; 2],
    pub switched: bool,
    pub preparation_attempts: usize,
    pub preparation_ns: u64,
    pub retirement_started_ns: Option<u64>,
    pub retirement_finished_ns: Option<u64>,
    pub switch_reason: String,
    pub fallback_used: bool,
    pub composite: Option<Composite>,
    pub permitted_successor: Option<brass_ai_core::search::brass::Input>,
}
#[derive(Serialize)]
pub struct ActionEvidence {
    pub action_ordinal: usize,
    pub root_epoch: u64,
    pub owner: Option<usize>,
    pub watermark: u64,
    pub conditional_support: u64,
    pub source: String,
    pub expected_revision: u32,
    pub path: Vec<String>,
}
#[derive(Serialize)]
pub struct Composite {
    pub version: &'static str,
    pub joint_support: Option<u64>,
    pub actions: Vec<ActionEvidence>,
}
pub struct Coordinator {
    rx: [Receiver<Message>; 2],
    tx: [Sender<Control>; 2],
    latest: [Option<Observation>; 2],
    parked: [Option<Observation>; 2],
    stability: Stability,
    pub metrics: Metrics,
    ordinary: bool,
    requested: bool,
    preparing: bool,
    first_path: Option<brass_ai_core::search::session::plans::Action>,
    pub first_selection: Option<merge::Selection>,
    pub prepared: Option<Box<Prepared>>,
}
impl Coordinator {
    pub fn new(method: Method, ordinary: bool) -> (Self, [Transport; 2]) {
        let (a, ar) = bounded(16);
        let (b, br) = bounded(16);
        let (ac, acr) = bounded(2);
        let (bc, bcr) = bounded(2);
        (
            Self {
                rx: [ar, br],
                tx: [ac, bc],
                latest: [None, None],
                parked: [None, None],
                stability: Stability::default(),
                ordinary,
                requested: false,
                preparing: false,
                first_path: None,
                first_selection: None,
                prepared: None,
                metrics: Metrics {
                    method,
                    policy: focus::POLICY,
                    events: vec![],
                    extraction_ns: [0, 0],
                    switched: false,
                    preparation_attempts: 0,
                    preparation_ns: 0,
                    retirement_started_ns: None,
                    retirement_finished_ns: None,
                    switch_reason: "no_qualified_checkpoint".into(),
                    fallback_used: false,
                    composite: None,
                    permitted_successor: None,
                },
            },
            [Transport { tx: a, rx: acr }, Transport { tx: b, rx: bcr }],
        )
    }
    fn resume(&mut self, reason: &str) {
        self.metrics.switch_reason = reason.into();
        for tx in &self.tx {
            let _ = tx.try_send(Control::Resume);
        }
        self.preparing = false;
    }
    pub fn owner_finished(&mut self) {
        if self.requested && !self.metrics.switched {
            self.resume("owner_finished_before_switch");
        }
    }
    pub fn poll(&mut self, config: &Config, budget: Budget, clock: &impl Clock) -> Result<()> {
        // One event from each lane per pass preserves consecutive observations
        // even when the UI/diagnostic poller briefly falls behind.
        loop {
            let mut changed = false;
            for lane in 0..2 {
                if let Ok(message) = self.rx[lane].try_recv() {
                    match message {
                        Message::Progress(message) => {
                            self.metrics.extraction_ns[lane] += message.progress.extraction_ns;
                            self.latest[lane] = Some(message);
                            changed = true;
                        }
                        Message::Parked(message) => {
                            self.parked[lane] = Some(message);
                            changed = true;
                        }
                        Message::Prepared(result, ns) => {
                            self.metrics.preparation_ns = ns;
                            match result {
                                Ok(prepared) => {
                                    let useful = self.parked.iter().all(|p| {
                                        p.as_ref().is_some_and(|p| {
                                            focus::affordable(
                                                budget.remaining_work(clock.now_ns()),
                                                p.progress.cost_ns,
                                                config.k,
                                                prepared.boundary.actions.len(),
                                                0,
                                            )
                                            .is_ok_and(|a| a.useful)
                                        })
                                    });
                                    if useful && budget.work_open(clock) {
                                        self.metrics.permitted_successor =
                                            Some(prepared.input.clone());
                                        self.prepared = Some(prepared);
                                        self.metrics.switched = true;
                                        self.metrics.switch_reason = "certified_focus".into();
                                        self.metrics.retirement_finished_ns = Some(clock.now_ns());
                                        for tx in &self.tx {
                                            tx.try_send(Control::Switch).map_err(|_| {
                                                search::invalid("switch acknowledgment unavailable")
                                            })?;
                                        }
                                    } else {
                                        self.resume("insufficient_work_after_preparation");
                                    }
                                }
                                Err(e) => self.resume(&format!("projection_rejected: {e}")),
                            }
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
            if self.requested {
                if !self.preparing && self.metrics.switch_reason == "retirement_requested" {
                    if let [Some(a), Some(b)] = &self.parked {
                        let selection = merge::select(
                            &a.shared,
                            [&a.progress.outcome, &b.progress.outcome],
                            config.parameters(),
                            2,
                        )?;
                        let first = selection
                            .actions
                            .as_ref()
                            .and_then(|a| a.first())
                            .map(|a| &a.action);
                        if first != self.first_path.as_ref() {
                            self.resume("recommendation_changed_during_retirement");
                        } else if !budget.work_open(clock) {
                            self.resume("retirement_work_cutoff");
                        } else {
                            self.preparing = true;
                            self.metrics.preparation_attempts = 1;
                            self.tx[0]
                                .try_send(Control::Prepare(selection.actions.clone().unwrap()))
                                .map_err(|_| search::invalid("projection request unavailable"))?;
                            self.first_selection = Some(selection);
                        }
                    }
                }
                continue;
            }
            let [Some(a), Some(b)] = &self.latest else {
                continue;
            };
            search::check(a.shared.info == b.shared.info, "mixed checkpoint roots")?;
            let outcomes = [&a.progress.outcome, &b.progress.outcome];
            let watermarks = outcomes.map(|o| o.snapshot.as_ref().unwrap().support_watermark);
            let rounds = outcomes.map(|o| o.snapshot.as_ref().unwrap().completed_rounds);
            let selection = merge::select(&a.shared, outcomes, config.parameters(), 2)?;
            let actions = selection.actions.as_deref().unwrap_or_default();
            let first = actions.first().map(|a| &a.action);
            let stable = self.stability.observe(first, watermarks);
            let affordability = [&a.progress, &b.progress]
                .iter()
                .map(|p| {
                    focus::affordable(
                        budget.remaining_work(clock.now_ns()),
                        p.cost_ns,
                        config.k,
                        a.shared.info.actions.len(),
                        focus::PREPARATION_NS,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let reason = if !self.ordinary {
                "unsupported_initial_boundary"
            } else if first.is_none() {
                "no_complete_first_action_support"
            } else if !stable {
                "path_unstable_or_owner_not_advanced"
            } else if !affordability.iter().all(|a| a.useful) {
                "insufficient_remaining_work"
            } else {
                "cheap_gate_qualified"
            };
            self.metrics.events.push(GateEvent {
                at_ns: clock.now_ns(),
                watermarks,
                rounds,
                first_path: first.map_or_else(Vec::new, |a| a.ids()),
                supported_actions: actions.len(),
                stable,
                affordability,
                reason: reason.into(),
            });
            if reason == "cheap_gate_qualified"
                && matches!(
                    self.metrics.method,
                    Method::SearchV2FocusOnly | Method::SearchV2
                )
            {
                self.requested = true;
                self.first_path = first.cloned();
                self.metrics.retirement_started_ns = Some(clock.now_ns());
                self.metrics.switch_reason = "retirement_requested".into();
                if self
                    .tx
                    .iter()
                    .any(|tx| tx.try_send(Control::Pause).is_err())
                {
                    self.resume("owner_finished_before_retirement");
                }
            }
        }
        Ok(())
    }

    /// Search epochs give conditional support, never a fictitious joint count.
    pub fn compose(
        &mut self,
        second: &merge::Selection,
        focused: [&crate::pair::Lane; 2],
    ) -> Result<merge::Selection> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| search::invalid("missing incumbent"))?;
        let mut selection = self
            .first_selection
            .clone()
            .ok_or_else(|| search::invalid("missing first evidence"))?;
        let mut actions = prepared.incumbent.clone();
        let replacement = second
            .actions
            .as_ref()
            .and_then(|a| a.first())
            .filter(|a| a.support >= 2 && a.action.steps[0].precondition == prepared.boundary);
        self.metrics.fallback_used = replacement.is_none();
        let mut evidence = Vec::new();
        for ordinal in 0..2 {
            let fresh = ordinal == 1 && replacement.is_some();
            if fresh {
                actions[1] = replacement.unwrap().clone();
            }
            let owner = if fresh {
                second.source_session
            } else {
                selection.source_session
            };
            let watermark = if fresh {
                owner
                    .and_then(|o| focused[o].outcome.snapshot.as_ref())
                    .map_or(0, |s| s.support_watermark)
            } else {
                owner
                    .and_then(|o| self.parked[o].as_ref())
                    .and_then(|p| p.progress.outcome.snapshot.as_ref())
                    .map_or(0, |s| s.support_watermark)
            };
            let action = &actions[ordinal];
            evidence.push(ActionEvidence {
                action_ordinal: ordinal + 1,
                root_epoch: if fresh { 2 } else { 1 },
                owner,
                watermark,
                conditional_support: action.support,
                source: if fresh {
                    "focused_search"
                } else if action.support == 0 {
                    "incumbent_legal_completion"
                } else {
                    "incumbent_original_prefix"
                }
                .into(),
                expected_revision: action.action.steps[0]
                    .precondition
                    .key
                    .information
                    .observation
                    .view
                    .as_ref()
                    .unwrap()
                    .context
                    .revision,
                path: action.action.ids(),
            });
        }
        self.metrics.composite = Some(Composite {
            version: "brass-composite-plan/2.0.0",
            joint_support: None,
            actions: evidence,
        });
        // The legacy field means joint prefix support; it cannot describe an
        // independently searched action. Conditional counts are above instead.
        if replacement.is_some() {
            for a in &mut actions {
                a.support = 0;
                a.returns = [0.; 4];
            }
        }
        selection.actions = Some(actions);
        selection.provenance = if replacement.is_some() {
            "composite_focus_v2"
        } else {
            "incumbent_fallback_v2"
        }
        .into();
        Ok(selection)
    }
}
