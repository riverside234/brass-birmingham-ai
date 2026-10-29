//! Two persistent compute threads. Search, features and ORT run on their owner;
//! the caller only coordinates envelopes, pooled root evidence and retirement.
use brass_ai_core::{
    evaluation::{BatchPolicy, EvaluationStats},
    search::{
        self,
        brass::Input,
        session::{
            self,
            clock::{Budget, Clock, MonotonicClock},
            cooperative::SharedRoot,
            merge,
            plans::SupportedAction,
            Identity, Mode, Outcome, Session, Settings, TurnKey,
        },
        Config, Draws, LeafEvaluator,
    },
    work::{self, PollStats, Probe},
    CoreError,
};
use brass_ai_inference::{
    search::{Batched, Buffers},
    Control, Evaluator,
};
use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use serde::Serialize;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, CoreError>;

#[derive(Clone)]
struct Binding {
    identity: Identity,
    turn: TurnKey,
    config: Config,
    budget: Budget,
    clock: MonotonicClock,
    reset: bool,
    method: session::focus::Method,
    switched: bool,
}
struct Run {
    binding: Binding,
    request: u64,
    input: Input,
    mode: Mode,
    policy: BatchPolicy,
    remaining_deadline: u64,
    root: Root,
    probe: Option<Probe>,
    progress: Option<crate::focus::Transport>,
    projected: bool,
    first_action: Option<session::plans::Action>,
}
enum Root {
    Produce(Sender<Result<Arc<SharedRoot>>>),
    Consume(Receiver<Result<Arc<SharedRoot>>>),
}
enum Command {
    Run(Box<Run>),
    Focus(Box<Run>),
    Finish {
        input: Box<Input>,
        outcome: Box<Outcome>,
        selection: Box<merge::Selection>,
        clock: MonotonicClock,
        fallback: Option<Vec<SupportedAction>>,
    },
    Stop,
}
enum Reply {
    Lane(Result<Box<Lane>>),
    Finished(Result<Box<Outcome>>),
}
#[derive(Serialize)]
pub struct Lane {
    pub outcome: Outcome,
    pub neural: EvaluationStats,
    pub polls: PollStats,
    pub owner: String,
    pub started_ns: u64,
    pub retired_ns: u64,
    pub root_wait_ns: u64,
    #[serde(skip)]
    shared: Arc<SharedRoot>,
}
struct Owner {
    tx: Sender<Command>,
    rx: Receiver<Reply>,
    join: Option<JoinHandle<()>>,
}
struct Active {
    request: u64,
    input: Input,
    lanes: [Option<Lane>; 2],
    finished: [Option<Outcome>; 2],
    selection: Option<merge::Selection>,
    error: Option<CoreError>,
    received: [bool; 2],
    finishing: bool,
    focus: Option<crate::focus::Coordinator>,
    first_lanes: Option<[Lane; 2]>,
    mode: Mode,
    policy: BatchPolicy,
    remaining_deadline: u64,
}
#[derive(Serialize)]
pub struct Decision {
    pub identity: &'static str,
    pub request_id: u64,
    pub sessions: usize,
    pub threads_per_session: usize,
    pub lanes: [Lane; 2],
    pub selection: merge::Selection,
    pub result: Outcome,
    pub response_ready_ns: u64,
    pub retired_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus: Option<crate::focus::Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focused_lanes: Option<[Lane; 2]>,
}
pub struct Pair {
    owners: [Owner; 2],
    binding: Option<Binding>,
    active: Option<Active>,
    sequence: u64,
}
impl Pair {
    /// Evaluators are created and destroyed on their eventual compute thread.
    /// In particular E need not be Send (CpuEvaluator deliberately is not).
    pub fn new<E: Evaluator + 'static, F: Fn(usize) -> Result<E> + Send + Sync + 'static>(
        factory: F,
    ) -> Result<Self> {
        let factory = Arc::new(factory);
        let mut owners = Vec::new();
        for lane in 0..2 {
            let (tx, commands) = bounded(1);
            let (responses, rx) = bounded(1);
            let (ready, initial) = bounded(1);
            let factory = factory.clone();
            let join = thread::Builder::new()
                .name(format!("brass-search-{lane}"))
                .spawn(move || {
                    let mut evaluator = match factory(lane) {
                        Ok(e) => {
                            let _ = ready.send(Ok(()));
                            e
                        }
                        Err(e) => {
                            let _ = ready.send(Err(e));
                            return;
                        }
                    };
                    let mut buffers = Buffers::default();
                    let mut session: Option<Session> = None;
                    while let Ok(command) = commands.recv() {
                        if matches!(command, Command::Stop) {
                            break;
                        }
                        let result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                || match command {
                                    Command::Run(run) => Reply::Lane(
                                        run_lane(
                                            *run,
                                            lane,
                                            &mut session,
                                            &mut evaluator,
                                            &mut buffers,
                                        )
                                        .map(Box::new),
                                    ),
                                    Command::Focus(run) => Reply::Lane((|| {
                                        let original = session.as_mut().ok_or(CoreError::SessionFailed)?;
                                        let mut projected = Some(if run.binding.method == session::focus::Method::SearchV2 {
                                            let first = run.first_action.as_ref().ok_or_else(|| search::invalid("missing retained prefix"))?;
                                            let _scope = work::Scope::enter(work::Control {
                                                deadline: run.binding.clock.instant(original.budget().work_deadline_ns),
                                                cancelled: run.binding.clock.cancelled.clone(), probe: None });
                                            original.projected_owner_reusing(&run.input, first, &run.binding.clock)?
                                        } else { original.projected_owner(&run.input)? });
                                        let result = run_lane(*run, lane, &mut projected, &mut evaluator, &mut buffers);
                                        original.recover_projected_cache(projected.as_mut().unwrap());
                                        result.map(Box::new)
                                    })()),
                                    Command::Finish {
                                        input,
                                        outcome,
                                        mut selection,
                                        clock,
                                        fallback,
                                    } => {
                                        let control = work::Control {
                                            deadline: clock.instant(outcome.budget.deadline_ns),
                                            cancelled: clock.cancelled.clone(),
                                            probe: None,
                                        };
                                        let _scope = work::Scope::enter(control);
                                        let result = (|| {
                                            let s =
                                                session.as_mut().ok_or(CoreError::SessionFailed)?;
                                            let actions =
                                                if let Some(actions) = selection.actions.take() {
                                                    actions
                                                } else {
                                                    let (game, _) = s.preview((*input).clone())?;
                                                    vec![SupportedAction {
                                                        action: session::plans::complete(
                                                            &game,
                                                            &selection.proposal,
                                                            s.budget(),
                                                            &clock,
                                                        )?,
                                                        support: 0,
                                                        returns: [0.; 4],
                                                    }]
                                                };
                                            let mut outcome = *outcome;
                                            outcome.provenance = selection.provenance.clone();
                                            let adopted = s.adopt((*input).clone(), outcome.clone(), actions, &clock);
                                            let mut result = match adopted {
                                                Ok(result) => result,
                                                Err(error) => {
                                                    if let Some(fallback) = fallback {
                                                        let mut result = s.adopt(*input, outcome, fallback, &clock)?;
                                                        result.provenance = format!("incumbent_after_rejected_composition: {error}");
                                                        return Ok(result);
                                                    }
                                                    return Err(error);
                                                }
                                            };
                                            result.provenance = selection.provenance;
                                            Ok(result)
                                        })();
                                        Reply::Finished(result.map(Box::new))
                                    }
                                    Command::Stop => unreachable!(),
                                },
                            ));
                        let fatal = result.is_err();
                        let reply = result.unwrap_or(Reply::Lane(Err(CoreError::SessionFailed)));
                        if responses.send(reply).is_err() || fatal {
                            break;
                        }
                    }
                })
                .map_err(|e| search::invalid(e.to_string()))?;
            owners.push(Owner {
                tx,
                rx,
                join: Some(join),
            });
            if let Err(e) = initial.recv().unwrap_or(Err(CoreError::SessionFailed)) {
                for mut o in owners {
                    let _ = o.tx.send(Command::Stop);
                    if let Some(j) = o.join.take() {
                        let _ = j.join();
                    }
                }
                return Err(e);
            }
        }
        Ok(Self {
            owners: owners.try_into().ok().expect("two owners"),
            binding: None,
            active: None,
            sequence: 0,
        })
    }
    pub fn busy(&self) -> bool {
        self.active.is_some()
    }
    /// Diagnostic lifecycle observation; grants no authority to change clocks.
    pub fn focusing(&self) -> bool {
        self.active
            .as_ref()
            .is_some_and(|a| a.first_lanes.is_some() && !a.finishing)
    }
    /// Explicit admission only. A retry or changed sub-choice cannot call this
    /// while work remains owned; the live caller adds its generation guards L13.
    pub fn admit(
        &mut self,
        identity: Identity,
        input: &Input,
        config: Config,
        allowance_ns: u64,
    ) -> Result<()> {
        search::check(!self.busy(), "pair busy until both owners retire")?;
        config.validate()?;
        let turn = TurnKey::from_context(&input.view.context, u8::from(input.view.seat) as usize);
        if let Some(old) = &self.binding {
            search::check(
                old.identity != identity || old.turn != turn,
                "same turn cannot replenish its clock",
            )?;
            search::check(
                identity.generation >= old.identity.generation,
                "stale generation admission",
            )?;
        }
        let clock = MonotonicClock::new(Arc::new(AtomicBool::new(false)));
        let budget = Budget::new(0, allowance_ns)?;
        self.binding = Some(Binding {
            identity,
            turn,
            config,
            budget,
            clock,
            reset: true,
            method: session::focus::Method::SearchV1,
            switched: false,
        });
        Ok(())
    }
    /// Research admission only. Local packages continue to admit search_v1.
    pub fn method(&mut self, method: session::focus::Method) -> Result<()> {
        search::check(!self.busy(), "method change while busy")?;
        let b = self.binding.as_mut().ok_or(CoreError::NoGame)?;
        search::check(b.reset, "method must be frozen at turn admission")?;
        b.method = method;
        Ok(())
    }
    pub fn start(
        &mut self,
        input: Input,
        mode: Mode,
        policy: BatchPolicy,
        remaining_ns: u64,
        probes: [Option<Probe>; 2],
    ) -> Result<u64> {
        search::check(!self.busy(), "pair backpressure: drain both owners")?;
        session::cooperative::validate_policy(policy)?;
        let binding = self.binding.as_mut().ok_or(CoreError::NoGame)?;
        search::check(
            TurnKey::from_context(&input.view.context, u8::from(input.view.seat) as usize)
                == binding.turn,
            "different turn requires admission",
        )?;
        search::check(!binding.clock.cancelled(), "retired turn")?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| search::invalid("request exhausted"))?;
        let remaining_deadline = binding
            .budget
            .deadline_ns
            .min(binding.clock.now_ns().saturating_add(remaining_ns));
        let (root_tx, root_rx) = bounded(1);
        let roots = [Root::Produce(root_tx), Root::Consume(root_rx)];
        let (focus, mut progress) = if binding.method != session::focus::Method::SearchV1 {
            let ordinary = input.view.context.phase == brass_ai_core::types::Phase::Action
                && input.view.context.actions_remaining == 2
                && !binding.switched;
            let (coordinator, senders) = crate::focus::Coordinator::new(binding.method, ordinary);
            (Some(coordinator), senders.map(Some))
        } else {
            (None, [None, None])
        };
        self.active = Some(Active {
            request: self.sequence,
            input: input.clone(),
            lanes: [None, None],
            finished: [None, None],
            selection: None,
            error: None,
            received: [false, false],
            finishing: false,
            focus,
            first_lanes: None,
            mode: mode.clone(),
            policy,
            remaining_deadline,
        });
        for (lane, (root, probe)) in roots.into_iter().zip(probes).enumerate() {
            let mut input = input.clone();
            // Root perturbations are common; worlds/chance streams are private.
            if lane == 1 {
                input.sample_seed ^= 0x9e3779b97f4a7c15;
                input.shuffle_tape = None;
            }
            let run = Run {
                binding: binding.clone(),
                request: self.sequence,
                input,
                mode: mode.clone(),
                policy,
                remaining_deadline,
                root,
                probe,
                progress: progress[lane].take(),
                projected: false,
                first_action: None,
            };
            if self.owners[lane]
                .tx
                .send(Command::Run(Box::new(run)))
                .is_err()
            {
                let active = self.active.as_mut().expect("admitted before dispatch");
                active.error = Some(CoreError::SessionFailed);
                active.received[lane..].fill(true);
                binding.clock.cancelled.store(true, Ordering::Release);
                break;
            }
        }
        binding.reset = false;
        Ok(self.sequence)
    }
    pub fn cancel(&self, ticket: u64) -> Result<()> {
        search::check(
            self.active.as_ref().is_some_and(|a| a.request == ticket),
            "stale pair ticket",
        )?;
        self.binding
            .as_ref()
            .expect("active binding")
            .clock
            .cancelled
            .store(true, Ordering::Release);
        Ok(())
    }
    pub fn poll(&mut self, ticket: u64) -> Result<Option<Decision>> {
        search::check(
            self.active.as_ref().is_some_and(|a| a.request == ticket),
            "stale pair ticket",
        )?;
        let b = self.binding.as_ref().expect("active binding");
        let a = self.active.as_mut().expect("matched ticket");
        if let Some(focus) = &mut a.focus {
            let mut effective = b.budget;
            effective.narrow(0, a.remaining_deadline);
            if let Err(e) = focus.poll(&b.config, effective, &b.clock) {
                a.error = Some(e);
                b.clock.cancelled.store(true, Ordering::Release);
            }
        }
        for lane in 0..2 {
            if a.received[lane] {
                continue;
            }
            match self.owners[lane].rx.try_recv() {
                Ok(reply) => {
                    a.received[lane] = true;
                    let result = match reply {
                        Reply::Lane(r) if !a.finishing => r.map(|r| {
                            a.lanes[lane] = Some(*r);
                            if let Some(focus) = &mut a.focus {
                                focus.owner_finished();
                            }
                        }),
                        Reply::Finished(r) if a.finishing => r.map(|r| a.finished[lane] = Some(*r)),
                        _ => Err(CoreError::SessionFailed),
                    };
                    if let Err(e) = result {
                        a.error = Some(e);
                        b.clock.cancelled.store(true, Ordering::Release);
                    }
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    a.received[lane] = true;
                    a.error = Some(CoreError::SessionFailed);
                    b.clock.cancelled.store(true, Ordering::Release);
                }
            }
        }
        if !a.received.iter().all(|x| *x) {
            return Ok(None);
        }
        if let Some(error) = a.error.take() {
            self.active = None;
            return Err(error);
        }
        if b.clock.cancelled() || !b.budget.delivery_open(&b.clock) {
            self.active = None;
            return Err(search::invalid("cancelled/expired pair retired"));
        }
        if !a.finishing {
            if a.first_lanes.is_none() && a.focus.as_ref().is_some_and(|f| f.metrics.switched) {
                // Both original owners retired. Original live sessions stay
                // untouched while temporary hypothetical owners borrow caches.
                let focused = a.focus.as_ref().unwrap().prepared.as_ref().unwrap();
                let input = focused.input.clone();
                let (tx, rx) = bounded(1);
                let roots = [Root::Produce(tx), Root::Consume(rx)];
                a.first_lanes = Some([a.lanes[0].take().unwrap(), a.lanes[1].take().unwrap()]);
                a.received = [false, false];
                for (lane, root) in roots.into_iter().enumerate() {
                    let mut input = input.clone();
                    if lane == 1 {
                        input.sample_seed ^= 0x9e3779b97f4a7c15;
                    }
                    let mut binding = b.clone();
                    binding.reset = false;
                    let run = Run {
                        binding,
                        request: a.request,
                        input,
                        mode: a.mode.clone(),
                        policy: a.policy,
                        remaining_deadline: a.remaining_deadline,
                        root,
                        probe: None,
                        progress: None,
                        projected: true,
                        first_action: Some(focused.incumbent[0].action.clone()),
                    };
                    if self.owners[lane]
                        .tx
                        .send(Command::Focus(Box::new(run)))
                        .is_err()
                    {
                        a.error = Some(CoreError::SessionFailed);
                        a.received[lane..].fill(true);
                        b.clock.cancelled.store(true, Ordering::Release);
                        break;
                    }
                }
                return Ok(None);
            }
            let lanes = [a.lanes[0].as_ref().unwrap(), a.lanes[1].as_ref().unwrap()];
            let selected = merge::select(
                &lanes[0].shared,
                [&lanes[0].outcome, &lanes[1].outcome],
                b.config.parameters(),
                2,
            );
            let mut selection = match selected {
                Ok(s) => s,
                Err(e) => {
                    self.active = None;
                    return Err(e);
                }
            };
            if a.first_lanes.is_some() {
                selection = match a.focus.as_mut().unwrap().compose(&selection, lanes) {
                    Ok(selection) => selection,
                    Err(error) => {
                        // Both owners have retired: a rejected envelope must
                        // not leave the coordinator permanently busy.
                        self.active = None;
                        return Err(error);
                    }
                };
            }
            a.finishing = true;
            a.received = [false, false];
            let original = a
                .first_lanes
                .as_ref()
                .map(|l| [&l[0], &l[1]])
                .unwrap_or(lanes);
            for (lane, result) in original.iter().enumerate() {
                if self.owners[lane]
                    .tx
                    .send(Command::Finish {
                        input: Box::new(a.input.clone()),
                        outcome: Box::new(result.outcome.clone()),
                        selection: Box::new(selection.clone()),
                        clock: b.clock.clone(),
                        fallback: a
                            .focus
                            .as_ref()
                            .and_then(|f| f.prepared.as_ref())
                            .map(|p| p.incumbent.clone()),
                    })
                    .is_err()
                {
                    a.error = Some(CoreError::SessionFailed);
                    a.received[lane..].fill(true);
                    b.clock.cancelled.store(true, Ordering::Release);
                    break;
                }
            }
            a.selection = Some(selection);
            return Ok(None);
        }
        let mut a = self.active.take().expect("complete pair");
        let result = a.finished[0].take().unwrap();
        let second = a.finished[1].take().unwrap();
        search::check(
            !result.actions.is_empty() && result.action_ids == second.action_ids,
            "coordinator completion disagreement",
        )?;
        if result
            .provenance
            .starts_with("incumbent_after_rejected_composition:")
        {
            let focus = a
                .focus
                .as_mut()
                .ok_or_else(|| search::invalid("fallback missing focus evidence"))?;
            let mut empty = focus
                .first_selection
                .clone()
                .ok_or_else(|| search::invalid("fallback missing first evidence"))?;
            empty.actions = None;
            // compose(None) uses only the retained original evidence. In
            // particular, rejected focused support must not survive delivery.
            let selection = focus.compose(
                &empty,
                [a.lanes[0].as_ref().unwrap(), a.lanes[1].as_ref().unwrap()],
            )?;
            search::check(
                selection
                    .actions
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|a| a.action.ids())
                    .collect::<Vec<_>>()
                    == result.action_ids,
                "fallback delivery/evidence disagreement",
            )?;
            a.selection = Some(selection);
        }
        let retired_ns = b.clock.now_ns();
        let response_ready_ns = b.clock.now_ns();
        search::check(
            response_ready_ns < result.budget.deadline_ns,
            "late coordinator result",
        )?;
        let latest = a.lanes.map(|l| l.unwrap());
        let (lanes, focused_lanes) = if let Some(first) = a.first_lanes {
            (first, Some(latest))
        } else {
            (latest, None)
        };
        if a.focus.as_ref().is_some_and(|f| f.metrics.switched) {
            self.binding.as_mut().unwrap().switched = true;
        }
        Ok(Some(Decision {
            identity: merge::IDENTITY,
            request_id: ticket,
            sessions: 2,
            threads_per_session: 1,
            lanes,
            selection: a.selection.unwrap(),
            result,
            response_ready_ns,
            retired_ns,
            focus: a.focus.map(|f| f.metrics),
            focused_lanes,
        }))
    }
}
impl Drop for Pair {
    fn drop(&mut self) {
        if let Some(b) = &self.binding {
            b.clock.cancelled.store(true, Ordering::Release);
        }
        for o in &self.owners {
            let _ = o.tx.send(Command::Stop);
        }
        for o in &mut self.owners {
            if let Some(join) = o.join.take() {
                let _ = join.join();
            }
        }
    }
}
fn run_lane<E: Evaluator>(
    run: Run,
    lane: usize,
    session: &mut Option<Session>,
    evaluator: &mut E,
    buffers: &mut Buffers,
) -> Result<Lane> {
    let b = &run.binding;
    let clock = &b.clock;
    let started_ns = clock.now_ns();
    let scope = work::Scope::enter(work::Control {
        deadline: clock.instant(b.budget.deadline_ns),
        cancelled: clock.cancelled.clone(),
        probe: run.probe,
    });
    let mut config = b.config.clone();
    if lane == 1 {
        config.seed ^= 0xd1b54a32d192ed03;
    }
    if b.reset {
        *session = Some(Session::new(
            b.identity.clone(),
            &run.input,
            Settings {
                search: config,
                ..Default::default()
            },
            b.budget,
        )?);
        if b.method == session::focus::Method::SearchV2 {
            session.as_mut().unwrap().enable_descendant_reuse()?;
        }
    }
    let s = session.as_mut().ok_or(CoreError::SessionFailed)?;
    let mut effective = s.budget();
    effective.narrow(0, run.remaining_deadline);
    let _remaining_scope =
        work::Scope::narrow_remaining(effective.deadline_ns.saturating_sub(clock.now_ns()));
    let mut neural = Batched {
        evaluator,
        buffers,
        control: Control {
            deadline: clock.instant(effective.work_deadline_ns),
            cancelled: clock.cancelled.clone(),
        },
        generation: b.identity.generation,
        request: run.request,
        owner: lane,
        sequence: 0,
        stats: Default::default(),
    };
    let wait = Instant::now();
    let shared = match run.root {
        Root::Produce(sender) => {
            let result = (|| {
                // Root identity/legal preparation is also needed by bounded
                // completion when only the delivery reserve remains. Keep it
                // inside the narrowed delivery deadline; NN work still gets
                // its stricter work deadline below. A zero work budget must
                // not cancel the legal fallback before it can be constructed.
                let (_, info) = s.preview(run.input.clone())?;
                let stats = if info.actions.len() > 1
                    && session::plans::phase(&info)? != brass_ai_core::types::Phase::Debt
                    && !s.continuation_matches(&info)
                    && effective.work_open(clock)
                {
                    let _root_scope =
                        work::Scope::narrow_remaining(effective.remaining_work(clock.now_ns()));
                    let evaluated = if run.projected {
                        s.evaluate_projected_root(&info, &mut neural, clock)
                    } else {
                        neural.evaluate(&info)
                    };
                    match evaluated {
                        Ok(s) => Some(s),
                        Err(e) => {
                            if effective.work_open(clock) {
                                return Err(e);
                            } else {
                                None
                            }
                        }
                    }
                } else {
                    None
                };
                let mut draws = Draws::new(s.next_root_seed(), None, false);
                let uniforms = (0..info.actions.len())
                    .map(|_| draws.uniform())
                    .collect::<Result<_>>()?;
                Ok(Arc::new(SharedRoot {
                    info,
                    stats,
                    uniforms,
                }))
            })();
            match result {
                Ok(root) => {
                    let _ = sender.send(Ok(root.clone()));
                    root
                }
                Err(e) => {
                    let _ = sender.send(Err(search::invalid(e.to_string())));
                    return Err(e);
                }
            }
        }
        Root::Consume(receiver) => loop {
            search::check(s.budget().delivery_open(clock), "root wait retired")?;
            match receiver.recv_timeout(Duration::from_millis(1)) {
                Ok(root) => break root?,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(_) => return Err(CoreError::SessionFailed),
            }
        },
    };
    let root_wait_ns = if lane == 1 {
        brass_ai_inference::nanos(wait)
    } else {
        0
    };
    let outcome = if run.projected {
        s.decide_projected(
            run.request,
            run.input,
            run.mode,
            clock,
            &mut neural,
            &shared,
            run.policy,
        )?
    } else if let Some(transport) = run.progress {
        let mut monitor = crate::focus::Monitor {
            shared: shared.clone(),
            transport,
            clock: clock.clone(),
            budget: effective,
            pause_pending: false,
        };
        s.decide_observed(
            &b.identity,
            run.request,
            run.input,
            run.remaining_deadline,
            run.mode,
            clock,
            &mut neural,
            &shared,
            run.policy,
            &mut monitor,
        )?
    } else {
        s.decide_cooperative(
            &b.identity,
            run.request,
            run.input,
            run.remaining_deadline,
            run.mode,
            clock,
            &mut neural,
            None,
            &shared,
            run.policy,
        )?
    };
    let neural_stats = neural.stats;
    Ok(Lane {
        outcome,
        neural: neural_stats,
        polls: scope.stats(),
        owner: thread::current().name().unwrap_or("unnamed").into(),
        started_ns,
        retired_ns: clock.now_ns(),
        root_wait_ns,
        shared,
    })
}
