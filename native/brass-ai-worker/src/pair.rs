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
}
enum Root {
    Produce(Sender<Result<Arc<SharedRoot>>>),
    Consume(Receiver<Result<Arc<SharedRoot>>>),
}
enum Command {
    Run(Box<Run>),
    Finish {
        input: Box<Input>,
        outcome: Box<Outcome>,
        selection: Box<merge::Selection>,
        clock: MonotonicClock,
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
                                    Command::Finish {
                                        input,
                                        outcome,
                                        mut selection,
                                        clock,
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
                                            let mut result =
                                                s.adopt(*input, outcome, actions, &clock)?;
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
        });
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
        self.active = Some(Active {
            request: self.sequence,
            input: input.clone(),
            lanes: [None, None],
            finished: [None, None],
            selection: None,
            error: None,
            received: [false, false],
            finishing: false,
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
        for lane in 0..2 {
            if a.received[lane] {
                continue;
            }
            match self.owners[lane].rx.try_recv() {
                Ok(reply) => {
                    a.received[lane] = true;
                    let result = match reply {
                        Reply::Lane(r) if !a.finishing => r.map(|r| a.lanes[lane] = Some(*r)),
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
            let lanes = [a.lanes[0].as_ref().unwrap(), a.lanes[1].as_ref().unwrap()];
            let selected = merge::select(
                &lanes[0].shared,
                [&lanes[0].outcome, &lanes[1].outcome],
                b.config.parameters(),
                2,
            );
            let selection = match selected {
                Ok(s) => s,
                Err(e) => {
                    self.active = None;
                    return Err(e);
                }
            };
            a.finishing = true;
            a.received = [false, false];
            for (lane, result) in lanes.iter().enumerate() {
                if self.owners[lane]
                    .tx
                    .send(Command::Finish {
                        input: Box::new(a.input.clone()),
                        outcome: Box::new(result.outcome.clone()),
                        selection: Box::new(selection.clone()),
                        clock: b.clock.clone(),
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
        let retired_ns = b.clock.now_ns();
        let response_ready_ns = b.clock.now_ns();
        search::check(
            response_ready_ns < result.budget.deadline_ns,
            "late coordinator result",
        )?;
        let lanes = a.lanes.map(|l| l.unwrap());
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
    }
    let s = session.as_mut().ok_or(CoreError::SessionFailed)?;
    let mut effective = s.budget();
    effective.narrow(0, run.remaining_deadline);
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
                let _root_scope =
                    work::Scope::narrow_remaining(effective.remaining_work(clock.now_ns()));
                let (_, info) = s.preview(run.input.clone())?;
                let stats = if info.actions.len() > 1
                    && session::plans::phase(&info)? != brass_ai_core::types::Phase::Debt
                    && !s.continuation_matches(&info)
                    && effective.work_open(clock)
                {
                    match neural.evaluate(&info) {
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
    let outcome = s.decide_cooperative(
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
    )?;
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
