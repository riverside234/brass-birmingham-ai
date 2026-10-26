//! Bounded control transport; exactly two independent native state owners.
mod inference;
use brass_ai_core::{
    diagnostic::DiagnosticGame,
    framing::{read_frame, write_frame},
    protocol::{Operation, Reply, Request, Response, CAPABILITIES},
    state::{
        belief::{PreparedRoot, SamplingDraws, SAMPLER},
        dto::StateDto,
        observation::{Observation, ObservationMemory, Recall},
        NativeState, StateSnapshot,
    },
    types::Seat,
    CoreError,
};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::{
    io::{self, Read, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    thread::{self, JoinHandle},
};

enum Command {
    LoadModel(String, String),
    Evaluate(Box<inference::Run>),
    Features(brass_ai_core::features::session::FeatureCommand),
    Simulation(brass_ai_core::state::simulation::SimulationCommand),
    New {
        players: u8,
        seed: u64,
    },
    Snapshot,
    CloneCheck,
    Import(Box<StateDto>),
    Export,
    Save,
    Restore,
    StateCloneCheck,
    RulesSnapshot,
    Apply(u32, String),
    Chance(u32, String),
    Observe(Seat),
    Prepare(Box<Observation>, Recall),
    Sample(SamplingDraws),
    #[cfg(test)]
    Panic,
}
type Job = (Command, Sender<Result<Response, CoreError>>);

#[derive(Default)]
struct Session {
    inference: inference::Engine,
    features: brass_ai_core::features::session::FeatureSession,
    simulator: brass_ai_core::state::simulation::Simulator,
    diagnostic: Option<DiagnosticGame>,
    state: Option<NativeState>,
    saved: Option<StateSnapshot>,
    observation: ObservationMemory,
    saved_observation: Option<ObservationMemory>,
    prepared: Option<PreparedRoot>,
}

struct Owner {
    sender: Option<Sender<Job>>,
    exited: Receiver<()>,
    join: Option<JoinHandle<()>>,
}

impl Owner {
    fn spawn(session: u8) -> io::Result<Self> {
        let (sender, receiver) = bounded::<Job>(1);
        let (alive, exited) = bounded::<()>(0);
        let join = thread::Builder::new()
            .name(format!("brass-session-{session}"))
            .spawn(move || {
                // Closing this channel wakes callers even if an unconsumed job
                // still owns their reply sender in the disconnected job queue.
                let _alive = alive;
                let mut game = Session::default();
                let owner = format!("{:?}", thread::current().id());
                for (command, reply) in receiver {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        execute(&mut game, command, session, &owner)
                    }));
                    match result {
                        Ok(response) => {
                            let _ = reply.send(response);
                        }
                        Err(_) => {
                            let _ = reply.send(Err(CoreError::SessionFailed));
                            // A possibly partially mutated state is never reused.
                            break;
                        }
                    }
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            exited,
            join: Some(join),
        })
    }

    fn call(&self, command: Command) -> Result<Response, CoreError> {
        let (sender, receiver) = bounded(1);
        self.sender
            .as_ref()
            .ok_or(CoreError::SessionFailed)?
            .send((command, sender))
            .map_err(|_| CoreError::SessionFailed)?;
        crossbeam_channel::select_biased! {
            recv(receiver) -> result => result.map_err(|_| CoreError::SessionFailed)?,
            recv(self.exited) -> _ => Err(CoreError::SessionFailed),
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        // Closing the only request sender retires the owner even on stdin EOF.
        self.sender.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn execute(
    game: &mut Session,
    command: Command,
    session: u8,
    owner: &str,
) -> Result<Response, CoreError> {
    let check_clone = matches!(command, Command::CloneCheck);
    let state_clone = matches!(command, Command::StateCloneCheck);
    let rules = matches!(
        command,
        Command::RulesSnapshot | Command::Apply(..) | Command::Chance(..)
    );
    let semantic = matches!(
        command,
        Command::Import(_)
            | Command::Export
            | Command::Save
            | Command::Restore
            | Command::StateCloneCheck
    );
    match command {
        Command::LoadModel(path, digest) => {
            return Ok(Response::Inference {
                session,
                data: Box::new(game.inference.load(path, digest, owner)?),
            })
        }
        Command::Evaluate(run) => {
            return Ok(Response::Inference {
                session,
                data: Box::new(game.inference.run(*run, &game.simulator, owner)?),
            })
        }
        Command::Features(command) => {
            return Ok(Response::Features {
                session,
                data: Box::new(game.features.execute(command, &game.simulator, session)?),
            })
        }
        Command::Simulation(command) => {
            return Ok(Response::Simulation {
                session,
                data: Box::new(game.simulator.execute(command, session)?),
            })
        }
        Command::New { players, seed } => {
            let replacement = DiagnosticGame::new(players, seed)?;
            game.diagnostic = Some(replacement);
        }
        Command::Snapshot | Command::CloneCheck => {}
        Command::Import(dto) => {
            // Build and validate the entire replacement before touching live or saved state.
            let replacement = NativeState::import(&dto)?;
            game.observation = ObservationMemory::new(&replacement);
            game.state = Some(replacement);
        }
        Command::Observe(seat) => {
            game.state.as_ref().ok_or(CoreError::NoGame)?;
            let information = game.observation.information(seat);
            let reference_digest = information.reference_digest();
            return Ok(Response::Observation {
                session,
                owner: owner.into(),
                information: Box::new(information),
                reference_digest,
            });
        }
        Command::Prepare(view, recall) => {
            let replacement = PreparedRoot::new(&view, recall)?;
            let response = Response::Prepared {
                session,
                owner: owner.into(),
                sampler: SAMPLER,
                unseen_cards: replacement.unseen_cards(),
                draw_count: replacement.draw_count(),
            };
            game.prepared = Some(replacement);
            return Ok(response);
        }
        Command::Sample(random) => {
            let (next, memory, draws) = game
                .prepared
                .as_ref()
                .ok_or(CoreError::NoGame)?
                .sample(random)?;
            let response = Response::Sampled {
                session,
                owner: owner.into(),
                sampler: SAMPLER,
                draws,
                state: Box::new(next.export()),
                choices: next.legal_choices()?,
            };
            game.state = Some(next);
            game.observation = memory;
            return Ok(response);
        }
        Command::Export | Command::StateCloneCheck => {}
        Command::RulesSnapshot => {}
        Command::Apply(revision, choice) => {
            let next = game
                .state
                .as_ref()
                .ok_or(CoreError::NoGame)?
                .after_choice(revision, &choice)?;
            let response = rules_response(&next, session, owner)?;
            game.observation.refresh(&next);
            game.state = Some(next);
            return Ok(response);
        }
        Command::Chance(revision, card) => {
            let next = game
                .state
                .as_ref()
                .ok_or(CoreError::NoGame)?
                .after_chance(revision, &card)?;
            let response = rules_response(&next, session, owner)?;
            game.observation.refresh(&next);
            game.state = Some(next);
            return Ok(response);
        }
        Command::Save => {
            game.saved = Some(game.state.as_ref().ok_or(CoreError::NoGame)?.snapshot());
            game.saved_observation = Some(game.observation.clone());
        }
        Command::Restore => {
            let saved = game
                .saved
                .as_ref()
                .ok_or_else(|| CoreError::Invalid("session has no saved state".into()))?;
            game.state.as_mut().ok_or(CoreError::NoGame)?.restore(saved);
            game.observation = game
                .saved_observation
                .as_ref()
                .expect("paired save")
                .clone();
        }
        #[cfg(test)]
        Command::Panic => panic!("test owner failure"),
    }
    if rules {
        let state = game.state.as_ref().ok_or(CoreError::NoGame)?;
        return rules_response(state, session, owner);
    }
    if semantic {
        let state = game.state.as_ref().ok_or(CoreError::NoGame)?;
        return Ok(Response::State {
            session,
            owner: owner.to_owned(),
            state: Box::new(state.export()),
            indexes: Box::new(state.indexes()),
            clone_isolated: state_clone.then(|| state.clone_isolated()),
            saved: game.saved.is_some(),
        });
    }
    let game = game.diagnostic.as_ref().ok_or(CoreError::NoGame)?;
    Ok(Response::Diagnostic {
        session,
        owner: owner.to_owned(),
        state: game.snapshot(),
        clone_isolated: check_clone.then(|| game.clone_isolated()),
    })
}

fn rules_response(state: &NativeState, session: u8, owner: &str) -> Result<Response, CoreError> {
    // Even response generation may reject an unreachable trusted import. Build
    // the complete response before publishing a successor to its owner.
    Ok(Response::Rules {
        session,
        owner: owner.to_owned(),
        state: Box::new(state.export()),
        choices: state.legal_choices()?,
        chance: state.chance_outcomes(),
    })
}

#[derive(Default)]
struct Server {
    owners: Option<[Owner; 2]>,
    active: [Option<inference::Active>; 2],
    generations: [u64; 2],
    sequence: u64,
}
impl Drop for Server {
    fn drop(&mut self) {
        for active in self.active.iter().flatten() {
            active.cancel();
        }
    }
}
impl Server {
    fn handle(&mut self, request: Request) -> Reply {
        let id = Some(request.request_id.clone());
        if let Err(error) = request.validate() {
            return Reply::error(id, error);
        }
        if let Operation::Hello {
            sessions,
            threads_per_session,
            required_capabilities,
        } = &request.request
        {
            let error = if self.owners.is_some() {
                Some(CoreError::Invalid("hello has already completed".into()))
            } else if *sessions != 2 || *threads_per_session != 1 {
                Some(CoreError::Invalid(
                    "requires two sessions, one thread each".into(),
                ))
            } else if required_capabilities
                .iter()
                .any(|cap| !CAPABILITIES.contains(&cap.as_str()))
            {
                Some(CoreError::Unavailable("requested handshake capability"))
            } else {
                None
            };
            if let Some(error) = error {
                return Reply::error(id, error);
            }
            let owners = (|| Ok::<_, io::Error>([Owner::spawn(0)?, Owner::spawn(1)?]))();
            match owners {
                Ok(owners) => self.owners = Some(owners),
                Err(_) => return Reply::error(id, CoreError::SessionFailed),
            }
            return Reply::hello(request.request_id);
        }
        let Some(owners) = &self.owners else {
            return Reply::error(id, CoreError::Invalid("hello must complete first".into()));
        };
        let (session, command) = match request.request {
            Operation::Inference { session, command } => {
                return match self.inference(session, command) {
                    Ok(response) => Reply::new(id, response),
                    Err(error) => Reply::error(id, error),
                }
            }
            Operation::Features { session, command } => (session, Command::Features(command)),
            Operation::Simulation { session, command } => (session, Command::Simulation(command)),
            Operation::DiagnosticNew {
                session,
                players,
                seed,
            } => (session, Command::New { players, seed }),
            Operation::DiagnosticSnapshot { session } => (session, Command::Snapshot),
            Operation::DiagnosticCloneCheck { session } => (session, Command::CloneCheck),
            Operation::StateImport { session, state } => (session, Command::Import(state)),
            Operation::StateExport { session } => (session, Command::Export),
            Operation::StateSave { session } => (session, Command::Save),
            Operation::StateRestore { session } => (session, Command::Restore),
            Operation::StateCloneCheck { session } => (session, Command::StateCloneCheck),
            Operation::RulesSnapshot { session } => (session, Command::RulesSnapshot),
            Operation::ObservationSnapshot { session, seat } => (session, Command::Observe(seat)),
            Operation::BeliefPrepare {
                session,
                view,
                recall,
            } => (session, Command::Prepare(view, recall)),
            Operation::BeliefSample { session, random } => (session, Command::Sample(random)),
            Operation::RulesApply {
                session,
                revision,
                choice,
            } => (session, Command::Apply(revision, choice)),
            Operation::RulesChance {
                session,
                revision,
                card,
            } => (session, Command::Chance(revision, card)),
            Operation::ImportReference { .. } => {
                return Reply::error(
                    id,
                    CoreError::Unavailable(
                        "replay-journal import; use state_import with a semantic DTO",
                    ),
                )
            }
            Operation::Search { .. } => {
                return Reply::error(id, CoreError::Unavailable("neural search (L08-L10)"))
            }
            Operation::Shutdown => {
                for active in self.active.iter().flatten() {
                    active.cancel();
                }
                self.owners.take();
                return Reply::new(id, Response::Shutdown);
            }
            Operation::Hello { .. } => unreachable!("handled above"),
        };
        let Some(owner) = owners.get(usize::from(session)) else {
            return Reply::error(id, CoreError::Invalid("session must be 0 or 1".into()));
        };
        if self.active[usize::from(session)].is_some() {
            return Reply::error(
                id,
                CoreError::Invalid("session busy until inference ticket is drained".into()),
            );
        }
        match owner.call(command) {
            Ok(response) => Reply::new(id, response),
            Err(error) => Reply::error(id, error),
        }
    }
}

fn serve(reader: &mut impl Read, writer: &mut impl Write) -> io::Result<()> {
    let mut server = Server::default();
    while let Some(payload) = read_frame(reader)? {
        let reply = match serde_json::from_slice::<Request>(&payload) {
            Ok(request) => server.handle(request),
            Err(error) => {
                // Echo a well-formed envelope identity even if a nested DTO was rejected.
                let id = serde_json::from_slice::<serde_json::Value>(&payload)
                    .ok()
                    .and_then(|v| {
                        v.get("request_id")
                            .and_then(|x| x.as_str())
                            .filter(|s| {
                                !s.is_empty() && s.len() <= 128 && !s.chars().any(char::is_control)
                            })
                            .map(str::to_owned)
                    });
                Reply::error(id, CoreError::Invalid(format!("request JSON: {error}")))
            }
        };
        let shutdown = matches!(reply.response, Response::Shutdown);
        let mut payload = serde_json::to_vec(&reply).map_err(io::Error::other)?;
        if payload.len() > brass_ai_core::framing::MAX_FRAME_BYTES {
            payload=serde_json::to_vec(&Reply::error(reply.request_id,CoreError::Invalid(
                "diagnostic response exceeds frame bound; use a smaller batch (no candidates were truncated)".into()
            ))).map_err(io::Error::other)?;
        }
        write_frame(writer, &payload)?;
        if shutdown {
            break;
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    if std::env::args_os().len() != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "worker accepts framed stdin, no CLI arguments",
        ));
    }
    serve(&mut io::stdin().lock(), &mut io::stdout().lock())
}

#[cfg(test)]
mod tests {
    use super::*;
    use brass_ai_core::CONTROL_PROTOCOL;
    use std::io::Cursor;

    fn request(operation: Operation) -> Request {
        Request {
            protocol: CONTROL_PROTOCOL.into(),
            request_id: "test".into(),
            request: operation,
        }
    }
    fn hello() -> Request {
        request(Operation::Hello {
            sessions: 2,
            threads_per_session: 1,
            required_capabilities: vec![],
        })
    }
    #[test]
    fn handshake_is_required_and_configuration_is_fixed() {
        let mut server = Server::default();
        assert!(matches!(
            server
                .handle(request(Operation::DiagnosticSnapshot { session: 0 }))
                .response,
            Response::Error { .. }
        ));
        assert!(matches!(
            server
                .handle(request(Operation::Hello {
                    sessions: 2,
                    threads_per_session: 2,
                    required_capabilities: vec![]
                }))
                .response,
            Response::Error { .. }
        ));
        assert!(matches!(
            server.handle(hello()).response,
            Response::Hello { .. }
        ));
        assert!(matches!(
            server.handle(hello()).response,
            Response::Error { .. }
        ));
    }
    #[test]
    fn panicked_owner_is_retired_other_owner_survives() {
        let failed = Owner::spawn(0).expect("owner");
        let healthy = Owner::spawn(1).expect("owner");
        assert!(matches!(
            failed.call(Command::Panic),
            Err(CoreError::SessionFailed)
        ));
        assert!(matches!(
            failed.call(Command::New {
                players: 4,
                seed: 1
            }),
            Err(CoreError::SessionFailed)
        ));
        assert!(matches!(
            healthy.call(Command::New {
                players: 4,
                seed: 1
            }),
            Ok(Response::Diagnostic { .. })
        ));
    }
    #[test]
    fn owner_exit_wakes_a_request_left_in_the_disconnected_queue() {
        let (sender, jobs) = bounded::<Job>(1);
        let (alive, exited) = bounded::<()>(0);
        let owner = Owner {
            sender: Some(sender),
            exited,
            join: None,
        };
        let (done, result) = bounded(1);
        let caller = thread::spawn(move || {
            let _ = done.send(owner.call(Command::Snapshot));
        });
        let limit = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while jobs.is_empty() {
            assert!(
                std::time::Instant::now() < limit,
                "request did not reach queue"
            );
            thread::yield_now();
        }
        // The owner exits without receiving this queued job. Crossbeam retains
        // the queued reply sender while the caller still owns the job sender.
        drop(jobs);
        drop(alive);
        assert!(matches!(
            result.recv_timeout(std::time::Duration::from_secs(5)),
            Ok(Err(CoreError::SessionFailed))
        ));
        caller.join().unwrap();
    }
    #[test]
    fn malformed_json_does_not_desynchronize_frames() {
        let mut input = Vec::new();
        write_frame(&mut input, b"{").unwrap();
        write_frame(&mut input, &serde_json::to_vec(&hello()).unwrap()).unwrap();
        write_frame(
            &mut input,
            &serde_json::to_vec(&request(Operation::Shutdown)).unwrap(),
        )
        .unwrap();
        let mut output = Vec::new();
        serve(&mut Cursor::new(input), &mut output).unwrap();
        let mut output = Cursor::new(output);
        for expected in ["error", "hello", "shutdown"] {
            let value: serde_json::Value =
                serde_json::from_slice(&read_frame(&mut output).unwrap().unwrap()).unwrap();
            assert_eq!(value["response"]["status"], expected);
        }
        assert!(read_frame(&mut output).unwrap().is_none());
    }
    #[test]
    fn response_failure_preserves_live_and_saved_states() {
        use brass_ai_core::types::Phase;
        let mut dto: StateDto = serde_json::from_str(include_str!(
            "../../brass-ai-core/tests/data/initial-state.json"
        ))
        .unwrap();
        // Structurally valid, deliberately unreachable: after this Finish there
        // would still be an action left, but no card with which to take it.
        dto.context.round = 2;
        dto.context.actions_remaining = 2;
        dto.context.phase = Phase::Continuation;
        dto.pending.kind = "pass".into();
        dto.pending.parts = 1;
        let actor = dto.context.actor as usize;
        dto.cards.deck.append(&mut dto.cards.hands[actor]);
        let state = NativeState::import(&dto).unwrap();
        let id = state.legal_choices().unwrap()[0].id.clone();
        let saved = state.snapshot();
        let observation = ObservationMemory::new(&state);
        let mut session = Session {
            state: Some(state),
            saved: Some(saved),
            saved_observation: Some(observation.clone()),
            observation,
            ..Session::default()
        };
        assert!(execute(
            &mut session,
            Command::Apply(dto.context.revision, id),
            0,
            "test"
        )
        .is_err());
        assert_eq!(session.state.as_ref().unwrap().export(), dto);
        execute(&mut session, Command::Restore, 0, "test").unwrap();
        assert_eq!(session.state.as_ref().unwrap().export(), dto);
    }
}
