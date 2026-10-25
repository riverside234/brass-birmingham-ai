//! Bounded control transport; exactly two independent native state owners.
use brass_ai_core::{
    diagnostic::DiagnosticGame,
    framing::{read_frame, write_frame},
    protocol::{Operation, Reply, Request, Response, CAPABILITIES},
    state::{dto::StateDto, NativeState, StateSnapshot},
    CoreError,
};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::{
    io::{self, Read, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    thread::{self, JoinHandle},
};

enum Command {
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
    #[cfg(test)]
    Panic,
}
type Job = (Command, Sender<Result<Response, CoreError>>);

#[derive(Default)]
struct Session {
    diagnostic: Option<DiagnosticGame>,
    state: Option<NativeState>,
    saved: Option<StateSnapshot>,
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
    let semantic = matches!(
        command,
        Command::Import(_)
            | Command::Export
            | Command::Save
            | Command::Restore
            | Command::StateCloneCheck
    );
    match command {
        Command::New { players, seed } => {
            let replacement = DiagnosticGame::new(players, seed)?;
            game.diagnostic = Some(replacement);
        }
        Command::Snapshot | Command::CloneCheck => {}
        Command::Import(dto) => {
            // Build and validate the entire replacement before touching live or saved state.
            let replacement = NativeState::import(&dto)?;
            game.state = Some(replacement);
        }
        Command::Export | Command::StateCloneCheck => {}
        Command::Save => {
            game.saved = Some(game.state.as_ref().ok_or(CoreError::NoGame)?.snapshot())
        }
        Command::Restore => {
            let saved = game
                .saved
                .as_ref()
                .ok_or_else(|| CoreError::Invalid("session has no saved state".into()))?;
            game.state.as_mut().ok_or(CoreError::NoGame)?.restore(saved);
        }
        #[cfg(test)]
        Command::Panic => panic!("test owner failure"),
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

#[derive(Default)]
struct Server {
    owners: Option<[Owner; 2]>,
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
                self.owners.take();
                return Reply::new(id, Response::Shutdown);
            }
            Operation::Hello { .. } => unreachable!("handled above"),
        };
        let Some(owner) = owners.get(usize::from(session)) else {
            return Reply::error(id, CoreError::Invalid("session must be 0 or 1".into()));
        };
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
        let payload = serde_json::to_vec(&reply).map_err(io::Error::other)?;
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
}
