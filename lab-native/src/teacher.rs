//! One Rust game/search lane. Only permitted tensors and final labels cross IPC.
//! Python retains the frozen-model GPU scheduler; no Python rules/tree callbacks.
use crate::{check, ordinary_after, scoring::Recorder, seal, Result};
use brass_ai_core::{
    features::{FeatureContext, FeatureEncoder},
    search::{
        self,
        brass::{Brass, Input, Key},
        math::{self, Statistics},
        reward::{Ledger, LedgerState, Reward},
        session::{
            clock::{Budget, Clock, Estimate, MonotonicClock},
            teacher::Evidence,
            Identity, Mode, Session, Settings, Startup, TurnKey,
        },
        Config, Draws, Game, Info, LeafEvaluator,
    },
    state::{
        dto::StateDto,
        observation::{Observation, Recall, RecallEntry},
        NativeState,
    },
    types::{Phase, Seat},
};
use serde_json::{json, Value};
use std::{
    io::{self, Read, Write},
    sync::{atomic::AtomicBool, Arc},
    time::Instant,
};

pub const CONTRACT: &str = "brass-lab-teacher/2.0.0";
const LIMIT: usize = 64 * 1024 * 1024;
const ALLOWANCE: u64 = 4_000_000_000;
const POST: u64 = 1_000_000_000;
pub const TIMING: &str = "search-plus-post/1.0.0";

/// Four seconds of search admission, plus at most one second of finalization.
/// Session::narrow retains both absolute cutoffs across action two/replanning.
pub fn search_budget(started_ns: u64) -> Result<Budget> {
    profile_budget(started_ns, false)
}
pub fn profile_budget(started_ns: u64, fast: bool) -> Result<Budget> {
    let (search, post) = if fast { (800_000_000, 500_000_000) } else { (ALLOWANCE, POST) };
    let deadline_ns = started_ns.checked_add(search + post).ok_or("turn clock overflow")?;
    Ok(Budget { started_ns, work_deadline_ns: deadline_ns - post, deadline_ns, reserve_ns: post })
}

/// Session::decide accepts an absolute deadline on the shared game clock.
/// Passing remaining time only happens to work near clock origin zero.
pub fn decide_turn<E: LeafEvaluator<Key, String>, C: Clock>(
    owner: &mut Session, identity: &Identity, request: u64, root: Input,
    mode: Mode, clock: &C, evaluator: &mut E,
) -> search::Result<search::session::Outcome> {
    let deadline = owner.budget().deadline_ns;
    owner.decide(identity, request, root, deadline, mode, clock, evaluator, None)
}

#[derive(Default)]
struct TurnWork {
    search_ns: u64,
    selection_ns: u64,
    labels_ns: u64,
    commit_ns: u64,
    rows: Vec<Value>,
    value_row: Option<Value>,
    fast: bool,
}

fn finish_turn<R: Read, W: Write>(peer: &mut Peer<R, W>, budget: Budget,
    work: &mut TurnWork, index: u64, diagnostic: bool) -> Result<Value> {
    let before = peer.clock.now_ns();
    let post_ns = before.saturating_sub(budget.started_ns).saturating_sub(work.search_ns);
    let post = budget.reserve_ns;
    check(work.search_ns <= budget.work_deadline_ns - budget.started_ns,
          "teacher search allowance exceeded")?;
    // Only search admission is capped. Native serialization/label handoff may
    // outlast the old reserve, just as independent Python verification may.
    let remaining_ns = post.saturating_sub(post_ns).min(budget.deadline_ns.saturating_sub(before));
    let row_count = work.rows.len();
    let mut message = json!({"event":"turn","job":peer.expected,"index":index,
        "timing":TIMING,"remaining_ns":remaining_ns,"rows":null,"diagnostic":diagnostic,
        "search_ns":work.search_ns,"native_post_ns":post_ns,
        "profile":if work.fast {"fast"} else {"full"},"value_row":null});
    // Move owned JSON trees rather than cloning whole board histories again.
    message["rows"] = Value::Array(std::mem::take(&mut work.rows));
    message["value_row"] = work.value_row.take().unwrap_or(Value::Null);
    emit(&mut peer.output, message)?;
    let ack: Value = serde_json::from_slice(&read_frame(&mut peer.input)?)?;
    let after = peer.clock.now_ns();
    check(ack["job"] == peer.expected && ack["index"] == index && ack["status"] == "ok",
          "stale/rejected turn verification")?;
    let total_post = after.saturating_sub(budget.started_ns).saturating_sub(work.search_ns);
    // The search and native selection budgets above are unchanged. Independent
    // Python verification may finish later; its ACK never admits search work.
    let mut report = json!({"index":index,"timing":TIMING,"search_ns":work.search_ns,"label_rows":row_count,
        "selection_ns":work.selection_ns,"labels_ns":work.labels_ns,"commit_ns":work.commit_ns,
        "native_post_ns":post_ns,"verification_limit_seconds":null,
        "post_search_ns":total_post,"elapsed_ns":after.saturating_sub(budget.started_ns),
        "budget":budget,"profile":if work.fast {"fast"} else {"full"}});
    if index == 1 || index % 64 == 0 {
        report["verification_ipc_ns"] = json!(after.saturating_sub(before));
    }
    *work = TurnWork::default();
    Ok(report)
}

fn read_frame(input: &mut impl Read) -> Result<Vec<u8>> {
    let mut size = [0; 4];
    input.read_exact(&mut size)?;
    let n = u32::from_le_bytes(size) as usize;
    check(n > 0 && n <= LIMIT, "teacher frame bound")?;
    let mut bytes = vec![0; n];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod verification_tests {
    use super::*;
    use std::{io::Cursor, thread, time::Duration};

    struct DelayedAck { bytes: Cursor<Vec<u8>>, delay: bool }
    impl Read for DelayedAck {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            if self.delay { self.delay = false; thread::sleep(Duration::from_millis(60)); }
            self.bytes.read(output)
        }
    }

    fn turn(fast: bool, ack: Value) -> Result<Value> {
        turn_case(fast,ack,false,0)
    }

    fn turn_case(fast: bool, ack: Value, late_handoff: bool, search_ns: u64) -> Result<Value> {
        let raw = serde_json::to_vec(&ack)?;
        let mut frame = (raw.len() as u32).to_le_bytes().to_vec(); frame.extend(raw);
        let mut peer = Peer {
            input: DelayedAck { bytes: Cursor::new(frame), delay: true }, output: Vec::<u8>::new(),
            encoder: FeatureEncoder::default(), clock: MonotonicClock::new(Arc::new(AtomicBool::new(false))),
            expected: json!({"job_id":"frozen"}), sequence:0, epoch:0, deadline:0,
            encoding_ns:0, wait_ns:0, requests:0, bytes:0, search_value:None,root_watch:None,root_stats:None,
        };
        // A deliberately tiny native handoff window, not a search benchmark.
        let budget = Budget { started_ns:0,work_deadline_ns:1,deadline_ns:50_000_001,reserve_ns:50_000_000 };
        let expected_rows = if fast {vec![]} else {vec![json!({"history":["row",1],"chosen":0})]};
        let mut work = TurnWork { fast,search_ns,rows:expected_rows.clone(), ..TurnWork::default() };
        if late_handoff { thread::sleep(Duration::from_millis(60)); }
        let result = finish_turn(&mut peer,budget,&mut work,1,false);
        if result.is_ok() {
            let payload = read_frame(&mut Cursor::new(peer.output))?;
            let size = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
            let message: Value = serde_json::from_slice(&payload[4..4+size])?;
            assert_eq!(message["rows"],json!(expected_rows));
            if late_handoff { assert_eq!(message["remaining_ns"],0); }
            assert!(work.rows.is_empty());
        }
        result
    }

    #[test]
    fn both_profiles_accept_late_verified_ack_without_changing_search_budget() {
        for fast in [false,true] {
            let report = turn(fast,json!({"job":{"job_id":"frozen"},"index":1,"status":"ok"})).unwrap();
            assert!(report["post_search_ns"].as_u64().unwrap() > 50_000_000);
            assert!(report["verification_limit_seconds"].is_null());
            assert_eq!(report["budget"]["work_deadline_ns"],1);
        }
    }

    #[test]
    fn late_ack_still_requires_exact_job_index_and_success() {
        for ack in [json!({"job":{"job_id":"wrong"},"index":1,"status":"ok"}),
                    json!({"job":{"job_id":"frozen"},"index":2,"status":"ok"}),
                    json!({"job":{"job_id":"frozen"},"index":1,"status":"cancelled"})] {
            assert!(turn(false,ack).is_err());
        }
    }

    #[test]
    fn both_profiles_accept_handoff_after_old_post_window_but_still_cap_search() {
        for fast in [false,true] {
            let ack = json!({"job":{"job_id":"frozen"},"index":1,"status":"ok"});
            let report = turn_case(fast,ack.clone(),true,0).unwrap();
            assert!(report["native_post_ns"].as_u64().unwrap() >= 50_000_000);
            assert_eq!(report["label_rows"],if fast {0} else {1});
            assert!(turn_case(fast,ack,false,2).is_err());
        }
    }

    #[test]
    fn single_buffer_emit_preserves_original_wire_bytes() {
        let history = vec![json!({"x":0.1234567890123456});64];
        let value = json!({"unicode":"黄", "history":history,"rows":[]});
        let data = serde_json::to_vec(&value).unwrap();
        let mut expected = ((data.len()+4) as u32).to_le_bytes().to_vec();
        expected.extend_from_slice(&(data.len() as u32).to_le_bytes());expected.extend(data);
        let mut actual = Vec::new();emit(&mut actual,value).unwrap();
        assert_eq!(actual,expected);
    }
}
fn write_frame(output: &mut impl Write, bytes: &[u8]) -> Result<()> {
    check(
        !bytes.is_empty() && bytes.len() <= LIMIT,
        "teacher output bound",
    )?;
    output.write_all(&(bytes.len() as u32).to_le_bytes())?;
    output.write_all(bytes)?;
    output.flush()?;
    Ok(())
}
fn emit(output: &mut impl Write, value: Value) -> Result<()> {
    let mut bytes = vec![0; 4];
    serde_json::to_writer(&mut bytes, &value)?;
    check(bytes.len() <= LIMIT, "teacher output bound")?;
    let metadata_len = (bytes.len() - 4) as u32;
    bytes[..4].copy_from_slice(&metadata_len.to_le_bytes());
    write_frame(output, &bytes)
}

struct Peer<R, W> {
    input: R,
    output: W,
    encoder: FeatureEncoder,
    clock: MonotonicClock,
    expected: Value,
    sequence: u64,
    epoch: u64,
    deadline: u64,
    encoding_ns: u64,
    wait_ns: u64,
    requests: u64,
    bytes: u64,
    search_value: Option<Value>,
    root_watch: Option<Info<Key, String>>,
    root_stats: Option<Statistics>,
}
impl<R: Read, W: Write> Peer<R, W> {
    fn evaluate_inner(&mut self, info: &Info<Key, String>) -> Result<Statistics> {
        check(self.clock.now_ns() < self.deadline, "evaluation deadline")?;
        self.sequence += 1;
        let start = Instant::now();
        let view = info
            .key
            .information
            .observation
            .view
            .as_ref()
            .ok_or("missing permitted view")?;
        self.encoder
            .encode(view, &info.key.choices, &info.key.context, &mut || Ok(()))?;
        let tensors = self.encoder.tensors().ok_or("missing tensors")?;
        self.encoding_ns += start.elapsed().as_nanos() as u64;
        let mut binding = json!({"job":self.expected,"sequence":self.sequence,"root_epoch":self.epoch,
            "revision":view.context.revision,"seat":info.actor,"legal_ids":info.actions,
            "context":info.key.context,"information":seal(&serde_json::to_value(info)?)});
        if let Some(value) = &self.search_value {
            binding["search_value"] = value.clone();
            binding["value_seat"] = json!(self.root_watch.as_ref().map_or(info.actor, |r| r.actor));
        }
        let meta = serde_json::to_vec(&json!({"event":"evaluate","binding":binding,
            "remaining_ns":self.deadline.saturating_sub(self.clock.now_ns()),
            "candidates":info.actions.len(),"state_len":tensors.state.len(),"action_len":tensors.actions.len()}))?;
        let mut bytes = (meta.len() as u32).to_le_bytes().to_vec();
        bytes.extend(meta);
        for slice in [&tensors.state[..], &tensors.actions[..]] {
            for value in slice {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        for value in tensors.stable_to_relative {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.bytes += bytes.len() as u64;
        self.requests += 1;
        let began = Instant::now();
        write_frame(&mut self.output, &bytes)?;
        let reply: Value = serde_json::from_slice(&read_frame(&mut self.input)?)?;
        self.wait_ns += began.elapsed().as_nanos() as u64;
        check(reply["binding"] == binding, "stale/model/epoch GPU reply")?;
        check(
            reply["status"] == "ok" && self.clock.now_ns() < self.deadline,
            "expired/cancelled GPU reply",
        )?;
        let logits: Vec<f64> = serde_json::from_value(reply["logits"].clone())?;
        let field = if let Some(value) = &self.search_value {
            check(reply["search_value"] == *value, "search value reply binding")?;
            "search_utility"
        } else { "utility" };
        let utility: [f64; 4] = serde_json::from_value(reply[field].clone())?;
        check(
            logits.len() == info.actions.len() && reply["reward_id"] == info.key.context.reward_id,
            "GPU legal width/reward differs",
        )?;
        let stats = Statistics::new(logits, utility)?;
        if self.root_watch.as_ref() == Some(info) { self.root_stats = Some(stats.clone()); }
        Ok(stats)
    }
}
impl<R: Read, W: Write> LeafEvaluator<Key, String> for Peer<R, W> {
    fn evaluate(&mut self, info: &Info<Key, String>) -> search::Result<Statistics> {
        self.evaluate_inner(info)
            .map_err(|e| search::invalid(e.to_string()))
    }
}

fn phase(view: &Observation) -> &'static str {
    use brass_ai_core::types::Era;
    match (view.context.era, view.context.round) {
        (Era::CanalEra, 1..=4) => "canal_early",
        (Era::CanalEra, _) => "canal_late",
        (_, 1..=6) => "rail_early",
        _ => "rail_late",
    }
}
pub fn label(
    e: &Evidence,
    actual: &Info<Key, String>,
    expected: &Value,
    played: &str,
    kind: &str,
    supported: bool,
    budget: Budget,
    now: u64,
) -> Result<Option<Value>> {
    check(e.info == *actual, "teacher successor identity differs")?;
    let view = actual
        .key
        .information
        .observation
        .view
        .as_ref()
        .ok_or("missing label view")?;
    let n: u64 = e.stats.count.iter().sum();
    if view.context.phase != Phase::Action || actual.actions.len() == 1 || n == 0 {
        return Ok(None);
    }
    check(
        n <= e.completed && actual.actions.iter().any(|a| a == played),
        "teacher watermark/played move",
    )?;
    let (_, mixed) = e.stats.completed_q(actual.actor)?;
    let target: Vec<f32> = e.policy()?.into_iter().map(|v| v as f32).collect();
    let means: Vec<f64> = e
        .stats
        .count
        .iter()
        .enumerate()
        .map(|(i, n)| {
            if *n == 0 {
                0.
            } else {
                e.stats.returns[i][actual.actor] / *n as f64
            }
        })
        .collect();
    let mut l = expected.clone();
    let fields = json!({"source":"search-v1-improved-policy/1.0.0","method":"search_v1",
        "algorithm":search::IDENTITY,"session":search::session::IDENTITY,"v205_enabled":false,
        "seat":actual.actor,"turn_key":[view.context.era,view.context.round,view.context.turn,view.context.turn_owner],
        "revision":view.context.revision,"phase":phase(view),"kind":if e.continuation {"continuation"} else {kind},
        "legal_ids":actual.actions,"logits":e.stats.logits.iter().map(|x|*x as f32).collect::<Vec<_>>(),
        "counts":e.stats.count,"mean_returns":means,"node_value":e.stats.value[actual.actor],"v_mix":mixed,
        "c1":50,"c2":0.1,"max_n":e.stats.count.iter().max(),"target":target,
        "requested_k":16,"effective_k":e.effective_k,"root_legal_count":e.root_legal_count,
        "frozen_n":e.frozen_n,"completed_trajectories":e.completed,"node_trajectories":n,"completed_rounds":e.rounds,
        "deadline_hit":now >= budget.work_deadline_ns,"overrun_seconds":now.saturating_sub(budget.deadline_ns) as f64 / 1e9,
        "played_action":played,"plan_supported":supported,"fallback_played":!supported,"cancelled":false,
        "gumbel_draw_id":seal(&json!(e.gumbels))});
    l.as_object_mut()
        .ok_or("invalid expected identity")?
        .extend(fields.as_object().unwrap().clone());
    Ok(Some(
        json!({"seat":actual.actor,"view":view,"choices":actual.key.choices,
        "style":[0.,0.,0.,0.],"intent":vec![0.;8],"teacher":l}),
    ))
}

pub fn run() -> Result<()> {
    let mut input = io::stdin().lock();
    let output = io::stdout().lock();
    let request: Value = serde_json::from_slice(&read_frame(&mut input)?)?;
    check(request["contract"] == CONTRACT, "teacher protocol")?;
    let job = &request["job"];
    let profile = &job["search"];
    check(
        job["kind"] == "teacher_phase"
            && job["ordinary_training"] == false
            && profile["method"] == "search_v1"
            && profile.as_object().is_some_and(|p| p.len() == 17)
            && profile["algorithm"] == search::IDENTITY
            && profile["session"] == search::session::IDENTITY
            && profile["gumbel_noise"] == true
            && profile["prior_temperature"] == 1.0
            && profile["v205_enabled"] == false
            && profile["k"] == 16
            && profile["sessions"] == 1
            && profile["threads_per_session"] == 1
            && profile["pending_paths"] == 1
            && profile["allowance_seconds"] == 4.0
            && profile["post_search_seconds"] == 1.0
            && profile["timing"] == TIMING
            && profile["c1"] == 50
            && profile["c2"] == 0.1
            && profile["watchdog_seconds"] == 30.0,
        "unsupported teacher profile",
    )?;
    let all_seat = job["seating"] == "all-seat-search";
    let h2h = job["h2h"] == true;
    check(!h2h || (all_seat && job["seat_values"].as_array().is_some_and(|v| v.len() == 4)), "head-to-head seat bindings")?;
    if all_seat {
        let mut fast = profile.clone();
        fast["k"] = json!(4); fast["allowance_seconds"] = json!(0.8); fast["post_search_seconds"] = json!(0.5);
        check(job["fast_search"] == fast, "fast profile differs")?;
        let w = job["search_value"]["w"].as_f64().ok_or("search blend")?;
        check(job["search_value"]["protocol"] == "search-utility/1.0.0" && w.is_finite() && (0.0..=1.0).contains(&w)
            && (w * 20. - (w * 20.).round()).abs() < 1e-8
            && job["search_value"]["value_source"] == if w == 0. { "main" } else { "corrected" }, "search value binding")?;
    }
    let expected = job["expected"].clone();
    check(
        job["readiness"]["ready"] == true && job["readiness"]["reward_id"] == expected["reward_id"],
        "teacher readiness deferred",
    )?;
    check(
        request["core_source"] == brass_ai_core::protocol::BuildIdentity::current().source_sha256,
        "teacher build differs",
    )?;
    let reward: Reward = serde_json::from_value(profile["reward"].clone())?;
    reward.validate()?;
    check(
        expected["reward_id"] == reward.identity()?,
        "teacher reward identity",
    )?;
    check(
        job["seats"]
            .as_array()
            .is_some_and(|a| a.len() == 4 && a.iter().all(|v| *v == expected["model_id"])),
        "teacher models differ",
    )?;
    let initial: StateDto = serde_json::from_value(request["origin"]["state"].clone())?;
    let rail: Vec<u32> = serde_json::from_value(request["origin"]["rail_keys"].clone())?;
    check(rail.len() == 64, "rail RNG keys")?;
    let mut state = NativeState::import(&initial)?;
    let mut recorder = Recorder::new(&initial);
    let clock = MonotonicClock::new(Arc::new(AtomicBool::new(false)));
    let mut peer = Peer {
        input,
        output,
        encoder: FeatureEncoder::default(),
        clock: clock.clone(),
        expected: expected.clone(),
        sequence: 0,
        epoch: 0,
        deadline: 0,
        encoding_ns: 0,
        wait_ns: 0,
        requests: 0,
        bytes: 0,
        search_value: if all_seat { Some(job["search_value"].clone()) } else { None },
        root_watch: None, root_stats: None,
    };
    let identity = Identity {
        game_instance: expected["game_id"].as_str().ok_or("game id")?.into(),
        generation: expected["generation"].as_u64().ok_or("generation")?,
        export_sha256: expected["export_hash"].as_str().ok_or("export")?.into(),
    };
    let mut ledger = Ledger::new(
        reward.clone(),
        LedgerState {
            spent: [0.; 4],
            settled: vec![],
        },
    )?;
    let mut recalls: [Recall; 4] = std::array::from_fn(|_| vec![]);
    let seed = job["sampling_seed"].as_u64().ok_or("sampling seed")?;
    let mut draws = Draws::new(seed, None, false);
    let mut session: Option<Session> = None;
    let mut turn = None;
    let mut first_action = true;
    let mut searches = Vec::new();
    let mut rows = 0;
    let mut labels = 0;
    let mut no_evidence = 0;
    let mut reserved = 0;
    let mut searched_turns = 0;
    let mut values = Vec::new();
    let mut request_id = 0;
    let mut turn_work = TurnWork::default();
    let mut turn_reports = Vec::new();
    let calibration = request["op"] == "calibrate";
    let probe = request["op"] == "turn_probe";
    check(calibration || probe || request["op"] == "run", "unknown teacher operation")?;
    let estimate = request["estimate_ns"].as_u64().unwrap_or(0);
    check(
        calibration || (1_000..=ALLOWANCE).contains(&estimate),
        "missing GPU trajectory calibration",
    )?;
    emit(
        &mut peer.output,
        json!({"event":"hello","core":brass_ai_core::protocol::BuildIdentity::current(),
        "contract":CONTRACT,"job":expected,"method":"search_v1","v205_enabled":false,
        "verification_policy":"sampled-unbounded-verification/2"}),
    )?;
    loop {
        if state.context().terminal {
            break;
        }
        check(
            rows < 5000,
            "teacher decision cap; incomplete game rejected",
        )?;
        let seat = state.context().actor as usize;
        let view = state.observation(Seat::try_from(seat as u8)?)?;
        ledger.observe(&view)?;
        if recalls[seat].last().is_none_or(|r| r.hand != view.own_hand) {
            recalls[seat].push(RecallEntry {
                revision: view.context.revision,
                hand: view.own_hand.clone(),
            });
        }
        let key = TurnKey::from_context(&view.context, seat);
        if turn.as_ref() != Some(&key) {
            if let Some(owner) = &session {
                if let Some(value) = &turn_work.value_row { values.push(value.clone()); }
                turn_reports.push(finish_turn(&mut peer, owner.budget(), &mut turn_work, searched_turns, probe)?);
                if probe {
                    emit(&mut peer.output,json!({"event":"turn_probe","job":expected,"diagnostic":true,
                        "metrics":{"turns":turn_reports,"searches":searches,"labels":labels,"reserved_turns":reserved}}))?;
                    return Ok(());
                }
            }
            session = None;
            turn = Some(key);
            first_action = true;
            request_id = 0;
        }
        let context = FeatureContext {
            model_id: expected["model_id"].as_str().ok_or("model")?.into(),
            reward_id: reward.identity()?,
            profile_id: "brass-lab-teacher/1.1.0".into(),
            style: [0.; 4],
            intent: [0.; 8],
            remaining: ledger.remaining(),
            max_cost: reward.insolvency.max_cost,
            stable_descriptors: None,
        };
        let root = Input {
            view: view.clone(),
            recall: recalls[seat].clone(),
            context,
            reward: reward.clone(),
            ledger: ledger.state(),
            shuffle_tape: None,
            sample_seed: seed.wrapping_add(rows),
        };
        let mut game = Brass::new(root.clone(), false)?;
        let actual = game.root()?;
        let scheduled = calibration || probe || (h2h && view.context.phase == Phase::Action)
            || (view.context.phase == Phase::Action
                && job["reservations"][seat.to_string()][phase(&view)].as_u64()
                    == Some(view.context.round as u64));
        let forced_fast = request["probe_profile"] == "fast";
        if session.is_none() && (scheduled || (all_seat && view.context.phase == Phase::Action)) {
            turn_work.fast = forced_fast || !scheduled;
            peer.root_watch = Some(actual.clone()); peer.root_stats = None;
            if h2h { peer.search_value = Some(job["seat_values"][seat].clone()); }
            let mut settings = Settings::default();
            settings.search = Config {
                k: if turn_work.fast { 4 } else { 16 },
                simulations: 8,
                seed: seed.wrapping_add(rows),
                ..Config::default()
            };
            let mut owner = Session::new(
                identity.clone(),
                &root,
                settings,
                profile_budget(clock.now_ns(), turn_work.fast)?,
            )?;
            owner.enable_teacher_exports()?;
            session = Some(owner);
            searched_turns += 1;
            if !turn_work.fast { reserved += 1; }
        }
        let mut evidence = None;
        let mut supported = false;
        let choice = if let Some(owner) = &mut session {
            request_id += 1;
            peer.epoch += 1;
            peer.deadline = owner.budget().work_deadline_ns;
            let began = clock.now_ns();
            let mode = if calibration {
                Mode::Count
            } else {
                Mode::Timed {
                    startup: Some(Startup {
                        model_id: root.context.model_id.clone(),
                        export_sha256: identity.export_sha256.clone(),
                        reward_id: root.context.reward_id.clone(),
                        phase: view.context.phase,
                        estimate: Estimate {
                            nanoseconds: estimate,
                            source: request["calibration_id"]
                                .as_str()
                                .ok_or("calibration id")?
                                .into(),
                        },
                    }),
                }
            };
            let out = decide_turn(
                owner,
                &identity,
                request_id,
                root.clone(),
                mode,
                &clock,
                &mut peer,
            )?;
            let work_end = if out.quota.is_some() {
                out.work_stopped_ns.min(out.budget.work_deadline_ns).max(began)
            } else { began }; // cached continuation/forced delivery performs no new search
            turn_work.search_ns += work_end.saturating_sub(began);
            turn_work.selection_ns += clock.now_ns().saturating_sub(work_end);
            evidence = owner.take_teacher_export();
            if let Some(quota) = &out.quota {
                searches.push(json!({"phase":phase(&view),"revision":view.context.revision,"seat":seat,
                    "frozen_n":quota.simulations,"effective_k":out.effective_k,"completed":out.completed,
                    "rounds":out.snapshot.as_ref().map_or(0,|s|s.completed_rounds),"partial":out.partial_completed,
                    "deadline_hit":out.status == "work_deadline","overrun_seconds":out.response_ready_ns.saturating_sub(out.budget.deadline_ns) as f64/1e9,
                    "elapsed_ns":clock.now_ns().saturating_sub(began),"budget":out.budget,"provenance":out.provenance,
                    "profile":if turn_work.fast {"fast"} else {"full"}}));
            }
            if calibration {
                check(
                    out.completed == 8,
                    "GPU calibration incomplete; no invented CPU estimate",
                )?;
                let played = out
                    .action_ids
                    .first()
                    .and_then(|a| a.first())
                    .unwrap_or(&actual.actions[0]);
                let probe = evidence
                    .as_ref()
                    .map(|e| {
                        label(
                            e,
                            &actual,
                            &expected,
                            played,
                            "root",
                            false,
                            out.budget,
                            clock.now_ns(),
                        )
                    })
                    .transpose()?
                    .flatten()
                    .map(|mut r| {
                        r["diagnostic"] = json!(true);
                        r
                    });
                emit(
                    &mut peer.output,
                    json!({"event":"calibration","completed":out.completed,
                    "elapsed_ns":clock.now_ns().saturating_sub(began),"searches":searches,
                    "requests":peer.requests,"wait_ns":peer.wait_ns,"encoding_ns":peer.encoding_ns,
                    "probe":probe}),
                )?;
                return Ok(());
            }
            supported = out.actions.first().is_some_and(|a| a.support >= 2);
            out.action_ids
                .first()
                .and_then(|a| a.first())
                .cloned()
                .unwrap_or_else(|| actual.actions[0].clone())
        } else if actual.actions.len() == 1 {
            actual.actions[0].clone()
        } else {
            peer.epoch += 1;
            peer.deadline = clock.now_ns() + 10_000_000_000;
            let stats = peer.evaluate(&actual)?;
            let p = math::softmax(&stats.logits)?;
            draws.categorical(&actual.actions.iter().cloned().zip(p).collect::<Vec<_>>())?
        };
        if all_seat && session.is_some() && view.context.phase == Phase::Action && first_action {
            let stats = if let Some(e) = &evidence { e.stats.clone() }
                else if let Some(stats) = &peer.root_stats { stats.clone() }
                else { peer.evaluate(&actual)? };
            let n: u64 = stats.count.iter().sum();
            let root_value = root_value(&stats)?;
            peer.encoder.encode(&view, &actual.key.choices, &root.context, &mut || Ok(()))?;
            let t = peer.encoder.tensors().ok_or("missing root tensor")?;
            turn_work.value_row = Some(json!({"job":expected,"index":searched_turns,"seat":seat,
                "generation":expected["generation"],"model_id":expected["model_id"],"reward_id":expected["reward_id"],
                "game_id":expected["game_id"],"split":job["split"],"phase":phase(&view),
                "profile":if turn_work.fast {"fast"} else {"full"},"search_value":peer.search_value,
                "state":t.state,"mapping":t.stable_to_relative,"remaining":ledger.remaining(),
                "spent":ledger.spent,"r":root_value,"v":stats.value,"n":n,
                "counts":stats.count,"returns":stats.returns,"revision":view.context.revision,
                "turn_key":[view.context.era,view.context.round,view.context.turn,view.context.turn_owner]}));
        }
        if session.is_some() && view.context.phase == Phase::Action && !turn_work.fast && !h2h && job["shadow_probe"] != true {
            let label_started = clock.now_ns();
            if let Some(e) = evidence {
                if let Some(mut row) = label(
                    &e,
                    &actual,
                    &expected,
                    &choice,
                    if first_action {
                        "root"
                    } else {
                        "replanned_root"
                    },
                    supported,
                    session.as_ref().unwrap().budget(),
                    clock.now_ns(),
                )? {
                    if all_seat { row["teacher"]["search_value"] = job["search_value"].clone(); }
                    if probe { row["diagnostic"] = json!(true); }
                    turn_work.rows.push(row);
                    labels += 1;
                } else {
                    no_evidence += 1;
                }
            } else {
                no_evidence += 1;
            }
            turn_work.labels_ns += clock.now_ns().saturating_sub(label_started);
        }
        if session.is_some() && view.context.phase == Phase::Action { first_action = false; }
        let commit_started = clock.now_ns();
        let selected = actual
            .key
            .choices
            .iter()
            .find(|c| c.id == choice)
            .ok_or("illegal recommendation")?;
        ledger.chosen(&view, selected);
        state = ordinary_after(&state, &choice, &rail, &mut recorder)?;
        if session.is_some() { turn_work.commit_ns += clock.now_ns().saturating_sub(commit_started); }
        rows += 1;
    }
    if let Some(owner) = &session {
        if let Some(value) = &turn_work.value_row { values.push(value.clone()); }
        turn_reports.push(finish_turn(&mut peer, owner.budget(), &mut turn_work, searched_turns, probe)?);
    }
    if probe {
        emit(&mut peer.output,json!({"event":"turn_probe","job":expected,"diagnostic":true,
            "metrics":{"turns":turn_reports,"searches":searches,"labels":labels,"reserved_turns":reserved}}))?;
        return Ok(());
    }
    // Terminal contexts use actor=255 (no player to act). The ledger consumes
    // public end-state fields; use a valid observer rather than that sentinel.
    ledger.observe(&state.observation(Seat::Zero)?)?;
    check(!all_seat || (searched_turns == 64 && values.len() == 64 && reserved == if h2h { 64 } else { 16 }), "all-seat turn coverage")?;
    emit(
        &mut peer.output,
        json!({"event":"finished","job":expected,"result":state.export().result,
        "value_rows":values,"final_spent":ledger.spent,
        "metrics":{"decisions":rows,"labels":labels,"no_evidence":no_evidence,"reserved_turns":reserved,"searched_turns":searched_turns,
            "searches":searches,"turns":turn_reports,"requests":peer.requests,"encoding_ns":peer.encoding_ns,"gpu_wait_ns":peer.wait_ns,
            "input_bytes":peer.bytes,"elapsed_ns":clock.now_ns()},"ordinary_training":false}),
    )?;
    Ok(())
}

/// Absolute four-seat mean: root prediction plus completed edge returns.
pub fn root_value(stats: &Statistics) -> Result<[f64; 4]> {
    check(stats.count.len() == stats.returns.len(), "root statistics width")?;
    let n: u64 = stats.count.iter().sum();
    let mut value = stats.value;
    for returns in &stats.returns {
        for seat in 0..4 { value[seat] += returns[seat]; }
    }
    for v in &mut value { *v /= 1. + n as f64; }
    check(value.iter().all(|v| v.is_finite()), "nonfinite root value")?;
    Ok(value)
}
