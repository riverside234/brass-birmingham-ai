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

fn read_frame(input: &mut impl Read) -> Result<Vec<u8>> {
    let mut size = [0; 4];
    input.read_exact(&mut size)?;
    let n = u32::from_le_bytes(size) as usize;
    check(n > 0 && n <= LIMIT, "teacher frame bound")?;
    let mut bytes = vec![0; n];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
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
    let data = serde_json::to_vec(&value)?;
    let mut bytes = (data.len() as u32).to_le_bytes().to_vec();
    bytes.extend(data);
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
    root_owner: usize,
    strategy: usize,
    intrinsic: Option<f64>,
    intrinsic_info: Option<Info<Key,String>>,
    objective_active: bool,
    intrinsic_requests: u64,
}
impl<R: Read, W: Write> Peer<R, W> {
    fn evaluate_inner(&mut self, info: &Info<Key, String>, want_intrinsic: bool) -> Result<Statistics> {
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
        let binding = json!({"job":self.expected,"sequence":self.sequence,"root_epoch":self.epoch,
            "revision":view.context.revision,"seat":info.actor,"legal_ids":info.actions,
            "context":info.key.context,"information":seal(&serde_json::to_value(info)?),
            "root_owner":self.root_owner,"strategy_id":self.strategy,"want_intrinsic":want_intrinsic});
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
        let utility: [f64; 4] = serde_json::from_value(reply["utility"].clone())?;
        check(
            logits.len() == info.actions.len() && reply["reward_id"] == info.key.context.reward_id,
            "GPU legal width/reward differs",
        )?;
        if want_intrinsic {
            check(info.actor==self.root_owner && self.strategy>0 && reply["evaluator_id"]==self.expected["bundle_id"],"intrinsic owner/bundle")?;
            let value=reply["intrinsic"].as_f64().ok_or("missing intrinsic reply")?;
            check(value.is_finite(),"nonfinite intrinsic reply")?;
            self.intrinsic=Some(value);self.intrinsic_requests+=1;
            self.intrinsic_info=Some(info.clone());
        }
        Ok(Statistics::new(logits, utility)?)
    }
}
impl<R: Read, W: Write> LeafEvaluator<Key, String> for Peer<R, W> {
    fn evaluate(&mut self, info: &Info<Key, String>) -> search::Result<Statistics> {
        self.evaluate_inner(info,self.objective_active && info.actor==self.root_owner)
            .map_err(|e| search::invalid(e.to_string()))
    }
    fn intrinsic(&mut self,info:&Info<Key,String>)->search::Result<f64> {
        if self.intrinsic_info.as_ref()==Some(info) {return Ok(self.intrinsic.expect("intrinsic cache"));}
        self.evaluate_inner(info,true).map_err(|e|search::invalid(e.to_string()))?;
        self.intrinsic.ok_or_else(||search::invalid("missing intrinsic scalar"))
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
    if let Some(d)=&e.stats.diversity {
        let assigned=&expected["seat_contexts"][actual.actor];
        let means=|s:&Vec<f64>|s.iter().zip(&e.stats.count).map(|(v,n)|if *n==0 {0.} else {*v / *n as f64}).collect::<Vec<_>>();
        let extra=json!({"source":"search-v1-domino-improved-policy/1.0.0","algorithm":search::domino::ALGORITHM,
            "session":search::domino::SESSION,"base_algorithm":search::IDENTITY,"base_session":search::session::IDENTITY,
            "strategy_id":assigned["strategy_id"],"seat_context":assigned,"root_owner":d.owner,
            "opponent_model":"root-latent-self-play","coefficient":d.coefficient,
            "task_node_value":d.task_value[d.owner],"intrinsic_node_value":d.intrinsic_value,
            "task_mean_returns":means(&d.task_returns),"intrinsic_mean_returns":means(&d.intrinsic_returns),
            "channel_counts":e.stats.count,"channel_watermark":e.completed,"intrinsic_prefix":actual.key.intrinsic_prefix,
            "normalization":"four-player-phase-actions-7-8-12-4/1.0.0","work":e.work});
        l.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    }
    Ok(Some(
        json!({"seat":actual.actor,"view":view,"choices":actual.key.choices,
        "style":actual.key.context.style,"intent":actual.key.context.intent,"teacher":l}),
    ))
}

pub fn run() -> Result<()> {
    let mut input = io::stdin().lock();
    let output = io::stdout().lock();
    let request: Value = serde_json::from_slice(&read_frame(&mut input)?)?;
    check(request["contract"] == CONTRACT, "teacher protocol")?;
    if request["op"]=="descriptor" {
        let mut view:Observation=if request.get("view").is_some() {serde_json::from_value(request["view"].clone())?} else {
            let dto:StateDto=serde_json::from_value(request["state"].clone())?;
            NativeState::import(&dto)?.observation(Seat::try_from(dto.context.actor as u8)?)?
        };
        if let Some(era)=request.get("era") {view.context.era=serde_json::from_value(era.clone())?;}
        if let Some(round)=request.get("round") {view.context.round=round.as_u64().ok_or("round")? as u8;}
        let choice=serde_json::from_value(request["choice"].clone())?;
        let slots:Vec<String>=serde_json::from_value(request["descriptor"]["slots"].clone())?;
        let regions:Vec<String>=serde_json::from_value(request["descriptor"]["regions"].clone())?;
        let (phase,x)=search::domino::event(&view,&choice,&slots,&regions)?;
        let mut output=output;
        emit(&mut output,json!({"event":"descriptor","phase":phase,"x":x,"view":view}))?;return Ok(());
    }
    let job = &request["job"];
    let profile = &job["search"];
    check(
        job["kind"] == "teacher_phase"
            && job["ordinary_training"] == false
            && profile["method"] == "search_v1"
            && profile.as_object().is_some_and(|p| p.len() == 15)
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
            && profile["c1"] == 50
            && profile["c2"] == 0.1
            && profile["watchdog_seconds"] == 30.0,
        "unsupported teacher profile",
    )?;
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
        root_owner: 0, strategy: 0, intrinsic: None, intrinsic_info: None, objective_active: false, intrinsic_requests: 0,
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
    let mut request_id = 0;
    let diverse=expected.get("bundle_id").is_some();
    let mut intrinsic_prefix=[0.;4];
    let mut phase_counts=[[0u32;4];4];
    let calibration = request["op"] == "calibrate";
    let benchmark = request["op"] == "benchmark";
    let context_control=request["control"]=="context_only";
    check(!context_control || calibration || benchmark,"diagnostic objective control in live collection")?;
    let estimate = request["estimate_ns"].as_u64().unwrap_or(0);
    check(
        calibration || (1_000..=ALLOWANCE).contains(&estimate),
        "missing GPU trajectory calibration",
    )?;
    emit(
        &mut peer.output,
        json!({"event":"hello","core":brass_ai_core::protocol::BuildIdentity::current(),
        "contract":CONTRACT,"job":expected,"method":"search_v1","v205_enabled":false}),
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
        peer.root_owner=seat;
        peer.strategy=if diverse {expected["seat_contexts"][seat]["strategy_id"].as_u64().ok_or("strategy")? as usize} else {0};
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
            session = None;
            turn = Some(key);
            first_action = true;
            request_id = 0;
        }
        let context = FeatureContext {
            model_id: expected["model_id"].as_str().ok_or("model")?.into(),
            reward_id: reward.identity()?,
            profile_id: if diverse {expected["bundle_id"].as_str().ok_or("bundle id")?.into()} else {CONTRACT.into()},
            style: if diverse {serde_json::from_value(expected["seat_contexts"][seat]["style"].clone())?} else {[0.;4]},
            intent: if diverse {serde_json::from_value(expected["seat_contexts"][seat]["intent"].clone())?} else {[0.;8]},
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
        let objective=if diverse {
            let b=&job["evaluator"];
            let o=search::domino::Objective{owner:seat,strategy:peer.strategy,
                coefficient:if context_control {1.} else {b["coefficients"][peer.strategy].as_f64().ok_or("coefficient")?},
                direction:serde_json::from_value(b["directions"][peer.strategy].clone())?,
                bundle_id:expected["bundle_id"].as_str().ok_or("bundle")?.into(),
                slots:serde_json::from_value(request["descriptor"]["slots"].clone())?,
                regions:serde_json::from_value(request["descriptor"]["regions"].clone())?,prefix:intrinsic_prefix[seat]};
            game.set_objective(o.clone())?;Some(o)
        } else {None};
        let actual = game.root()?;
        let scheduled = calibration || benchmark
            || (view.context.phase == Phase::Action
                && job["reservations"][seat.to_string()][phase(&view)].as_u64()
                    == Some(view.context.round as u64));
        if session.is_none() && scheduled {
            let mut settings = Settings::default();
            settings.search = Config {
                k: if calibration && request["continuation_probe"]==true {1} else {16},
                simulations: 8,
                seed: seed.wrapping_add(rows),
                ..Config::default()
            };
            let mut owner = Session::new(
                identity.clone(),
                &root,
                settings,
                Budget::new(clock.now_ns(), ALLOWANCE)?,
            )?;
            owner.enable_teacher_exports()?;
            if let Some(o)=&objective {owner.enable_diversity(o.clone())?;}
            session = Some(owner);
            reserved += 1;
        }
        let mut evidence = None;
        let mut supported = false;
        let choice = if let Some(owner) = &mut session {
            peer.objective_active=objective.as_ref().is_some_and(|o|o.active());
            if diverse {owner.set_intrinsic_prefix(intrinsic_prefix[seat])?;}
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
            let out = owner.decide(
                &identity,
                request_id,
                root.clone(),
                owner.budget().deadline_ns.saturating_sub(clock.now_ns()),
                mode,
                &clock,
                &mut peer,
                None,
            )?;
            evidence = owner.take_teacher_export();
            if let Some(quota) = &out.quota {
                searches.push(json!({"phase":phase(&view),"revision":view.context.revision,"seat":seat,
                    "frozen_n":quota.simulations,"effective_k":out.effective_k,"completed":out.completed,
                    "rounds":out.snapshot.as_ref().map_or(0,|s|s.completed_rounds),"partial":out.partial_completed,
                    "deadline_hit":out.status == "work_deadline","overrun_seconds":out.response_ready_ns.saturating_sub(out.budget.deadline_ns) as f64/1e9,
                    "elapsed_ns":clock.now_ns().saturating_sub(began),"budget":out.budget,"provenance":out.provenance,"work":out.teacher_work}));
            }
            if calibration || benchmark {
                check(
                    !calibration || out.completed == 8,
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
                    json!({"event":if benchmark {"benchmark"} else {"calibration"},"completed":out.completed,
                    "elapsed_ns":clock.now_ns().saturating_sub(began),"searches":searches,
                    "requests":peer.requests,"wait_ns":peer.wait_ns,"encoding_ns":peer.encoding_ns,
                    "intrinsic_requests":peer.intrinsic_requests,"work":out.teacher_work,"budget":out.budget,
                    "diagnostic":true,"control":request["control"],"continuation_probe":request["continuation_probe"],
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
            peer.objective_active=false;
            actual.actions[0].clone()
        } else {
            peer.objective_active=false;
            peer.epoch += 1;
            peer.deadline = clock.now_ns() + 10_000_000_000;
            let stats = peer.evaluate(&actual)?;
            let p = math::softmax(&stats.logits)?;
            draws.categorical(&actual.actions.iter().cloned().zip(p).collect::<Vec<_>>())?
        };
        if session.is_some() && view.context.phase == Phase::Action {
            if let Some(e) = evidence {
                if let Some(row) = label(
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
                    emit(&mut peer.output, json!({"event":"label","row":row}))?;
                    labels += 1;
                } else {
                    no_evidence += 1;
                }
            } else {
                no_evidence += 1;
            }
            first_action = false;
        }
        let selected = actual
            .key
            .choices
            .iter()
            .find(|c| c.id == choice)
            .ok_or("illegal recommendation")?;
        ledger.chosen(&view, selected);
        let next = ordinary_after(&state, &choice, &rail, &mut recorder)?;
        if let Some(o)=&objective {
            intrinsic_prefix[seat]+=o.reward(&view,selected)?;
        }
        if view.context.phase==Phase::Action {
            let p=match phase(&view) {"canal_early"=>0,"canal_late"=>1,"rail_early"=>2,_=>3};
            phase_counts[seat][p]+=1;
        }
        state=next;
        rows += 1;
    }
    if diverse {check(phase_counts.iter().all(|counts|*counts==[7,8,12,4]),"unsupported complete-game phase normalization")?;}
    emit(
        &mut peer.output,
        json!({"event":"finished","job":expected,"result":state.export().result,
        "metrics":{"decisions":rows,"labels":labels,"no_evidence":no_evidence,"reserved_turns":reserved,
            "searches":searches,"requests":peer.requests,"encoding_ns":peer.encoding_ns,"gpu_wait_ns":peer.wait_ns,
            "input_bytes":peer.bytes,"elapsed_ns":clock.now_ns(),"phase_action_counts":phase_counts,
            "intrinsic_requests":peer.intrinsic_requests},"ordinary_training":false}),
    )?;
    Ok(())
}
