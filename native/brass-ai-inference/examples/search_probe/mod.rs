//! Two independent same-process owners; startup and one warm search excluded.
use brass_ai_core::search::{self, brass, Config};
use brass_ai_inference::{
    manifest::{hash, Bundle},
    runtime::CpuEvaluator,
    search::Neural,
    Control,
};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

pub fn run<T>(
    start: impl FnOnce() -> T,
    finish: impl FnOnce(T) -> Value,
) -> Result<Value, Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() < 2 || args.len() > 3 {
        return Err("usage: search_probe MANIFEST CASE.json [trace]".into());
    }
    let manifest = PathBuf::from(&args[0]);
    let digest = hash(&std::fs::read(&manifest)?);
    let case: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let case = &case["cases"][0];
    let input: brass::Input = serde_json::from_value(case["input"].clone())?;
    let mut config: Config = serde_json::from_value(case["config"].clone())?;
    config.trace = args.len() == 3;
    let tape: Vec<f64> = serde_json::from_value(case["draws"].clone())?;
    let ready = Arc::new(Barrier::new(3));
    let done = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for owner in 0..2 {
        let (manifest, digest, input, config, tape, ready, done) = (
            manifest.clone(),
            digest.clone(),
            input.clone(),
            config.clone(),
            tape.clone(),
            ready.clone(),
            done.clone(),
        );
        workers.push(thread::spawn(move||{
            // Always cross barriers on error, so the probe reports failure rather than hangs.
            let setup=(||->Result<_,brass_ai_core::CoreError>{
                let runtime=CpuEvaluator::load(Bundle::load(&manifest,&digest)?)?;
                Ok(runtime)
            })();
            let mut runtime=match setup {Ok(r)=>r,Err(e)=>{ready.wait();ready.wait();done.wait();ready.wait();return Err(e);}};
            let mut neural=Neural {evaluator:&mut runtime,encoder:Default::default(),control:Control {deadline:Instant::now()+Duration::from_secs(3600),cancelled:Arc::new(AtomicBool::new(false))},generation:owner+1,sequence:0,encoding_ns:0,inference_ns:0};
            let warm=(||{let mut game=brass::Brass::new(input.clone(),false)?;search::run(&mut game,&mut neural,config.clone(),Some(tape.clone()))})();
            if let Err(e)=warm {ready.wait();ready.wait();done.wait();ready.wait();return Err(e);}
            neural.encoding_ns=0;neural.inference_ns=0;
            let mut measurements=Vec::with_capacity(5);
            ready.wait();
            ready.wait();
            let begin=brass_ai_inference::process_clock_ns();
            let result=(||->Result<(),brass_ai_core::CoreError>{
                for _ in 0..5 {
                    let mut game=brass::Brass::new(input.clone(),false)?;
                    let result=search::run(&mut game,&mut neural,config.clone(),Some(tape.clone()))?;
                    game.finish()?;measurements.push((result.metrics,result.selected_action));
                }
                Ok(())
            })();
            let end=brass_ai_inference::process_clock_ns();
            done.wait();ready.wait();result?;
            Ok(serde_json::json!({"owner":owner,"thread":format!("{:?}",thread::current().id()),"started_ns":begin,"finished_ns":end,"runs":measurements,"encoding_ns":neural.encoding_ns,"inference_ns":neural.inference_ns}))
        }));
    }
    // Workers are parked at ready until instrumentation has started. The runtime
    // clock is initialized before both work intervals for comparable timestamps.
    brass_ai_inference::process_clock_ns();
    // Readiness, release and retirement phases bound the measurement while both
    // owners are alive. No startup or response serialization overlaps the region.
    ready.wait();
    let state = start();
    ready.wait();
    done.wait();
    let measured = finish(state);
    ready.wait();
    let results = workers
        .into_iter()
        .map(|w| {
            w.join()
                .map_err(|_| "probe owner panicked")?
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(
        serde_json::json!({"sessions":2,"threads_per_session":1,"trace":config.trace,"measurement":measured,"owners":results,"scope":"Five fixed-count searches per owner after model loading and one warm search; no turn deadline, merger or strength claim."}),
    )
}
