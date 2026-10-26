//! Separate instrumented probe: two owners, no timing claims, excludes ORT's C allocator.
use brass_ai_core::{
    evaluation::EvaluationId,
    features::{FeatureContext, FeatureEncoder},
    state::{dto::StateDto, NativeState},
    types::Seat,
};
use brass_ai_inference::{
    manifest::{hash, Bundle},
    runtime::CpuEvaluator,
    Control, Evaluator, Input,
};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{
    alloc::System,
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc, Barrier},
    thread,
    time::{Duration, Instant},
};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let manifest = PathBuf::from(args.first().ok_or("manifest")?);
    let fixture = PathBuf::from(args.get(1).ok_or("fixture")?);
    let barrier = Arc::new(Barrier::new(3));
    let mut threads = Vec::new();
    // A failed setup aborts this diagnostic process instead of stranding a barrier.
    std::panic::set_hook(Box::new(|info| {
        eprintln!("allocation probe failed: {info}");
        std::process::exit(1);
    }));
    for owner in 0..2 {
        let manifest = manifest.clone();
        let fixture = fixture.clone();
        let barrier = barrier.clone();
        threads.push(thread::spawn(move || {
            let bundle =
                Bundle::load(&manifest, &hash(&std::fs::read(&manifest).unwrap())).unwrap();
            let m = bundle.manifest.clone();
            let mut runtime = CpuEvaluator::load(bundle).unwrap();
            let dto: StateDto = serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
            let state = NativeState::import(&dto).unwrap();
            let observer = dto.context.actor as u8;
            let view = state
                .observation(Seat::try_from(observer).unwrap())
                .unwrap();
            let choices = state.legal_choices().unwrap();
            let c = FeatureContext {
                model_id: m.model_id.clone(),
                reward_id: m.reward_id.clone(),
                profile_id: "allocation".into(),
                style: [0.; 4],
                intent: [0.; 8],
                remaining: [0.075; 4],
                max_cost: m.max_cost,
                stable_descriptors: None,
            };
            let ids: Vec<_> = (0..4)
                .map(|i| EvaluationId {
                    request_id: format!("{owner}-{i}"),
                    generation: 1,
                    model_id: m.model_id.clone(),
                    export_sha256: m.export_sha256.clone(),
                    reward_id: m.reward_id.clone(),
                    profile_id: c.profile_id.clone(),
                    revision: dto.context.revision,
                    observer,
                })
                .collect();
            let mut encoders: [FeatureEncoder; 4] =
                std::array::from_fn(|_| FeatureEncoder::default());
            for e in &mut encoders {
                e.encode(&view, &choices, &c, &mut || Ok(())).unwrap();
            }
            let inputs: Vec<_> = (0..4)
                .map(|i| Input {
                    id: &ids[i],
                    context: &c,
                    choices: &choices,
                    tensors: encoders[i].tensors().unwrap(),
                })
                .collect();
            let control = Control {
                deadline: Instant::now() + Duration::from_secs(120),
                cancelled: Arc::new(AtomicBool::new(false)),
            };
            for cap in [1, 2, 4] {
                runtime.evaluate(&inputs[..cap], &control).unwrap(); // warm every shape
                barrier.wait();
                barrier.wait();
                for _ in 0..100 {
                    std::hint::black_box(runtime.evaluate(&inputs[..cap], &control).unwrap());
                }
                barrier.wait();
                barrier.wait();
            }
        }));
    }
    let mut rows = Vec::new();
    for cap in [1, 2, 4] {
        barrier.wait();
        let region = Region::new(GLOBAL);
        barrier.wait();
        barrier.wait();
        let s = region.change();
        barrier.wait();
        rows.push(serde_json::json!({"cap":cap,"forwards":200,"positions":200*cap,"allocations":s.allocations,"reallocations":s.reallocations,
            "bytes_allocated":s.bytes_allocated,"bytes_deallocated":s.bytes_deallocated}));
    }
    for t in threads {
        t.join().expect("allocation owner");
    }
    println!(
        "{}",
        serde_json::json!({"sessions":2,"threads_per_session":1,"rows":rows,
        "scope":"Rust allocation counters only, warmed owned replies included; ONNX C allocations excluded. No performance timings from this instrumented run."})
    );
    Ok(())
}
