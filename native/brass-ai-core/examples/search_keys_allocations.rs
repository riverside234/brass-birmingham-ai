//! Isolated allocator attribution, not a trajectory throughput benchmark.
use brass_ai_core::{
    features::FeatureEncoder,
    search::{brass, math::Statistics, tree::Tree, Draws, Game},
};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{alloc::System, hint::black_box};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn measure(mut f: impl FnMut()) -> serde_json::Value {
    let region = Region::new(GLOBAL);
    for _ in 0..100 {
        f();
    }
    let s = region.change();
    serde_json::json!({"calls":100,"allocations":s.allocations,"reallocations":s.reallocations,"bytes_allocated":s.bytes_allocated,"bytes_deallocated":s.bytes_deallocated})
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args_os().nth(1).ok_or("case JSON required")?;
    let request: serde_json::Value = serde_json::from_slice(&std::fs::read(file)?)?;
    let mut input: brass::Input = serde_json::from_value(request["cases"][0]["input"].clone())?;
    input.shuffle_tape = None;
    let mut game = brass::Brass::new(input, false)?;
    let mut world = game.sample(&mut Draws::new(0, None, false))?;
    let info = game.root()?;
    let mut tree = Tree::new(2, 8192)?;
    tree.insert(
        info.clone(),
        Statistics::new(vec![0.; info.actions.len()], [0.; 4])?,
    )?;
    let mut encoder = FeatureEncoder::default();
    let k = &info.key;
    let view = k.information.observation.view.as_ref().ok_or("view")?;
    encoder.encode(view, &k.choices, &k.context, &mut || Ok(()))?;
    let key_clone = measure(|| {
        black_box(info.clone());
    });
    let deep_key_clone = measure(|| {
        black_box(brass_ai_core::search::Info {
            key: std::sync::Arc::new((*info.key).clone()),
            actor: info.actor,
            actions: std::sync::Arc::new((*info.actions).clone()),
        });
    });
    let information = measure(|| {
        black_box(game.inspect(&mut world).unwrap());
    });
    let lookup = measure(|| {
        black_box(tree.find(&info).unwrap());
    });
    let features = measure(|| {
        black_box(
            encoder
                .encode(view, &k.choices, &k.context, &mut || Ok(()))
                .unwrap(),
        );
    });
    println!(
        "{}",
        serde_json::json!({"candidate_count":info.actions.len(),"typed_key_clone":key_clone,"deep_key_clone_ablation":deep_key_clone,"actor_information_and_choices":information,"hash_lookup_with_semantic_equality":lookup,"warm_feature_encoder":features,"scope":"100 isolated calls; deep-copy ablation includes two Arc containers. Information includes public cost observation, permitted view, recall and canonical choices. No full-tree copy or serialization. Not a two-session throughput measurement."})
    );
    Ok(())
}
