//! Ordinary self-play played inside the CPU collector (build step 9).
//!
//! The Python adapter's v3 sampler, replayed inside the worker with the same
//! per-seat Mersenne Twister streams and the same float64 arithmetic: CPU
//! policy forward, opening temperature, family exploration and the ordinary
//! cotton recipes. Every row still goes through `append_compact`, so records,
//! features and seals are built exactly as for the Python loop. Python hands
//! over each seat's RNG state once per game and gets it back at the end. The
//! publisher replays the draws of audited seats against these rows as before,
//! so any divergence from the Python sampler fails publication.
use crate::{check, cpu_eval::Cache, Lane, Result, CONTRACT as COLLECTION};
use brass_ai_core::{
    features::{FeatureContext, FEATURE_VERSION},
    state::{
        dto::{Choice, Industry, Kind},
        observation::Observation,
    },
    types::{Era, Phase},
};
use serde_json::{json, Value};
use std::time::Instant;

pub const CONTRACT: &str = "ordinary-autoplay/1.0.0";
const COTTON_CAP: u32 = 3;

/// CPython's `random.Random`: MT19937 plus its `random()`, `getrandbits` and
/// `_randbelow`. The state is the 624 words and index of `getstate()[1]`.
#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; 624],
    index: usize,
}

impl PyRandom {
    pub fn from_state(value: &Value) -> Result<Self> {
        let words = value.as_array().filter(|a| a.len() == 625).ok_or("autoplay RNG state")?;
        let mut mt = [0u32; 624];
        for (slot, word) in mt.iter_mut().zip(words) {
            *slot = word.as_u64().filter(|x| *x <= u32::MAX as u64).ok_or("autoplay RNG word")? as u32;
        }
        let index = words[624].as_u64().filter(|x| *x <= 624).ok_or("autoplay RNG index")? as usize;
        Ok(Self { mt, index })
    }

    pub fn state(&self) -> Value {
        let mut words: Vec<u64> = self.mt.iter().map(|x| *x as u64).collect();
        words.push(self.index as u64);
        json!(words)
    }

    fn next_u32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        fn mag(y: u32) -> u32 {
            if y & 1 == 1 { 0x9908_b0df } else { 0 }
        }
        if self.index >= N {
            for k in 0..N - M {
                let y = (self.mt[k] & UPPER) | (self.mt[k + 1] & LOWER);
                self.mt[k] = self.mt[k + M] ^ (y >> 1) ^ mag(y);
            }
            for k in N - M..N - 1 {
                let y = (self.mt[k] & UPPER) | (self.mt[k + 1] & LOWER);
                self.mt[k] = self.mt[k + M - N] ^ (y >> 1) ^ mag(y);
            }
            let y = (self.mt[N - 1] & UPPER) | (self.mt[0] & LOWER);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag(y);
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.random()`: 53 bits from two draws.
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 { 0 } else { self.next_u32() >> (32 - k) }
    }

    /// `random._randbelow_with_getrandbits`, as used by `choice` and `randrange`.
    pub fn randbelow(&mut self, n: usize) -> Result<usize> {
        check(n > 0 && n <= u32::MAX as usize, "autoplay randbelow bound")?;
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k) as usize;
            if r < n {
                return Ok(r);
            }
        }
    }

    /// `random.choices(range(n), weights=w, k=1)[0]`.
    pub fn choices(&mut self, weights: &[f64]) -> Result<usize> {
        check(!weights.is_empty(), "autoplay empty weights")?;
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut total = weights[0];
        cumulative.push(total);
        for w in &weights[1..] {
            total += w;
            cumulative.push(total);
        }
        let total = total + 0.0;
        check(total > 0.0 && total.is_finite(), "autoplay weights must have a positive finite total")?;
        let x = self.random() * total;
        let (mut lo, mut hi) = (0usize, weights.len() - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if x < cumulative[mid] { hi = mid } else { lo = mid + 1 }
        }
        Ok(lo)
    }
}

/// NumPy's pairwise float64 summation (`ndarray.sum` on a contiguous vector).
pub fn npsum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.;
        for x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        npsum(&a[..half]) + npsum(&a[half..])
    }
}

/// The same summation for float32, as in `softmax32`.
fn npsum32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = 0f32;
        for x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        npsum32(&a[..half]) + npsum32(&a[half..])
    }
}

/// `lab_cpu_inference.softmax32` on the logits of a CPU reply, widened to float64.
fn softmax(bytes: &[u8], count: usize) -> Result<Vec<f64>> {
    check(bytes.len() == 4 * (count + 16), "autoplay CPU inference shape")?;
    let values: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    check(values.iter().all(|x| x.is_finite()), "Nonfinite CPU inference output")?;
    let logits = &values[..count];
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponent: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let total = npsum32(&exponent);
    Ok(exponent.iter().map(|x| (x / total) as f64).collect())
}

/// `step5.policy.INDUSTRIES` order, which indexes the intent vector.
fn intent_index(industry: Industry) -> usize {
    match industry {
        Industry::CottonMill => 0,
        Industry::Manufacturer => 1,
        Industry::Pottery => 2,
        Industry::Brewery => 3,
        Industry::IronWorks => 4,
        Industry::CoalMine => 5,
    }
}

/// `step5.policy.FAMILY_WEIGHTS`, default 1.
fn family_weight(kind: Kind) -> i64 {
    match kind {
        Kind::Build | Kind::Develop | Kind::Network => 4,
        Kind::Loan | Kind::Sell => 2,
        _ => 1,
    }
}

fn boosted(industry: Option<Industry>, intent: &[f64; 8]) -> f64 {
    match industry {
        Some(i) if intent[intent_index(i)] > 0.5 => 2.,
        _ => 1.,
    }
}

fn grouped<K: PartialEq + Copy>(keys: impl Iterator<Item = (usize, K)>) -> Vec<(K, Vec<usize>)> {
    let mut groups: Vec<(K, Vec<usize>)> = vec![];
    for (i, key) in keys {
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, members)) => members.push(i),
            None => groups.push((key, vec![i])),
        }
    }
    groups
}

fn industry_key(c: &Choice) -> Option<Industry> {
    if matches!(c.kind, Kind::Build | Kind::Develop) { c.industry } else { None }
}

/// `exploration_distribution(choices, intent, 3, phase)`.
pub fn exploration(choices: &[Choice], intent: &[f64; 8], phase: Phase) -> Result<Vec<f64>> {
    check(!choices.is_empty(), "No legal candidates")?;
    let mut out = vec![0.; choices.len()];
    if phase == Phase::Action {
        let mut families = grouped(choices.iter().enumerate().map(|(i, c)| (i, c.kind)));
        if !(families.len() == 1 && families[0].0 == Kind::Pass) {
            families.retain(|(k, _)| *k != Kind::Pass);
        }
        let total: i64 = families.iter().map(|(k, _)| family_weight(*k)).sum();
        for (kind, indices) in &families {
            let share = family_weight(*kind) as f64 / total as f64;
            let sub = grouped(indices.iter().map(|&i| (i, industry_key(&choices[i]))));
            let weights: Vec<f64> = sub.iter().map(|(industry, _)| boosted(*industry, intent)).collect();
            let sum = weights.iter().fold(0., |a, w| a + w);
            for ((_, members), w) in sub.iter().zip(&weights) {
                let value = share * w / sum / members.len() as f64;
                for &m in members {
                    out[m] = value;
                }
            }
        }
    } else {
        let groups = grouped(choices.iter().enumerate().map(|(i, c)| (i, (c.kind, industry_key(c)))));
        let weights: Vec<f64> = groups.iter().map(|((_, industry), _)| boosted(*industry, intent)).collect();
        let total = weights.iter().fold(0., |a, w| a + w);
        for ((_, members), w) in groups.iter().zip(&weights) {
            let value = w / total / members.len() as f64;
            for &m in members {
                out[m] = value;
            }
        }
    }
    Ok(out)
}

fn era_index(era: Era) -> usize {
    usize::from(era == Era::RailEra)
}

fn seat_of(view: &Observation) -> usize {
    u8::from(view.seat) as usize
}

fn active(phase: Phase) -> bool {
    matches!(phase, Phase::Action | Phase::Continuation)
}

fn cotton_build_available(view: &Observation, seat: usize, minimum: u8) -> bool {
    view.players[seat]
        .mat
        .iter()
        .find(|m| m.industry == Industry::CottonMill)
        .and_then(|m| m.tiles.first())
        .is_some_and(|t| t.level >= minimum)
}

fn has_unflipped_cotton(view: &Observation, seat: usize) -> bool {
    view.slots
        .iter()
        .filter_map(|s| s.tile.as_ref())
        .any(|t| u8::from(t.seat) as usize == seat && t.industry == Industry::CottonMill && !t.flipped)
}

fn cotton_candidates(choices: &[Choice], phase: Phase, builds: bool) -> Vec<usize> {
    if choices.len() <= 1 || !active(phase) {
        return vec![];
    }
    let with_builds = phase == Phase::Action && builds;
    choices
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.industry == Some(Industry::CottonMill) && (c.kind == Kind::Develop || with_builds && c.kind == Kind::Build)
        })
        .map(|(i, _)| i)
        .collect()
}

fn network_targets(view: &Observation) -> Vec<&str> {
    let cities: Vec<&str> = view
        .slots
        .iter()
        .filter(|s| s.tile.is_none() && s.types.contains(&Industry::CottonMill))
        .map(|s| s.id.split(':').next().unwrap_or(""))
        .collect();
    view.links
        .iter()
        .filter(|l| l.regions.iter().any(|r| cities.contains(&r.as_str())))
        .map(|l| l.id.as_str())
        .collect()
}

fn network_cotton_candidates(view: &Observation, choices: &[Choice], industry: bool, network: bool, develop: bool, minimum: u8) -> Vec<usize> {
    let phase = view.context.phase;
    if choices.len() <= 1 || !active(phase) {
        return vec![];
    }
    let builds = industry && phase == Phase::Action && cotton_build_available(view, seat_of(view), minimum);
    let links = if network && choices.iter().any(|c| c.kind == Kind::Network) { network_targets(view) } else { vec![] };
    choices
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            (c.kind == Kind::Network && c.target.as_deref().is_some_and(|t| links.contains(&t)))
                || (industry
                    && c.industry == Some(Industry::CottonMill)
                    && (develop && c.kind == Kind::Develop || builds && c.kind == Kind::Build))
        })
        .map(|(i, _)| i)
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Recipe {
    Plain,
    Limited,
    Network,
    RailNetwork,
    /// g3471: the g2654 recipe, timed within each era (cpu_cotton::timing).
    Timed,
    /// g3528: the g3471 rule retimed (cpu_cotton::retiming).
    Retimed,
}

/// `recovery.ordinary_cotton.TableExploration` for one table.
struct Cotton {
    seats: Vec<usize>,
    probability: f64,
    recipe: Recipe,
    used: [u32; 2],
    network_used: [u32; 2],
}

impl Cotton {
    fn parse(value: &Value) -> Result<Option<Self>> {
        if value.is_null() {
            return Ok(None);
        }
        let recipe = match value["contract"].as_str() {
            Some("ordinary-cotton-exploration/1.0.0") => Recipe::Plain,
            Some("ordinary-cotton-exploration/2.0.0") => Recipe::Limited,
            Some("ordinary-cotton-exploration/3.0.0") => Recipe::Network,
            Some("ordinary-cotton-exploration/4.0.0") => Recipe::RailNetwork,
            Some("ordinary-cotton-exploration/5.0.0") => Recipe::Timed,
            Some("ordinary-cotton-exploration/6.0.0") => Recipe::Retimed,
            _ => return Err("unknown ordinary cotton recipe".into()),
        };
        let seats: Vec<usize> = serde_json::from_value(value["seats"].clone())?;
        let probability = value["probability"].as_f64().ok_or("cotton probability")?;
        check(
            !seats.is_empty() && seats.iter().all(|s| *s < 4) && probability > 0. && probability <= 1.,
            "invalid ordinary cotton assignment",
        )?;
        Ok(Some(Self { seats, probability, recipe, used: [0; 2], network_used: [0; 2] }))
    }

    fn network(&self) -> bool {
        matches!(self.recipe, Recipe::Network | Recipe::RailNetwork | Recipe::Timed | Recipe::Retimed)
    }

    fn limited(&self) -> bool {
        self.recipe != Recipe::Plain
    }

    fn targets(&self, view: &Observation, choices: &[Choice]) -> Option<Vec<usize>> {
        if !self.network() {
            return None;
        }
        let era = era_index(view.context.era);
        let seat = seat_of(view);
        if !self.seats.contains(&seat) || has_unflipped_cotton(view, seat) {
            return Some(vec![]);
        }
        let minimum = if era == 1 && matches!(self.recipe, Recipe::RailNetwork | Recipe::Timed | Recipe::Retimed) { 3 } else { 2 };
        // g3471 / g3528: a quiet action explores no cotton; some actions exclude develop or network targets.
        let (timed, develop, network) = match self.recipe {
            Recipe::Timed => {
                let (timed, wide) = crate::cpu_cotton::timing(&view.context);
                (timed, wide, wide)
            }
            Recipe::Retimed => crate::cpu_cotton::retiming(&view.context),
            _ => (true, true, true),
        };
        if !timed {
            return Some(vec![]);
        }
        Some(network_cotton_candidates(view, choices, self.used[era] < COTTON_CAP, network && self.network_used[era] < 1, develop, minimum))
    }

    fn options(&self, view: &Observation, choices: &[Choice]) -> (f64, bool) {
        if self.network() || !self.limited() {
            return (self.probability, true);
        }
        let context = &view.context;
        if choices.len() <= 1 || !active(context.phase) || self.used[era_index(context.era)] >= COTTON_CAP {
            return (0., true);
        }
        let seat = seat_of(view);
        let builds = context.phase != Phase::Action || cotton_build_available(view, seat, 2);
        if !has_unflipped_cotton(view, seat) { (self.probability, builds) } else { (0., true) }
    }

    fn commit(&mut self, era: Era, cotton: bool, kind: Kind) {
        if self.limited() && cotton {
            let era = era_index(era);
            if self.network() && kind == Kind::Network {
                self.network_used[era] += 1;
            } else {
                self.used[era] += 1;
            }
        }
    }
}

/// One seat's `NeuralPolicy` (exploration version 3) and its RNG.
struct Policy {
    model: String,
    policy: String,
    exploration: f64,
    temperature: f64,
    opening: Option<(f64, u64)>,
    intent: [f64; 8],
    context: FeatureContext,
    rng: PyRandom,
    count: u64,
}

fn finite(value: &Value, name: &str) -> Result<f64> {
    value[name].as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("autoplay {name}").into())
}

impl Policy {
    fn parse(value: &Value) -> Result<Self> {
        let text = |name: &str| -> Result<String> {
            Ok(value[name].as_str().filter(|s| !s.is_empty()).ok_or_else(|| format!("autoplay {name}"))?.to_owned())
        };
        let (model, policy) = (text("model_id")?, text("policy_id")?);
        let exploration = finite(value, "exploration")?;
        let temperature = finite(value, "temperature")?;
        check((0. ..=0.5).contains(&exploration) && (0.1..=3.).contains(&temperature), "invalid behavior distribution")?;
        let opening = match &value["opening"] {
            Value::Null => None,
            o => {
                let t = finite(o, "temperature")?;
                let n = o["action_decisions_per_seat"].as_u64().ok_or("autoplay opening decisions")?;
                check(t > 0. && t <= 3., "invalid opening temperature")?;
                Some((t, n))
            }
        };
        let style: [f64; 4] = serde_json::from_value(value["style"].clone())?;
        let intent: [f64; 8] = serde_json::from_value(value["intent"].clone())?;
        let context = FeatureContext {
            model_id: model.clone(),
            reward_id: "ordinary-policy-only".into(),
            profile_id: COLLECTION.into(),
            style,
            intent,
            remaining: [0.; 4],
            max_cost: 0.,
            stable_descriptors: None,
        };
        context.validate()?;
        check(value["action_count"].as_u64() == Some(0), "autoplay starts a fresh game")?;
        Ok(Self { model, policy, exploration, temperature, opening, intent, context,
                  rng: PyRandom::from_state(&value["rng"])?, count: 0 })
    }

    /// `NeuralPolicy.sample_training`, with `raw` already normalised as `select_batch` does.
    fn sample(
        &mut self,
        choices: &[Choice],
        phase: Phase,
        raw: &[f64],
        cotton: Option<(f64, bool, Option<Vec<usize>>)>,
    ) -> Result<(usize, f64, Value)> {
        let count = self.count;
        let action = phase == Phase::Action;
        let temperature = match self.opening {
            Some((t, limit)) if action && count < limit => t,
            _ => self.temperature,
        };
        let exponent = 1. / temperature;
        let mut p: Vec<f64> = raw.iter().map(|x| x.max(1e-30).powf(exponent)).collect();
        let total = npsum(&p);
        for x in &mut p {
            *x /= total;
        }
        let q = exploration(choices, &self.intent, phase)?;
        let e = self.exploration;
        let mixed: Vec<f64> = p.iter().zip(&q).map(|(a, b)| (1. - e) * a + e * b).collect();
        let explore = self.rng.random() < e;
        let mut selected = self.rng.choices(if explore { &q } else { &p })?;
        let mut component = if explore { "exploration" } else { "policy" };
        let mut probability = mixed[selected];
        let decision = if action { json!(count) } else { Value::Null };
        let mut behavior = json!({"kind":"model","policy_id":self.policy,"temperature":temperature,
            "exploration":e,"exploration_version":3,"action_decision":decision});
        if let Some((rate, builds, targets)) = cotton {
            if rate != 0. {
                let candidates = match &targets {
                    Some(t) => t.clone(),
                    None => cotton_candidates(choices, phase, builds),
                };
                if !candidates.is_empty() {
                    let targeted = self.rng.random() < rate;
                    if targeted {
                        selected = candidates[self.rng.randbelow(candidates.len())?];
                    }
                    probability = (1. - rate) * mixed[selected];
                    if candidates.contains(&selected) {
                        probability += rate / candidates.len() as f64;
                    }
                    behavior["cotton_exploration"] = json!(rate);
                    if targets.is_some() {
                        behavior["cotton_candidates"] = json!(candidates);
                    } else if !builds {
                        behavior["cotton_develop_only"] = json!(true);
                    }
                    if targeted {
                        component = "cotton";
                    }
                }
            }
        }
        behavior["component"] = json!(component);
        behavior["probability"] = json!(probability);
        if action {
            self.count += 1;
        }
        Ok((selected, probability, behavior))
    }
}

impl Lane {
    /// Play a fresh game to its end with the given seats; Python exports it as usual.
    pub fn autoplay(&mut self, token: &str, request: &Value, cpu: &mut Cache) -> Result<Value> {
        check(request["contract"] == CONTRACT, "autoplay contract")?;
        check(
            token == self.token && self.pending.is_empty() && self.wire_rows.is_empty() && !self.state.context().terminal,
            "autoplay requires a fresh lane",
        )?;
        let seats = request["seats"].as_array().filter(|s| s.len() == 4).ok_or("autoplay needs four seats")?;
        let mut policies = seats.iter().map(Policy::parse).collect::<Result<Vec<_>>>()?;
        let mut cotton = Cotton::parse(&request["cotton"])?;
        let started = Instant::now();
        let (mut forced, mut inferences, mut forward_ns, mut policy_ns) = (0u64, 0u64, 0u64, 0u64);
        while !self.state.context().terminal {
            let began = Instant::now();
            let revision = self.state.context().revision;
            let actor = self.state.context().actor;
            let (seat, count) = {
                let (view, choices) = self.current()?;
                (seat_of(view), choices.len())
            };
            check(actor >= 0 && seat == actor as usize, "autoplay actor differs from observer")?;
            let raw = if count == 1 {
                forced += 1;
                vec![1.]
            } else {
                self.encode_features(token, revision, policies[seat].context.clone())?;
                let (bytes, ns) = cpu.infer(&policies[seat].model, self.encoder.tensors().ok_or("missing tensors")?)?;
                inferences += 1;
                forward_ns += ns;
                softmax(&bytes, count)?
            };
            let (view, choices) = self.decision.as_ref().ok_or("missing autoplay decision")?;
            let phase = view.context.phase;
            let rule = match &cotton {
                Some(c) if count > 1 && c.seats.contains(&seat) => {
                    let (rate, builds) = c.options(view, choices);
                    Some((rate, builds, c.targets(view, choices)))
                }
                _ => None,
            };
            let normalized: Vec<f64> = if count == 1 {
                raw.clone()
            } else {
                let total = npsum(&raw);
                raw.iter().map(|x| x / total).collect()
            };
            let p = &mut policies[seat];
            let (selected, _, behavior) = p.sample(choices, phase, &normalized, rule)?;
            let choice = &choices[selected];
            let (id, kind, era) = (choice.id.clone(), choice.kind, view.context.era);
            let targeted = behavior["component"] == "cotton";
            let seconds = began.elapsed();
            policy_ns += seconds.as_nanos() as u64;
            let record = json!({"revision":revision,"action_id":id,"context":&p.context,"seconds":seconds.as_secs_f64(),
                "behavior":behavior,"evidence":{"kind":"ordinary","model_id":p.model,"policy_id":p.policy,
                "forced":count == 1,"raw_probabilities":raw,"feature_version":FEATURE_VERSION,"semantic_id":id}});
            self.append_compact(token, &record)?;
            if let Some(c) = cotton.as_mut() {
                c.commit(era, targeted, kind);
            }
        }
        Ok(json!({"wire":CONTRACT,"token":self.token,"rows":self.wire_rows.len(),"forced_rows":forced,
            "inferences":inferences,"forward_ns":forward_ns,"policy_ns":policy_ns,
            "elapsed_ns":started.elapsed().as_nanos() as u64,
            "action_counts":policies.iter().map(|p| p.count).collect::<Vec<_>>(),
            "rng":policies.iter().map(|p| p.rng.state()).collect::<Vec<_>>(),
            "cotton":cotton.map(|c| json!({"used":c.used,"network_used":c.network_used}))}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPython `random.seed(int)`: init_by_array on the 32-bit words of the seed.
    fn seeded(seed: u32) -> PyRandom {
        let mut mt = [0u32; 624];
        mt[0] = 19_650_218;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        let key = [seed];
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        PyRandom { mt, index: 624 }
    }

    #[test]
    fn cpu_autoplay_rng_matches_cpython() {
        let mut r = seeded(12345);
        assert_eq!(r.mt[..3], [2147483648, 2105189241, 1699489545]);
        let draws = [r.random(), r.random(), r.random()];
        assert_eq!(draws, [0.41661987254534116, 0.010169169457068361, 0.8252065092537432]);
        assert_eq!(r.getrandbits(5), 9);
        assert_eq!(r.randbelow(7).unwrap(), 6);
        assert_eq!(r.randbelow(12).unwrap(), 5);
        let mut r = seeded(7);
        let w = [0.1, 0.2, 0.3, 0.25, 0.15];
        let picks: Vec<usize> = (0..12).map(|_| r.choices(&w).unwrap()).collect();
        assert_eq!(picks, [2, 1, 3, 0, 2, 2, 0, 2, 0, 2, 0, 0]);
        let mut r = seeded(99);
        let picks: Vec<usize> = (0..10).map(|_| [3, 8, 11][r.randbelow(3).unwrap()]).collect();
        assert_eq!(picks, [8, 8, 3, 11, 3, 3, 3, 3, 3, 8]);
        let back = PyRandom::from_state(&r.state()).unwrap();
        assert_eq!((back.mt, back.index), (r.mt, r.index));
    }

    fn choice(id: &str, kind: Kind, industry: Option<Industry>) -> Choice {
        Choice { id: id.into(), kind, card: None, target: None, industry, merchant: None, cards: None }
    }

    #[test]
    fn cpu_autoplay_exploration_matches_python() {
        use Industry::*;
        let choices = [
            choice("a", Kind::Build, Some(CottonMill)),
            choice("b", Kind::Build, Some(Pottery)),
            choice("c", Kind::Network, None),
            choice("d", Kind::Pass, None),
            choice("e", Kind::Loan, None),
            choice("f", Kind::Develop, Some(Pottery)),
            choice("g", Kind::Build, Some(Pottery)),
            choice("h", Kind::Scout, None),
        ];
        let intent = [0., 0., 1., 0., 0., 0., 0., 0.];
        assert_eq!(
            exploration(&choices, &intent, Phase::Action).unwrap(),
            [0.08888888888888889, 0.08888888888888889, 0.26666666666666666, 0.0, 0.13333333333333333,
             0.26666666666666666, 0.08888888888888889, 0.06666666666666667]
        );
        let other = exploration(&choices, &intent, Phase::Continuation).unwrap();
        assert_eq!(other[5], 0.2222222222222222);
        assert!(other.iter().enumerate().all(|(i, x)| i == 5 || *x == 0.1111111111111111));
        assert_eq!(exploration(&[choice("p", Kind::Pass, None)], &intent, Phase::Action).unwrap(), [1.0]);
    }

    #[test]
    fn cpu_autoplay_timed_cotton_is_quiet_at_era_edges_and_builds_only_at_their_ends() {
        use brass_ai_core::state::{dto::StateDto, NativeState};
        use brass_ai_core::types::Seat;
        let dto: StateDto = serde_json::from_str(include_str!("../../native/brass-ai-core/tests/data/initial-state.json")).unwrap();
        let state = NativeState::import(&dto).unwrap();
        let mut view = state.observation(Seat::try_from(state.context().actor as u8).unwrap()).unwrap();
        let seat = seat_of(&view);
        let mat = view.players[seat].mat.iter_mut().find(|m| m.industry == Industry::CottonMill).unwrap();
        while mat.tiles.first().is_some_and(|t| t.level < 3) { mat.tiles.remove(0); }
        let link = network_targets(&view)[0].to_string();
        let mut network = choice("n", Kind::Network, None);
        network.target = Some(link);
        let choices = [choice("b", Kind::Build, Some(Industry::CottonMill)), choice("d", Kind::Develop, Some(Industry::CottonMill)),
                       network, choice("p", Kind::Pass, None)];
        let parse = |contract: &str| Cotton::parse(&json!({"contract": contract, "seats": [seat], "probability": 0.2})).unwrap().unwrap();
        let (timed, plain) = (parse("ordinary-cotton-exploration/5.0.0"), parse("ordinary-cotton-exploration/4.0.0"));
        for (era, round, left, expected) in [
            (Era::CanalEra, 1, 1, vec![]), (Era::CanalEra, 4, 2, vec![0, 1, 2]), (Era::CanalEra, 8, 2, vec![0]),
            (Era::CanalEra, 8, 1, vec![]), (Era::RailEra, 1, 2, vec![]), (Era::RailEra, 1, 1, vec![]),
            (Era::RailEra, 2, 2, vec![0, 1, 2]), (Era::RailEra, 8, 2, vec![0]), (Era::RailEra, 8, 1, vec![]),
        ] {
            view.context.era = era;
            view.context.round = round;
            view.context.actions_remaining = left;
            view.context.phase = Phase::Action;
            assert_eq!(timed.targets(&view, &choices), Some(expected), "{era:?} {round} {left}");
            assert_eq!(plain.targets(&view, &choices), Some(vec![0, 1, 2]), "{era:?} {round} {left}");
        }
        // g3528 retiming: Canal 1 explores every target, Canal 14 builds and develops (no network); Rail as before.
        let retimed = parse("ordinary-cotton-exploration/6.0.0");
        for (era, round, left, expected) in [
            (Era::CanalEra, 1, 1, vec![0, 1, 2]), (Era::CanalEra, 4, 2, vec![0, 1, 2]), (Era::CanalEra, 8, 2, vec![0, 1]),
            (Era::CanalEra, 8, 1, vec![]), (Era::RailEra, 1, 2, vec![]), (Era::RailEra, 1, 1, vec![]),
            (Era::RailEra, 2, 2, vec![0, 1, 2]), (Era::RailEra, 8, 2, vec![0]), (Era::RailEra, 8, 1, vec![]),
        ] {
            view.context.era = era;
            view.context.round = round;
            view.context.actions_remaining = left;
            view.context.phase = Phase::Action;
            assert_eq!(retimed.targets(&view, &choices), Some(expected), "{era:?} {round} {left}");
        }
    }

    #[test]
    fn cpu_autoplay_pairwise_sum_orders_like_numpy() {
        // Below 8 terms a plain loop; up to 128 eight accumulators; above, halves.
        let a: Vec<f64> = (0..300).map(|i| 1. / (1. + i as f64).powi(2)).collect();
        for n in [1, 7, 8, 9, 64, 127, 128, 129, 300] {
            let mut naive = 0.;
            for x in &a[..n] {
                naive += x;
            }
            assert!((npsum(&a[..n]) - naive).abs() < 1e-14);
        }
        assert_eq!(npsum(&[0.1, 0.2, 0.3]), 0.1 + 0.2 + 0.3);
    }
}
