//! B02 ordinary lane. Policy RNG stays in the versioned Python adapter;
//! authoritative progression, exact features and complete row buffers stay here.
pub mod scoring;
#[cfg(feature = "teacher")]
pub mod teacher;
use brass_ai_core::{
    features::{FeatureContext, FeatureEncoder, FeatureReply},
    state::{
        dto::{Choice, StateDto},
        observation::Observation,
        NativeState,
    },
    types::Seat,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const CONTRACT: &str = "brass-lab-collection/1.0.0";
pub const MAX_BYTES: usize = 256 * 1024 * 1024;
pub fn check(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
pub fn seal(v: &Value) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(v).expect("finite JSON"))
    )
}

pub struct Lane {
    state: NativeState,
    job: String,
    token: String,
    rail_keys: Vec<u32>,
    records: Vec<Value>,
    evidence: Vec<Value>,
    bytes: usize,
    cap: usize,
    recorder: scoring::Recorder,
    encoder: FeatureEncoder,
    encoded: Option<FeatureContext>,
    decision: Option<(Observation, Vec<Choice>)>,
    projection_time: Duration,
    feature_time: Duration,
    transition_time: Duration,
    max_candidates: usize,
}

impl Lane {
    pub fn new(
        d: StateDto,
        job: String,
        token: String,
        rail_keys: Vec<u32>,
        cap: usize,
    ) -> Result<Self> {
        check(
            !job.is_empty() && !token.is_empty() && (1..=5000).contains(&cap),
            "invalid lane identity/cap",
        )?;
        check(
            !d.context.chance && !d.context.terminal && rail_keys.len() == 64,
            "invalid ordinary origin",
        )?;
        let recorder = scoring::Recorder::new(&d);
        Ok(Self {
            state: NativeState::import(&d)?,
            job,
            token,
            rail_keys,
            cap,
            recorder,
            records: vec![],
            evidence: vec![],
            bytes: 0,
            encoder: FeatureEncoder::default(),
            encoded: None,
            decision: None,
            projection_time: Duration::ZERO,
            feature_time: Duration::ZERO,
            transition_time: Duration::ZERO,
            max_candidates: 0,
        })
    }

    fn current(&mut self) -> Result<&(Observation, Vec<Choice>)> {
        check(!self.state.context().terminal, "terminal decision")?;
        if self.decision.is_none() {
            let began = Instant::now();
            let seat = Seat::try_from(self.state.context().actor as u8)?;
            self.decision = Some((self.state.observation(seat)?, self.state.legal_choices()?));
            self.projection_time += began.elapsed();
            self.max_candidates = self
                .max_candidates
                .max(self.decision.as_ref().unwrap().1.len());
        }
        Ok(self.decision.as_ref().unwrap())
    }

    pub fn decision(&mut self) -> Result<Value> {
        if self.state.context().terminal {
            return Ok(json!({"terminal":true,"token":self.token}));
        }
        let (view, choices) = self.current()?;
        let result = json!({"terminal":false,"view":view,"choices":choices});
        Ok(json!({"token":self.token,"step":self.records.len(),"decision":result}))
    }

    pub fn features(
        &mut self,
        token: &str,
        revision: u32,
        context: FeatureContext,
        start: usize,
    ) -> Result<Value> {
        check(
            token == self.token && revision == self.state.context().revision,
            "stale encoding",
        )?;
        check(
            context.stable_descriptors.is_none(),
            "critic descriptors forbidden in ordinary actor inputs",
        )?;
        self.current()?;
        let (view, choices) = self.decision.as_ref().unwrap();
        if start == 0 {
            let began = Instant::now();
            self.encoder
                .encode(view, choices, &context, &mut || Ok(()))?;
            self.feature_time += began.elapsed();
            self.encoded = Some(context.clone());
        }
        check(
            self.encoded.as_ref() == Some(&context),
            "stale feature page/context",
        )?;
        Ok(serde_json::to_value(FeatureReply::page(
            self.encoder.tensors().ok_or("missing feature buffer")?,
            choices,
            &context,
            start,
            64,
        )?)?)
    }

    pub fn checkpoint(&self, token: &str, revision: u32) -> Result<Value> {
        check(
            token == self.token && revision == self.state.context().revision,
            "stale checkpoint",
        )?;
        check(
            format!("{:?}", self.state.context().phase) == "Action",
            "fork requires action",
        )?;
        let value = json!({"contract":CONTRACT,"job":self.job,"state":self.state.export(),"rail_keys":self.rail_keys});
        Ok(json!({"payload":value,"sha256":seal(&value)}))
    }

    pub fn fork(
        cp: &Value,
        job: String,
        token: String,
        cap: usize,
        indices: &[usize],
    ) -> Result<Self> {
        check(
            cp["sha256"].as_str() == Some(&seal(&cp["payload"])),
            "checkpoint seal differs",
        )?;
        let payload = &cp["payload"];
        check(
            payload["contract"] == CONTRACT && payload["job"] == job,
            "foreign fork job",
        )?;
        let mut d: StateDto = serde_json::from_value(payload["state"].clone())?;
        check(
            format!("{:?}", d.context.phase) == "Action",
            "fork requires action",
        )?;
        check(
            indices.len() == d.cards.deck.len().saturating_sub(1),
            "fork shuffle draws differ",
        )?;
        for (i, &j) in (1..d.cards.deck.len()).rev().zip(indices) {
            check(j <= i, "fork shuffle index out of bounds")?;
            d.cards.deck.swap(i, j);
        }
        Self::new(
            d,
            job,
            token,
            serde_json::from_value(payload["rail_keys"].clone())?,
            cap,
        )
    }

    pub fn append(
        &mut self,
        token: &str,
        revision: u32,
        mut row: Value,
        evidence: Value,
    ) -> Result<()> {
        check(
            token == self.token && revision == self.state.context().revision,
            "stale policy reply",
        )?;
        check(
            self.records.len() < self.cap,
            "decision cap reached; incomplete lane rejected",
        )?;
        let (view, choices) = self.current()?;
        check(
            row["view"] == serde_json::to_value(view)?
                && row["choices"] == serde_json::to_value(&choices)?,
            "policy mutated permitted decision",
        )?;
        check(
            row["seat"] == serde_json::to_value(view.seat)? && row["teacher"].is_null(),
            "ordinary seat/search mismatch",
        )?;
        let chosen = row["chosen"].as_u64().ok_or("invalid chosen index")? as usize;
        check(chosen < choices.len(), "illegal choice index")?;
        let p = row["behavior"]["probability"]
            .as_f64()
            .ok_or("missing behavior probability")?;
        check(
            p > 0. && p <= 1. && (choices.len() != 1 || p == 1.),
            "invalid behavior probability",
        )?;
        let id = choices[chosen].id.clone();
        let context = self
            .encoded
            .as_ref()
            .ok_or("row requires native features")?;
        check(
            row["style"] == json!(context.style) && row["intent"] == json!(context.intent),
            "row feature context changed",
        )?;
        row["step"] = json!(self.records.len());
        let size = serde_json::to_vec(&row)?.len() + serde_json::to_vec(&evidence)?.len();
        check(self.bytes + size <= MAX_BYTES, "lane buffer limit reached")?;
        // Stage every fallible operation before publishing the row or new state.
        let mut recorder = self.recorder.clone();
        let began = Instant::now();
        let next = ordinary_after(&self.state, &id, &self.rail_keys, &mut recorder)?;
        self.transition_time += began.elapsed();
        self.state = next;
        self.recorder = recorder;
        self.records.push(row);
        self.evidence.push(evidence);
        self.bytes += size;
        self.decision = None;
        self.encoded = None;
        Ok(())
    }

    pub fn row(&self, index: usize) -> Result<Value> {
        check(
            self.state.context().terminal,
            "incomplete lane cannot publish records",
        )?;
        check(index < self.records.len(), "row index out of bounds")?;
        // Preserve the exact bytes used by the complete-array seals. Python
        // checks these before decoding, avoiding cross-language float formatting.
        Ok(
            json!({"row_json":serde_json::to_string(&self.records[index])?,
                  "evidence_json":serde_json::to_string(&self.evidence[index])?}),
        )
    }
    pub fn finish(&self) -> Result<Value> {
        check(
            self.state.context().terminal && !self.records.is_empty(),
            "incomplete game cannot publish",
        )?;
        let d = self.state.export();
        Ok(
            json!({"contract":CONTRACT,"job":self.job,"token":self.token,"rows":self.records.len(),
            "records_sha256":seal(&json!(self.records)),"evidence_sha256":seal(&json!(self.evidence)),
            "cost":{"projection_seconds":self.projection_time.as_secs_f64(),"feature_seconds":self.feature_time.as_secs_f64(),
                "transition_seconds":self.transition_time.as_secs_f64(),"max_candidates":self.max_candidates,
                "peak_process_rss_bytes":peak_rss()},
            "buffer_bytes":self.bytes,"result":d.result,"scoring":self.recorder.finish(&d)?}),
        )
    }
    pub fn export(&self) -> StateDto {
        self.state.export()
    }
}

// Linux VmHWM includes native heap and allocator overhead. A missing platform
// counter is reported as null, never inferred from serialized row size.
fn peak_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let value = status.lines().find(|s| s.starts_with("VmHWM:"))?;
    value
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

/// Keep the reference hidden deck order. Search's explicit chance sampler sorts
/// the unseen remainder, so the ordinary adapter restores that remainder before
/// exposing the next decision. Chance draws do not consume ordinary revisions.
pub fn ordinary_after(
    state: &NativeState,
    choice: &str,
    keys: &[u32],
    recorder: &mut scoring::Recorder,
) -> Result<NativeState> {
    let revision = state
        .context()
        .revision
        .checked_add(1)
        .ok_or("revision overflow")?;
    let before = state.export();
    let mut next = state.after_choice(state.context().revision, choice)?;
    let mut scoring_before = before.clone();
    loop {
        let d = next.export();
        let Some(ch) = d.chance.as_ref() else { break };
        let rail = serde_json::to_value(ch.kind)? == "rail-shuffle";
        let picks = if rail {
            scoring_before = d.clone();
            let pool: Vec<String> = d
                .order
                .iter()
                .flat_map(|s| d.cards.discards[u8::from(*s) as usize].clone())
                .filter(|c| !c.starts_with("wild:"))
                .collect();
            check(
                d.cards.deck.is_empty() && pool.len() == keys.len(),
                "unsupported rail card/RNG inventory",
            )?;
            let mut keyed: Vec<_> = pool
                .into_iter()
                .zip(keys.iter().copied())
                .enumerate()
                .collect();
            keyed.sort_by_key(|(i, (_, key))| (*key, *i));
            keyed
                .into_iter()
                .map(|(_, (card, _))| card)
                .collect::<Vec<_>>()
        } else {
            d.cards
                .deck
                .iter()
                .take(ch.needed as usize)
                .cloned()
                .collect()
        };
        check(
            picks.len() == ch.needed as usize && ch.picked.is_empty(),
            "unexpected partial ordinary chance",
        )?;
        for card in &picks {
            next = next.after_chance(next.context().revision, card)?;
        }
        let mut after = next.export();
        if !rail {
            after.cards.deck = d.cards.deck[picks.len()..].to_vec();
        }
        after.context.revision = revision;
        next = NativeState::import(&after)?;
    }
    let mut after = next.export();
    after.context.revision = revision;
    recorder.boundary(&scoring_before, &after)?;
    Ok(NativeState::import(&after)?)
}
