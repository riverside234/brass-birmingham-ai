//! Bounded ordinary-only transport. Never calls or changes search.
use super::*;
use brass_ai_core::features::{layout::META, ACTION_DIM, FEATURE_VERSION, FORMAT, STATE_DIM};

pub(crate) struct Forced {
    pub revision: u32,
    pub id: String,
    pub after: NativeState,
    pub recorder: scoring::Recorder,
}

impl Lane {
    pub fn public_context(&self) -> Result<Value> {
        Ok(serde_json::to_value(self.state.context())?)
    }
    pub fn feature_arrays(
        &mut self,
        token: &str,
        revision: u32,
        context: FeatureContext,
    ) -> Result<(Value, Vec<u8>)> {
        check(self.pending.is_empty(), "forced acknowledgements pending")?;
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
        let began = Instant::now();
        self.encoder
            .encode(view, choices, &context, &mut || Ok(()))?;
        self.feature_time += began.elapsed();
        self.encoded = Some(context.clone());
        let t = self.encoder.tensors().ok_or("missing tensors")?;
        check(t.legal_mask.iter().all(|x| *x), "ordinary candidate mask")?;
        let mut bytes = Vec::with_capacity((t.state.len() + t.actions.len()) * 4 + 32);
        for x in t.state.iter().chain(&t.actions) {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        for x in &t.stable_to_relative {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        Ok((
            json!({"wire":"ordinary-arrays/1", "token":token,"revision":revision,
            "format":FORMAT,"feature_version":FEATURE_VERSION,"schema_sha256":META.feature_schema_sha256,
            "group_layout":META.group_layout["identity"],"entity_layout":META.entity_layout["identity"],
            "state_dim":STATE_DIM,"action_dim":ACTION_DIM,"total_candidates":choices.len(),
            "choice_ids":choices.iter().map(|x| &x.id).collect::<Vec<_>>(),"context":context,
            "bytes":bytes.len()}),
            bytes,
        ))
    }

    pub fn append_compact(&mut self, token: &str, value: &Value) -> Result<()> {
        let revision = value["revision"]
            .as_u64()
            .filter(|x| *x <= u32::MAX as u64)
            .ok_or("invalid revision")? as u32;
        check(
            token == self.token && revision == self.state.context().revision,
            "stale compact reply",
        )?;
        let context: FeatureContext = serde_json::from_value(value["context"].clone())?;
        context.validate()?;
        check(
            context.reward_id == "ordinary-policy-only"
                && context.profile_id == CONTRACT
                && context.remaining == [0.; 4]
                && context.max_cost == 0.
                && context.stable_descriptors.is_none(),
            "ordinary context differs",
        )?;
        let id = value["action_id"].as_str().ok_or("missing action id")?;
        let (view, choices) = self.current()?;
        let chosen = choices
            .iter()
            .position(|x| x.id == id)
            .ok_or("illegal compact action")?;
        let forced = choices.len() == 1;
        let row = json!({"step":0,"seat":view.seat,"view":view,"choices":choices,"chosen":chosen,
            "teacher":null,"style":context.style,"intent":context.intent,
            "seconds":value["seconds"],"behavior":value["behavior"]});
        check(
            value["seconds"]
                .as_f64()
                .is_some_and(|x| x.is_finite() && x >= 0.),
            "invalid policy timing",
        )?;
        if forced {
            // No inference for a singleton. Admission still encodes and checks its
            // training features; do not ship unused actor tensors across the pipe.
            self.encoded = Some(context.clone());
        }
        check(
            self.encoded.as_ref() == Some(&context),
            "compact feature binding differs",
        )?;
        let evidence = &value["evidence"];
        check(
            evidence["kind"] == "excluded-fork"
                || (evidence["kind"] == "ordinary"
                    && evidence["model_id"] == context.model_id
                    && evidence["policy_id"] == row["behavior"]["policy_id"]
                    && evidence["semantic_id"] == id
                    && evidence["forced"] == forced
                    && evidence["feature_version"] == FEATURE_VERSION),
            "compact evidence binding differs",
        )?;
        self.append_row(token, revision, row, evidence.clone(), true)
    }

    /// Preview at most the reference collector's 16 forced stages. The live
    /// revision stays put until Python acknowledges every RNG/behavior record.
    pub fn preview_forced(&mut self, start: &Value, limit: usize) -> Result<Vec<Value>> {
        check(
            limit <= 16 && self.pending.is_empty(),
            "invalid forced batch",
        )?;
        let mut state = self.state.clone();
        let mut recorder = self.recorder.clone();
        let mut frames = vec![];
        let mut frame_bytes = 0;
        let began = Instant::now();
        for i in 0..limit.min(self.cap.saturating_sub(self.wire_rows.len())) {
            if state.context().terminal {
                break;
            }
            let context = serde_json::to_value(state.context())?;
            if context["actor"] != start["actor"]
                || context["phase"] == "debt"
                || ["era", "round", "turn", "turn_owner"]
                    .iter()
                    .any(|k| context[k] != start[k])
            {
                break;
            }
            let choices = state.legal_choices()?;
            if choices.len() != 1 {
                break;
            }
            let view = state.observation(Seat::try_from(state.context().actor as u8)?)?;
            let frame = json!({"token":self.token,"step":self.wire_rows.len()+i,
                "decision":{"terminal":false,"view":view,"choices":choices}});
            let size = serde_json::to_vec(&frame)?.len();
            if !frames.is_empty() && frame_bytes + size > 2 * 1024 * 1024 {
                break;
            }
            frame_bytes += size;
            frames.push(frame);
            let revision = state.context().revision;
            let id = choices[0].id.clone();
            state = ordinary_after(&state, &id, &self.rail_keys, &mut recorder)?;
            self.pending.push_back(Forced {
                revision,
                id,
                after: state.clone(),
                recorder: recorder.clone(),
            });
        }
        self.transition_time += began.elapsed();
        Ok(frames)
    }

    pub fn acknowledge_forced(&mut self, token: &str, values: &[Value]) -> Result<()> {
        check(
            !values.is_empty() && values.len() == self.pending.len(),
            "incomplete forced acknowledgement",
        )?;
        // Reject stale/foreign batches before touching any records.
        check(
            token == self.token
                && values
                    .iter()
                    .zip(&self.pending)
                    .all(|(v, p)| v["revision"] == p.revision && v["action_id"] == p.id),
            "stale forced batch",
        )?;
        for value in values {
            self.append_compact(token, value)?;
        }
        Ok(())
    }

    pub fn record_arrays(&self, token: &str, start: usize) -> Result<(Value, Vec<u8>)> {
        check(
            token == self.token && self.pending.is_empty() && self.state.context().terminal,
            "incomplete/foreign record export",
        )?;
        check(start < self.wire_rows.len(), "invalid record cursor")?;
        let mut bytes = vec![];
        let mut lengths = vec![];
        for (row, evidence) in self.wire_rows.iter().skip(start).take(64) {
            if !lengths.is_empty() && bytes.len() + row.len() + evidence.len() > 2 * 1024 * 1024 {
                break;
            }
            lengths.push([row.len(), evidence.len()]);
            bytes.extend_from_slice(row);
            bytes.extend_from_slice(evidence);
        }
        Ok((
            json!({"wire":"ordinary-records/1","token":token,"start":start,
            "end":start+lengths.len(),"rows":self.wire_rows.len(),"lengths":lengths,"bytes":bytes.len()}),
            bytes,
        ))
    }

    pub(super) fn records_seal(&self, evidence: bool) -> String {
        let mut hash = Sha256::new();
        hash.update(b"[");
        for (i, (row, ev)) in self.wire_rows.iter().enumerate() {
            if i > 0 {
                hash.update(b",");
            }
            hash.update(if evidence { ev } else { row });
        }
        hash.update(b"]");
        format!("{:x}", hash.finalize())
    }
}
