//! Lossless ordinary transport only. Search/teacher/core inputs are unchanged.
use brass_lab_collector::{check, Result};
use serde::Deserialize;
use serde_json::{json, value::RawValue, Value};
use sha2::{Digest, Sha256};

#[derive(Default)]
pub struct Delta {
    view: Option<Value>,
    choices: Option<Value>,
    revision: Option<u64>,
    history: Vec<u8>,
}

#[derive(Deserialize)]
struct Row<'a> {
    #[serde(borrow)]
    view: View<'a>,
}
#[derive(Deserialize)]
struct View<'a> {
    #[serde(borrow)]
    history: &'a RawValue,
}

impl Delta {
    pub fn frame(&mut self, frame: &mut Value) -> Result<()> {
        if !frame["decision"]["view"].is_object() {
            return Ok(());
        }
        let view = &mut frame["decision"]["view"];
        let revision = view["context"]["revision"]
            .as_u64()
            .ok_or("missing delta revision")?;
        let original = view.clone();
        let history = view["history"].as_array().ok_or("missing history")?;
        let base = self
            .view
            .as_ref()
            .and_then(|v| v["history"].as_array())
            .map_or(0, |previous| {
                previous
                    .iter()
                    .zip(history)
                    .take_while(|(a, b)| a == b)
                    .count()
            });
        let patch = json!({"base":base,"previous_revision":self.revision,"length":history.len(),"tail":&history[base..]});
        let mut reused = vec![];
        if let Some(previous) = &self.view {
            for field in [
                "players",
                "slots",
                "links",
                "merchants",
                "order",
                "own_hand",
                "pending",
            ] {
                if view[field] == previous[field] {
                    view.as_object_mut().unwrap().remove(field);
                    reused.push(field);
                }
            }
        }
        view.as_object_mut().unwrap().remove("history");
        frame["history_delta"] = patch;
        frame["view_reuse"] = json!(reused);
        let choices = frame["decision"]["choices"].clone();
        if self.choices.as_ref() == Some(&choices) {
            frame["decision"].as_object_mut().unwrap().remove("choices");
            frame["choices_reuse"] = json!(true);
        }
        self.choices = Some(choices);
        self.view = Some(original);
        self.revision = Some(revision);
        Ok(())
    }

    pub fn reply(&mut self, data: &mut Value) -> Result<()> {
        if let Some(forced) = data.get_mut("forced").and_then(Value::as_array_mut) {
            for frame in forced {
                self.frame(frame)?;
            }
        }
        if let Some(current) = data.get_mut("current") {
            self.frame(current)?;
        } else {
            self.frame(data)?;
        }
        Ok(())
    }

    pub fn records(&mut self, mut meta: Value, raw: &[u8]) -> Result<(Value, Vec<u8>)> {
        let mut output = vec![];
        let mut lengths = vec![];
        let mut offset = 0;
        for item in meta["lengths"].as_array().ok_or("record lengths")? {
            let a = item[0].as_u64().ok_or("record length")? as usize;
            let b = item[1].as_u64().ok_or("evidence length")? as usize;
            check(
                a > 0 && b > 0 && offset + a + b <= raw.len(),
                "record delta bounds",
            )?;
            let row = &raw[offset..offset + a];
            let parsed: Row<'_> = serde_json::from_slice(row)?;
            let history = parsed.view.history.get().as_bytes();
            let prefix = history.as_ptr() as usize - row.as_ptr() as usize;
            check(prefix + history.len() <= row.len(), "record history bounds")?;
            let suffix = &row[prefix + history.len()..];
            let base = self
                .history
                .iter()
                .zip(history)
                .take_while(|(a, b)| a == b)
                .count();
            let tail = &history[base..];
            output.extend_from_slice(&row[..prefix]);
            output.extend_from_slice(tail);
            output.extend_from_slice(suffix);
            output.extend_from_slice(&raw[offset + a..offset + a + b]);
            lengths.push(json!([prefix, tail.len(), suffix.len(), b, base]));
            self.history = history.to_vec();
            offset += a + b;
        }
        check(offset == raw.len(), "record delta trailing bytes")?;
        meta["wire"] = json!("ordinary-records-delta/1");
        meta["lengths"] = json!(lengths);
        meta["bytes"] = json!(output.len());
        Ok((meta, output))
    }
}

pub fn reuse_training(mut meta: Value, raw: &[u8], reuse: &[Value]) -> Result<(Value, Vec<u8>)> {
    let lengths = meta["lengths"].as_array().ok_or("feature lengths")?;
    check(
        reuse.len() >= lengths.len() && reuse.len() <= 64,
        "reuse coverage",
    )?;
    let mut bytes = vec![];
    let mut sent = vec![];
    let mut seals = vec![];
    let mut offset = 0;
    for (length, expected) in lengths.iter().zip(reuse) {
        let size = length.as_u64().ok_or("feature size")? as usize;
        check(offset + size <= raw.len(), "feature reuse bounds")?;
        let value = &raw[offset..offset + size];
        if expected.is_null() {
            bytes.extend_from_slice(value);
            sent.push(size);
            seals.push(Value::Null);
        } else {
            let actual = format!("{:x}", Sha256::digest(value));
            check(
                expected.as_str() == Some(actual.as_str()),
                "retained feature differs",
            )?;
            sent.push(0);
            seals.push(json!(actual));
        }
        offset += size;
    }
    check(offset == raw.len(), "feature reuse trailing bytes")?;
    meta["sizes"] = meta["lengths"].clone();
    meta["lengths"] = json!(sent);
    meta["reuse_sha256"] = json!(seals);
    meta["bytes"] = json!(bytes.len());
    meta["wire"] = json!("ordinary-training-reuse/1");
    Ok((meta, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_delta_preserves_canonical_bytes_and_history_rewrites() {
        let mut codec = Delta::default();
        let mut history = vec![];
        for text in [
            r#"[{"event":"city café [cancel]"}]"#,
            r#"[{"event":"city café [cancel]"},{"n":2}]"#,
            "[]",
        ] {
            let row = format!(r#"{{"view":{{"history":{text}}},"chosen":0}}"#).into_bytes();
            let mut raw = row.clone();
            raw.extend_from_slice(b"{}");
            let (meta, delta) = codec
                .records(json!({"lengths":[[row.len(),2]]}), &raw)
                .unwrap();
            let l = meta["lengths"][0].as_array().unwrap();
            let v: Vec<usize> = l.iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let [a, b, c, d, base] = v[..] else { panic!() };
            history.truncate(base);
            history.extend_from_slice(&delta[a..a + b]);
            let restored = [
                &delta[..a],
                &history,
                &delta[a + b..a + b + c],
                &delta[a + b + c..a + b + c + d],
            ]
            .concat();
            assert_eq!(restored, raw);
        }
    }
    #[test]
    fn feature_reuse_omits_exact_bytes_and_rejects_wrong_seal() {
        let data = b"abcdefgh";
        let seal = format!("{:x}", Sha256::digest(data));
        let (meta, raw) = reuse_training(json!({"lengths":[8]}), data, &[json!(seal)]).unwrap();
        assert!(raw.is_empty());
        assert_eq!(meta["lengths"], json!([0]));
        assert_eq!(meta["sizes"], json!([8]));
        assert!(reuse_training(json!({"lengths":[8]}), data, &[json!("wrong")]).is_err());
    }
}
