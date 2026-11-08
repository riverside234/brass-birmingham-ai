use brass_lab_collector::{check, Lane, Result, CONTRACT};
use serde_json::{json, Value};
use std::io::{self, Read, Write};
mod ordinary_delta;

const MAX_FRAME: usize = 4 * 1024 * 1024;
fn run() -> Result<()> {
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut lane: Option<Lane> = None;
    let mut previous = 0;
    let mut compact = false;
    let mut delta = ordinary_delta::Delta::default();
    let mut completion: Option<Value> = None;
    let mut export_cursors = (0usize, 0usize);
    loop {
        let mut size = [0u8; 4];
        match input.read(&mut size[..1])? {
            0 => return Ok(()),
            _ => input.read_exact(&mut size[1..])?,
        }
        let n = u32::from_le_bytes(size) as usize;
        check(n <= MAX_FRAME, "request frame too large")?;
        let mut bytes = vec![0; n];
        input.read_exact(&mut bytes)?;
        let r: Value = serde_json::from_slice(&bytes)?;
        let id = r["id"].as_u64().ok_or("missing request sequence")?;
        check(
            id == previous + 1 && r["protocol"] == CONTRACT,
            "stale/foreign envelope",
        )?;
        previous = id;
        let mut arrays = None;
        let result: Result<Value> = (|| {
            let token = r["token"].as_str().unwrap_or("");
            let revision = r["revision"].as_u64().unwrap_or(u64::MAX);
            match r["op"].as_str().unwrap_or("") {
                "hello" => Ok(
                    json!({"contract":CONTRACT,"core":brass_ai_core::protocol::BuildIdentity::current(),"search_enabled":false,
                        "collection_wire":"compact-arrays-forced/1", "training_arrays":"ordinary-training-arrays/1",
                        "performance_wire":"ordinary-collector-fast/1","export_pages":"ordinary-export-pages/1"}),
                ),
                "configure_wire" => {
                    check(lane.is_none(), "configure wire before game")?;
                    compact = r["enabled"].as_bool().ok_or("missing wire mode")?;
                    Ok(json!({"performance_wire":"ordinary-collector-fast/1","enabled":compact}))
                }
                "begin" | "fork" => {
                    check(lane.is_none(), "release prior lane before begin")?;
                    delta = ordinary_delta::Delta::default();
                    completion = None;
                    export_cursors = (0, 0);
                    let job = r["job"].as_str().ok_or("missing job")?.to_owned();
                    let cap = r["cap"].as_u64().ok_or("missing cap")? as usize;
                    lane = Some(if r["op"] == "begin" {
                        Lane::new(
                            serde_json::from_value(r["state"].clone())?,
                            job,
                            token.into(),
                            serde_json::from_value(r["rail_keys"].clone())?,
                            cap,
                        )?
                    } else {
                        Lane::fork(
                            &r["checkpoint"],
                            job,
                            token.into(),
                            cap,
                            &serde_json::from_value::<Vec<usize>>(r["indices"].clone())?,
                        )?
                    });
                    lane.as_mut().unwrap().decision()
                }
                "decision" => lane.as_mut().ok_or("missing lane")?.decision(),
                "features" => {
                    check(revision <= u32::MAX as u64, "invalid revision")?;
                    lane.as_mut().ok_or("missing lane")?.features(
                        token,
                        revision as u32,
                        serde_json::from_value(r["context"].clone())?,
                        r["start"].as_u64().ok_or("missing offset")? as usize,
                    )
                }
                "append" => {
                    check(revision <= u32::MAX as u64, "invalid revision")?;
                    let l = lane.as_mut().ok_or("missing lane")?;
                    l.append(
                        token,
                        revision as u32,
                        r["row"].clone(),
                        r["evidence"].clone(),
                    )?;
                    l.decision()
                }
                "feature_arrays" => {
                    check(revision <= u32::MAX as u64, "invalid revision")?;
                    let (meta, bytes) = lane.as_mut().ok_or("missing lane")?.feature_arrays(
                        token,
                        revision as u32,
                        serde_json::from_value(r["context"].clone())?,
                    )?;
                    arrays = Some(vec![bytes]);
                    Ok(meta)
                }
                "append_compact" => {
                    let l = lane.as_mut().ok_or("missing lane")?;
                    let start = l.public_context()?;
                    l.append_compact(token, &r["record"])?;
                    let forced = l.preview_forced(
                        &start,
                        r["forced_limit"].as_u64().ok_or("missing forced limit")? as usize,
                    )?;
                    let current = if forced.is_empty() {
                        l.decision()?
                    } else {
                        Value::Null
                    };
                    Ok(json!({"current":current,"forced":forced}))
                }
                "append_forced" => {
                    let l = lane.as_mut().ok_or("missing lane")?;
                    l.acknowledge_forced(
                        token,
                        r["records"].as_array().ok_or("missing forced records")?,
                    )?;
                    l.decision()
                }
                "record_arrays" => {
                    let (meta, bytes) = lane.as_ref().ok_or("missing lane")?.record_arrays(
                        token,
                        r["start"].as_u64().ok_or("missing record cursor")? as usize,
                    )?;
                    arrays = Some(vec![bytes]);
                    Ok(meta)
                }
                "training_arrays" => {
                    let (meta, bytes) = lane.as_ref().ok_or("missing lane")?.training_arrays(
                        token,
                        r["start"].as_u64().ok_or("missing training cursor")? as usize,
                    )?;
                    arrays = Some(vec![bytes]);
                    Ok(meta)
                }
                "drain_records" | "drain_training" => {
                    let l = lane.as_mut().ok_or("missing lane")?;
                    let start = r["start"].as_u64().ok_or("missing drain cursor")? as usize;
                    let (meta, bytes) = if r["op"] == "drain_records" {
                        l.drain_records(token, start)?
                    } else {
                        l.drain_training(token, start)?
                    };
                    arrays = Some(vec![bytes]);
                    Ok(meta)
                }
                "drain_records_compact" | "drain_training_reuse" => {
                    check(compact, "performance wire not enabled")?;
                    let l = lane.as_mut().ok_or("missing lane")?;
                    let start = r["start"].as_u64().ok_or("missing drain cursor")? as usize;
                    let (meta, bytes) = if r["op"] == "drain_records_compact" {
                        let (meta, bytes) = l.drain_records(token, start)?;
                        delta.records(meta, &bytes)?
                    } else {
                        let (meta, bytes) = l.drain_training(token, start)?;
                        ordinary_delta::reuse_training(
                            meta,
                            &bytes,
                            r["reuse"].as_array().ok_or("missing reuse")?,
                        )?
                    };
                    arrays = Some(vec![bytes]);
                    Ok(meta)
                }
                "export_page" => {
                    check(compact, "performance wire not enabled")?;
                    let records =
                        r["record_start"].as_u64().ok_or("missing record cursor")? as usize;
                    let features = r["feature_start"]
                        .as_u64()
                        .ok_or("missing feature cursor")?
                        as usize;
                    let l = lane.as_mut().ok_or("missing lane")?;
                    let first = completion.is_none();
                    check(
                        (records, features) == export_cursors,
                        "stale export cursors",
                    )?;
                    if first {
                        check(records == 0 && features == 0, "first export cursors differ")?;
                        completion = Some(l.finish()?);
                    }
                    let receipt = completion.as_ref().unwrap();
                    check(receipt["token"] == token, "foreign export token")?;
                    let rows = receipt["rows"].as_u64().ok_or("missing completed rows")? as usize;
                    check(
                        records <= rows && features <= rows && (records < rows || features < rows),
                        "export cursor bounds",
                    )?;
                    let mut payloads = vec![];
                    let mut record_meta = Value::Null;
                    let mut feature_meta = Value::Null;
                    let mut record_end = records;
                    let mut feature_end = features;
                    if records < rows {
                        let (meta, raw) = l.drain_records(token, records)?;
                        let (meta, raw) = delta.records(meta, &raw)?;
                        record_end = meta["end"].as_u64().ok_or("record export end")? as usize;
                        record_meta = meta;
                        check(
                            raw.len() <= 16 * 1024 * 1024,
                            "record export section too large",
                        )?;
                        payloads.push(raw);
                    }
                    if features < rows {
                        let (meta, raw) = l.drain_training(token, features)?;
                        let (meta, raw) = ordinary_delta::reuse_training(
                            meta,
                            &raw,
                            r["reuse"].as_array().ok_or("missing reuse")?,
                        )?;
                        feature_end = meta["end"].as_u64().ok_or("feature export end")? as usize;
                        feature_meta = meta;
                        check(
                            raw.len() <= 16 * 1024 * 1024,
                            "feature export section too large",
                        )?;
                        payloads.push(raw);
                    }
                    let done = record_end == rows && feature_end == rows;
                    export_cursors = (record_end, feature_end);
                    let data = json!({"wire":"ordinary-export-pages/1","token":token,
                        "receipt":if first {receipt.clone()} else {Value::Null},"record":record_meta,"training":feature_meta,
                        "done":done,"bytes":payloads.iter().map(Vec::len).sum::<usize>()});
                    arrays = Some(payloads);
                    if done {
                        lane = None;
                        completion = None;
                    }
                    Ok(data)
                }
                "checkpoint" => {
                    check(revision <= u32::MAX as u64, "invalid revision")?;
                    lane.as_ref()
                        .ok_or("missing lane")?
                        .checkpoint(token, revision as u32)
                }
                "export" => Ok(serde_json::to_value(
                    lane.as_ref().ok_or("missing lane")?.export(),
                )?),
                "finish" => lane.as_ref().ok_or("missing lane")?.finish(),
                "row" => lane
                    .as_ref()
                    .ok_or("missing lane")?
                    .row(r["index"].as_u64().ok_or("missing row index")? as usize),
                "release" => {
                    lane = None;
                    completion = None;
                    Ok(json!({"released":true}))
                }
                _ => Err("unsupported collector operation".into()),
            }
        })();
        let result = result.and_then(|mut data| {
            if compact {
                delta.reply(&mut data)?;
            }
            Ok(data)
        });
        let failed = result.is_err();
        let reply = match result {
            Ok(data) => json!({"protocol":CONTRACT,"id":id,"ok":true,"data":data}),
            Err(e) => json!({"protocol":CONTRACT,"id":id,"ok":false,"error":e.to_string()}),
        };
        let bytes = serde_json::to_vec(&reply)?;
        check(bytes.len() <= MAX_FRAME, "response frame too large")?;
        if let Some(parts) = arrays {
            let length = 4 + bytes.len() + parts.iter().map(Vec::len).sum::<usize>();
            // The new envelope contains two independently bounded old pages.
            // Keep all old operations at their original 16-MiB limit.
            let limit = if r["op"] == "export_page" {
                36 * 1024 * 1024
            } else {
                16 * 1024 * 1024
            };
            check(length <= limit, "array response too large")?;
            output.write_all(&((length as u32) | (1 << 31)).to_le_bytes())?;
            output.write_all(&(bytes.len() as u32).to_le_bytes())?;
            output.write_all(&bytes)?;
            for raw in parts {
                output.write_all(&raw)?;
            }
        } else {
            output.write_all(&(bytes.len() as u32).to_le_bytes())?;
            output.write_all(&bytes)?;
        }
        output.flush()?;
        // Rejected input retires the process; no partially failed lane can resume.
        if failed {
            return Ok(());
        }
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("collector: {e}");
        std::process::exit(1);
    }
}
