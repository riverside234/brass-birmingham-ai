use brass_lab_collector::{check, Lane, Result, CONTRACT};
use serde_json::{json, Value};
use std::io::{self, Read, Write};

const MAX_FRAME: usize = 4 * 1024 * 1024;
fn run() -> Result<()> {
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut lane: Option<Lane> = None;
    let mut previous = 0;
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
        let result: Result<Value> = (|| {
            let token = r["token"].as_str().unwrap_or("");
            let revision = r["revision"].as_u64().unwrap_or(u64::MAX);
            match r["op"].as_str().unwrap_or("") {
                "hello" => Ok(
                    json!({"contract":CONTRACT,"core":brass_ai_core::protocol::BuildIdentity::current(),"search_enabled":false}),
                ),
                "begin" | "fork" => {
                    check(lane.is_none(), "release prior lane before begin")?;
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
                    Ok(json!({"released":true}))
                }
                _ => Err("unsupported collector operation".into()),
            }
        })();
        let failed = result.is_err();
        let reply = match result {
            Ok(data) => json!({"protocol":CONTRACT,"id":id,"ok":true,"data":data}),
            Err(e) => json!({"protocol":CONTRACT,"id":id,"ok":false,"error":e.to_string()}),
        };
        let bytes = serde_json::to_vec(&reply)?;
        check(bytes.len() <= MAX_FRAME, "response frame too large")?;
        output.write_all(&(bytes.len() as u32).to_le_bytes())?;
        output.write_all(&bytes)?;
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
