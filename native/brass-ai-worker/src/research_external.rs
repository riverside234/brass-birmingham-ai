//! Research-only external evaluator for matched G_ref/G_ref_ORT attribution.
//! Uses the production tensor encoder, admission and reply validation. Each
//! search owner has one synchronous, one-thread evaluator process.
use brass_ai_core::{
    evaluation::EvaluationId,
    search::{check, invalid, Result},
};
use brass_ai_inference::{
    manifest::Manifest,
    service::{finish, validate_batch, RawRow},
    BatchOutput, Control, Evaluator, Input,
};
use serde::Deserialize;
use std::{
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::Instant,
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub python: PathBuf,
    pub script: PathBuf,
    pub mode: String,
    pub model: PathBuf,
    pub checkpoint: PathBuf,
    pub reference_source: PathBuf,
}
pub struct External {
    manifest: Manifest,
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
    buffer: Vec<u8>,
}
#[derive(Deserialize)]
struct Reply {
    rows: Vec<Row>,
    compute_ns: u64,
}
#[derive(Deserialize)]
struct Row {
    id: EvaluationId,
    choice_ids: Vec<String>,
    logits: Vec<f32>,
    values: [[f32; 4]; 4],
}
fn io_error(e: impl std::fmt::Display) -> brass_ai_core::CoreError {
    invalid(format!("external evaluator: {e}"))
}
// Binary tensors have a separate research-only bound. The production game's
// one-MiB JSON framing contract is deliberately unchanged.
const LIMIT: usize = 64 * 1024 * 1024;
fn receive(reader: &mut impl Read) -> Result<Vec<u8>> {
    let mut header = [0; 4];
    reader.read_exact(&mut header).map_err(io_error)?;
    let n = u32::from_le_bytes(header) as usize;
    check(n > 0 && n <= LIMIT, "external frame bound")?;
    let mut data = vec![0; n];
    reader.read_exact(&mut data).map_err(io_error)?;
    Ok(data)
}
impl External {
    pub fn load(settings: &Settings, manifest: Manifest) -> Result<Self> {
        check(
            ["torch", "ort", "scripted"].contains(&settings.mode.as_str()),
            "external mode",
        )?;
        let mut command = Command::new(&settings.python);
        command
            .arg("-u")
            .arg(&settings.script)
            .arg("--mode")
            .arg(&settings.mode)
            .arg("--model")
            .arg(&settings.model)
            .arg("--checkpoint")
            .arg(&settings.checkpoint)
            .arg("--reference-source")
            .arg(&settings.reference_source)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn().map_err(io_error)?;
        let input = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| invalid("external stdin"))?,
        );
        let output = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| invalid("external stdout"))?,
        );
        let mut result = Self {
            manifest,
            child,
            input,
            output,
            buffer: Vec::new(),
        };
        let ready = receive(&mut result.output)?;
        let ready: serde_json::Value = serde_json::from_slice(&ready).map_err(io_error)?;
        check(
            ready["ready"] == true
                && ready["threads"] == 1
                && ready["model_id"] == result.manifest.model_id
                && ready["export_sha256"] == result.manifest.export_sha256,
            "external evaluator identity",
        )?;
        Ok(result)
    }
}
impl Evaluator for External {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn evaluate(&mut self, inputs: &[Input<'_>], control: &Control) -> Result<BatchOutput> {
        validate_batch(&self.manifest, inputs)?;
        control.check()?;
        let start = Instant::now();
        let rows:Vec<_>=inputs.iter().map(|i|serde_json::json!({"id":i.id,"choice_ids":i.choices.iter().map(|c|&c.id).collect::<Vec<_>>() })).collect();
        let metadata = serde_json::to_vec(&serde_json::json!({"rows":rows})).map_err(io_error)?;
        self.buffer.clear();
        self.buffer
            .extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        self.buffer.extend_from_slice(&metadata);
        for input in inputs {
            let t = input.tensors;
            for slice in [&t.state[..], &t.actions[..]] {
                for v in slice {
                    self.buffer.extend_from_slice(&v.to_le_bytes());
                }
            }
            self.buffer
                .extend(t.legal_mask.iter().map(|v| u8::from(*v)));
            for v in t.stable_to_relative {
                self.buffer.extend_from_slice(&v.to_le_bytes());
            }
            for v in t.remaining.iter().chain(&t.descriptors) {
                self.buffer.extend_from_slice(&v.to_le_bytes());
            }
        }
        let packing_ns = brass_ai_inference::nanos(start);
        control.check()?;
        check(self.buffer.len() <= LIMIT, "external input bound")?;
        self.input
            .write_all(&(self.buffer.len() as u32).to_le_bytes())
            .map_err(io_error)?;
        self.input.write_all(&self.buffer).map_err(io_error)?;
        self.input.flush().map_err(io_error)?;
        let response = receive(&mut self.output)?;
        control.check()?;
        let reply: Reply = serde_json::from_slice(&response).map_err(io_error)?;
        finish(
            inputs,
            reply
                .rows
                .into_iter()
                .map(|r| RawRow {
                    id: r.id,
                    choice_ids: r.choice_ids,
                    logits: r.logits,
                    values: r.values,
                })
                .collect(),
            brass_ai_core::evaluation::EvaluationStats {
                packing_ns,
                inference_ns: reply.compute_ns,
                input_copy_bytes: self.buffer.len(),
                output_copy_bytes: response.len(),
                ..Default::default()
            },
        )
    }
}
impl Drop for External {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
