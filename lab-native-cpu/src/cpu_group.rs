//! Several teacher tables in one worker process (g4288).
//!
//! Each table runs the unchanged single-table teacher (`teacher::run_request`) on its
//! own thread with its own ONNX session, and every session in the process shares one
//! prepacked copy of the network weights (`cpu_eval::shared_session`). With one process
//! per table, 48 private 17 MB copies were read at every forward: about 440 GB/s at
//! g4287, where a search simulation cost 1.85 ms while every table ran and 0.48 ms once
//! most had finished.
//!
//! Protocol, after the group header `{"contract": CONTRACT}` on stdin:
//! * in: one frame per table, `{"table": i, "request": <teacher request>}` (other fields
//!   are the trainer's); end of input means no more tables, and the process exits once
//!   they all finish;
//! * out: every frame a table writes, whole, as `[u32 length][u32 table][frame]`; first a
//!   `{"event": "thread", "tid": n}` frame (Linux) so the trainer can pin the table's
//!   thread to its CPU, and a final `{"event": "error"}` frame for a table that fails.
use crate::{check, Result};
use serde_json::{json, Value};
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};

pub const CONTRACT: &str = "brass-lab-teacher-group/1.0.0";
const LIMIT: usize = 64 * 1024 * 1024;
/// The single-table teacher runs on a process's main thread (8 MB on Linux); be generous.
const STACK: usize = 64 * 1024 * 1024;

/// One table's request, run with its own input and output.
pub type Table = fn(Value, io::Empty, Tagged) -> Result<()>;

fn read_frame(input: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut size = [0u8; 4];
    match input.read_exact(&mut size) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let n = u32::from_le_bytes(size) as usize;
    check(n > 0 && n <= LIMIT, "teacher group frame bound")?;
    let mut data = vec![0; n];
    input.read_exact(&mut data)?;
    Ok(Some(data))
}

/// A table's output stream: the bytes of each flushed frame go out together, tagged.
pub struct Tagged {
    index: u32,
    shared: Arc<Mutex<io::Stdout>>,
    buffer: Vec<u8>,
}

impl Tagged {
    fn new(index: u32, shared: Arc<Mutex<io::Stdout>>) -> Self {
        Self { index, shared, buffer: Vec::new() }
    }

    /// One whole frame as teacher::emit writes it: metadata length, then the metadata.
    fn event(&mut self, value: &Value) -> Result<()> {
        let meta = value.to_string().into_bytes();
        let mut frame = (meta.len() as u32).to_le_bytes().to_vec();
        frame.extend(meta);
        self.write_all(&(frame.len() as u32).to_le_bytes())?;
        self.write_all(&frame)?;
        self.flush()?;
        Ok(())
    }
}

impl Write for Tagged {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        if self.buffer.len() > LIMIT + 4 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "teacher group output bound"));
        }
        let mut out = self.shared.lock().map_err(|_| io::Error::other("teacher group output poisoned"))?;
        out.write_all(&((self.buffer.len() + 4) as u32).to_le_bytes())?;
        out.write_all(&self.index.to_le_bytes())?;
        out.write_all(&self.buffer)?;
        out.flush()?;
        self.buffer.clear();
        Ok(())
    }
}

/// The calling thread's kernel id on Linux (`/proc/thread-self` is `pid/task/tid`).
fn thread_id() -> Option<u64> {
    let link = std::fs::read_link("/proc/thread-self").ok()?;
    link.file_name()?.to_str()?.parse().ok()
}

fn table_thread(index: u32, request: Value, shared: Arc<Mutex<io::Stdout>>, table: Table) {
    let output = shared.clone();
    // A panic is reported as the table's failure at once, not left to the trainer's watchdog.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || -> Result<()> {
        if let Some(tid) = thread_id() {
            Tagged::new(index, output.clone()).event(&json!({"event": "thread", "tid": tid}))?;
        }
        table(request, io::empty(), Tagged::new(index, output))
    }))
    .unwrap_or_else(|_| Err("teacher table panicked".into()));
    if let Err(error) = outcome {
        let _ = Tagged::new(index, shared).event(&json!({"event": "error", "error": error.to_string()}));
    }
}

/// Read table requests until end of input; run each on its own thread; wait for all.
pub fn run(mut input: impl Read, table: Table) -> Result<()> {
    let shared = Arc::new(Mutex::new(io::stdout()));
    let mut threads = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    while let Some(frame) = read_frame(&mut input)? {
        let message: Value = serde_json::from_slice(&frame)?;
        let index = message["table"].as_u64().filter(|i| *i < u32::MAX as u64).ok_or("teacher group table index")? as u32;
        check(seen.insert(index), "repeated teacher group table")?;
        let request = message["request"].clone();
        let output = shared.clone();
        threads.push(
            std::thread::Builder::new()
                .name(format!("teacher-table-{index}"))
                .stack_size(STACK)
                .spawn(move || table_thread(index, request, output, table))?,
        );
    }
    for thread in threads {
        thread.join().map_err(|_| "teacher group table panicked")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_group_frames_are_tagged_and_whole() {
        // A table frame as teacher::write_frame writes it: length, then bytes, then flush.
        let mut out = Tagged::new(7, Arc::new(Mutex::new(io::stdout())));
        out.write_all(&3u32.to_le_bytes()).unwrap();
        out.write_all(b"abc").unwrap();
        assert_eq!(out.buffer, [3, 0, 0, 0, b'a', b'b', b'c']);
        let mut input: &[u8] = &[];
        assert!(read_frame(&mut input).unwrap().is_none());
        let mut framed: &[u8] = &[2, 0, 0, 0, b'{', b'}'];
        assert_eq!(read_frame(&mut framed).unwrap().unwrap(), b"{}");
    }
}
