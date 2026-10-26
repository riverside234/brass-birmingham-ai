//! Bounded diagnostic pagination. The native NN path borrows tensors directly.
use super::*;
use crate::state::simulation::Simulator;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeatureCommand {
    Encode {
        view: Box<Observation>,
        choices: Vec<Choice>,
        context: Box<FeatureContext>,
    },
    FromState {
        handle: u64,
        revision: u32,
        context: Box<FeatureContext>,
    },
    Page {
        encoding_id: u64,
        start: usize,
        limit: usize,
    },
}
#[derive(Debug, Serialize)]
pub struct FeatureResponse {
    pub encoding_id: u64,
    pub revision: u32,
    pub observer: Seat,
    pub tensors: FeatureReply,
}
struct Snapshot {
    encoding_id: u64,
    revision: u32,
    observer: Seat,
    choices: Vec<Choice>,
    context: FeatureContext,
}
#[derive(Default)]
pub struct FeatureSession {
    encoder: FeatureEncoder,
    sequence: u64,
    current: Option<Snapshot>,
}
impl FeatureSession {
    pub fn execute(
        &mut self,
        command: FeatureCommand,
        simulator: &Simulator,
        owner: u8,
    ) -> Result<FeatureResponse, CoreError> {
        check(owner < 2, "invalid feature owner")?;
        let (view, choices, context) = match command {
            FeatureCommand::Page {
                encoding_id,
                start,
                limit,
            } => return self.page(encoding_id, start, limit),
            FeatureCommand::Encode {
                view,
                choices,
                context,
            } => {
                self.current = None;
                (*view, choices, *context)
            }
            FeatureCommand::FromState {
                handle,
                revision,
                context,
            } => {
                self.current = None;
                let (view, choices) = simulator.feature_input(handle, revision)?;
                (view, choices, *context)
            }
        };
        self.encoder
            .encode(&view, &choices, &context, &mut || Ok(()))?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .filter(|s| *s < u64::MAX / 2)
            .ok_or_else(|| invalid("feature epoch exhausted"))?;
        let encoding_id = self.sequence * 2 + u64::from(owner);
        self.current = Some(Snapshot {
            encoding_id,
            revision: view.context.revision,
            observer: view.seat,
            choices,
            context,
        });
        self.page(encoding_id, 0, 64)
    }
    fn page(
        &self,
        encoding_id: u64,
        start: usize,
        limit: usize,
    ) -> Result<FeatureResponse, CoreError> {
        let snapshot = self
            .current
            .as_ref()
            .filter(|s| s.encoding_id == encoding_id)
            .ok_or_else(|| invalid("stale or foreign feature encoding"))?;
        let tensors = self
            .encoder
            .tensors()
            .ok_or_else(|| invalid("incomplete feature encoding"))?;
        Ok(FeatureResponse {
            encoding_id,
            revision: snapshot.revision,
            observer: snapshot.observer,
            tensors: FeatureReply::page(
                tensors,
                &snapshot.choices,
                &snapshot.context,
                start,
                limit,
            )?,
        })
    }
}
