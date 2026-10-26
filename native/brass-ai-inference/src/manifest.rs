use crate::{invalid, require, Input, Result};
use brass_ai_core::features::{layout::META, ACTION_DIM, MAX_CANDIDATES, STATE_DIM};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

pub const INPUTS: [&str; 6] = [
    "states",
    "actions",
    "legal_mask",
    "stable_to_relative",
    "remaining",
    "descriptors",
];
pub const OUTPUTS: [&str; 5] = [
    "policy_logits",
    "win_share_4",
    "placement_4",
    "vp_4",
    "remaining_utility_4",
];
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn digest(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub fn label(id: &str) -> bool {
    !id.is_empty() && id.len() <= 512 && !id.chars().any(char::is_control)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub model_id: String,
    pub checkpoint_sha256: String,
    pub export_sha256: String,
    pub export_file: String,
    pub metadata_sha256: String,
    pub runtime_file: String,
    pub runtime_sha256: String,
    pub runtime_version: String,
    pub binding_version: String,
    pub architecture: String,
    pub feature_schema: String,
    pub group_layout: String,
    pub entity_layout: String,
    pub reward_id: String,
    pub max_cost: f64,
    pub precision: String,
    pub opset: u32,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub mechanical_only: bool,
    pub utility_calibration: Option<serde_json::Value>,
    pub intra_threads: u8,
    pub inter_threads: u8,
    pub sequential: bool,
    pub provider: String,
    pub optimization: String,
    pub source_archive_sha256: String,
}
fn child(root: &Path, file: &str) -> Result<PathBuf> {
    require(
        !file.is_empty()
            && Path::new(file)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "model path must be bundle-relative",
    )?;
    let path = root
        .join(file)
        .canonicalize()
        .map_err(|e| invalid(e.to_string()))?;
    require(path.starts_with(root), "model file escapes bundle")?;
    Ok(path)
}
pub struct Bundle {
    pub manifest: Manifest,
    pub manifest_sha256: String,
    pub model_bytes: Vec<u8>,
    pub runtime_path: PathBuf,
}
fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let file = fs::File::open(path).map_err(|e| invalid(e.to_string()))?;
    require(
        file.metadata().map_err(|e| invalid(e.to_string()))?.len() <= maximum,
        "model bundle file exceeds bound",
    )?;
    let mut bytes = Vec::new();
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| invalid(e.to_string()))?;
    require(
        bytes.len() as u64 <= maximum,
        "model bundle file grew beyond bound",
    )?;
    Ok(bytes)
}
impl Bundle {
    pub fn load(path: &Path, expected: &str) -> Result<Self> {
        require(digest(expected), "invalid manifest digest")?;
        let path = path.canonicalize().map_err(|e| invalid(e.to_string()))?;
        let bytes = read_bounded(&path, 65536)?;
        require(
            bytes.len() <= 65536 && hash(&bytes) == expected,
            "manifest hash mismatch",
        )?;
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        manifest.validate()?;
        let root = path
            .parent()
            .ok_or_else(|| invalid("bundle directory missing"))?;
        let model_bytes = read_bounded(&child(root, &manifest.export_file)?, 128 * 1024 * 1024)?;
        require(
            model_bytes.len() <= 128 * 1024 * 1024 && hash(&model_bytes) == manifest.export_sha256,
            "export hash/size mismatch",
        )?;
        let metadata = read_bounded(&child(root, "model.json")?, 1024 * 1024)?;
        require(
            metadata.len() <= 1024 * 1024 && hash(&metadata) == manifest.metadata_sha256,
            "export metadata hash mismatch",
        )?;
        let meta: serde_json::Value =
            serde_json::from_slice(&metadata).map_err(|e| invalid(e.to_string()))?;
        require(
            meta["model_id"] == manifest.model_id
                && meta["reward_id"] == manifest.reward_id
                && meta["sha256"] == manifest.export_sha256
                && meta["actor"] == false
                && meta["parity"]["passed"] == true
                && meta["config"]["state_dim"] == STATE_DIM
                && meta["config"]["action_dim"] == ACTION_DIM
                && meta["config"]["architecture"] == manifest.architecture
                && meta["config"]["utility"]["insolvency"]["max_cost"].as_f64()
                    == Some(manifest.max_cost),
            "model metadata contract mismatch",
        )?;
        let runtime_path = child(root, &manifest.runtime_file)?;
        require(
            hash(&read_bounded(&runtime_path, 128 * 1024 * 1024)?) == manifest.runtime_sha256,
            "runtime hash mismatch",
        )?;
        Ok(Self {
            manifest,
            manifest_sha256: expected.into(),
            model_bytes,
            runtime_path,
        })
    }
}
impl Manifest {
    pub fn validate(&self) -> Result<()> {
        require(
            self.format == "brass-native-model/1.0.0"
                && self.architecture == "brass-four-seat-mlp-resnet/11.2.0"
                && self.precision == "float32"
                && self.opset == 17
                && self.inputs == INPUTS
                && self.outputs == OUTPUTS,
            "unsupported model/export contract",
        )?;
        require(
            self.feature_schema == META.feature_schema_sha256
                && self.group_layout == META.group_layout["identity"]
                && self.entity_layout == META.entity_layout["identity"],
            "feature layout mismatch",
        )?;
        require(
            [
                &self.model_id,
                &self.checkpoint_sha256,
                &self.export_sha256,
                &self.metadata_sha256,
                &self.runtime_sha256,
                &self.reward_id,
                &self.source_archive_sha256,
            ]
            .iter()
            .all(|s| digest(s)),
            "invalid pinned identity",
        )?;
        require(
            self.runtime_version == "1.22.1"
                && self.binding_version == "2.0.0-rc.10"
                && self.provider == "CPUExecutionProvider"
                && self.optimization == "all-exact-gelu"
                && self.intra_threads == 1
                && self.inter_threads == 1
                && self.sequential,
            "unsupported CPU runtime configuration",
        )?;
        require(
            self.max_cost.is_finite() && (0. ..=1.).contains(&self.max_cost),
            "invalid model cost cap",
        )?;
        // L09 has no authority to promote a calibration receipt into live admission.
        require(
            self.mechanical_only && self.utility_calibration.is_none(),
            "only mechanical model admission is qualified",
        )
    }
    pub fn validate_input(&self, input: &Input<'_>) -> Result<()> {
        let (id, c, t, n) = (input.id, input.context, input.tensors, input.choices.len());
        c.validate()?;
        require(
            label(&id.request_id)
                && id.generation > 0
                && id.observer < 4
                && id.model_id == self.model_id
                && id.export_sha256 == self.export_sha256
                && id.reward_id == self.reward_id
                && id.profile_id == c.profile_id
                && c.model_id == self.model_id
                && c.reward_id == self.reward_id
                && c.max_cost == self.max_cost,
            "evaluation model/context identity mismatch",
        )?;
        require(
            n > 0
                && n <= MAX_CANDIDATES
                && t.state.len() == STATE_DIM
                && t.actions.len() == n * ACTION_DIM
                && t.legal_mask.len() == n
                && t.legal_mask.iter().all(|x| *x)
                && t.descriptors.len() == 136,
            "invalid evaluation tensor shape/mask",
        )?;
        require(
            t.state
                .iter()
                .chain(&t.actions)
                .chain(&t.descriptors)
                .chain(&t.remaining)
                .all(|x| x.is_finite()),
            "nonfinite evaluation input",
        )?;
        require(
            (0..4).all(|s| {
                t.stable_to_relative[s] == ((s + 4 - id.observer as usize) % 4) as i64
                    && t.remaining[s] == c.remaining[s] as f32
            }),
            "wrong seat mapping/budget",
        )?;
        for stable in 0..4 {
            let relative = t.stable_to_relative[stable] as usize;
            for field in 0..34 {
                let expected = c
                    .stable_descriptors
                    .as_ref()
                    .map_or(if field == 4 { 1. } else { 0. }, |d| d[stable][field]);
                require(
                    t.descriptors[relative * 34 + field] == expected,
                    "critic descriptor mapping mismatch",
                )?;
            }
        }
        let mut ids = HashSet::new();
        require(
            input
                .choices
                .iter()
                .all(|c| label(&c.id) && ids.insert(&c.id)),
            "invalid/duplicate candidates",
        )
    }
}
