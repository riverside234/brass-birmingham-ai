//! Compile-time model admission for each local package. Both binaries use the
//! same game engine/search implementation; a package cannot replace its weights.
#[cfg(not(feature = "local-beta"))]
mod selected {
    pub const BACKEND: &str = "brass-gumbel-is-mcts/alpha-g890-v1";
    pub const MODEL: &str = "1145fecba3289511df1c518bb71fbfb5b2f6df351777151a5eb999b89cba0761";
    pub const EXPORT: &str = "64e4a8340f4beb7bbc4ee70219b5e461b1185bcde9e25c4bfc78e30936c91ae9";
    pub const CHECKPOINT: &str = "ecad3b932a8dd9d3b5d0cc49b5e9bb94a9e98b4536ffdc86a44179d003995cb8";
    pub const PROFILE: &str = "alpha-test-g890-neutral/1.0.0";
    pub const PROFILE_PREFIX: &str = "alpha-test-g890-";
    pub const SCOPE: &str = "part1-local-alpha-test";
    pub const DECISION: &str = "user-2026-10-25";
    pub const INDUSTRY_FOCUSES: &[&str] = &["neutral", "cotton", "manufacturer", "pottery", "brewery", "iron", "coal"];
    pub const BETA: bool = false;
}
#[cfg(feature = "local-beta")]
mod selected {
    pub const BACKEND: &str = "brass-gumbel-is-mcts/beta-g1290-v1";
    pub const MODEL: &str = "73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff";
    pub const EXPORT: &str = "1ee3340c0279592eb28cf2ce0bdbd24691e63be60f81d0196984bfc27c921424";
    pub const CHECKPOINT: &str = "ed8edb039a1257d591cc0d2e2b87eb7a2d7aa411ce0f7710a2d7f8ad487dc3b0";
    pub const PROFILE: &str = "beta-g1290-neutral/1.0.0";
    pub const PROFILE_PREFIX: &str = "beta-g1290-";
    pub const SCOPE: &str = "part1-local-beta-test";
    pub const DECISION: &str = "user-2026-11-01";
    pub const INDUSTRY_FOCUSES: &[&str] = &["neutral"];
    pub const BETA: bool = true;
}
pub use selected::*;
pub const CALIBRATION_REQUEST: &str = "f56bd03a91e661c5be0f941765ec9420c4729ef6caca0f8d294d61b3d3550670";
pub const SOURCE_CHECKPOINT: &str = "e14d5b7a4596944ac7dec1b30c676b52ab79beb7bbf2e55a8b8540cc8a312937";
