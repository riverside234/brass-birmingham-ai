//! Local isolation wrapper, 2026-10-24. Original modules are byte-identical to
//! the L01 snapshot. See UPSTREAM.json and LICENSE (GNU AGPL version 3).
//! This crate deliberately excludes upstream web, AI/search and training.
pub mod actions;
pub mod board;
pub mod cards;
pub mod consts;
pub mod core;
pub mod game;
pub mod market;
pub mod utils;
pub mod validation;

pub use actions::*;
pub use board::*;
pub use cards::*;
pub use core::*;
pub use game::*;
pub use market::*;
pub use validation::*;
