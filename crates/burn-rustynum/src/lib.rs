//! RustyNum backend for the Burn deep learning framework.
//!
//! Provides AVX-512 SIMD-accelerated tensor operations via rustynum's
//! explicit `portable_simd` kernels. CPU-only, nightly Rust required.

extern crate alloc;

mod backend;
pub mod tensor;
mod ops;

pub use backend::{RustyNum, RustyNumDevice};
pub use tensor::{RustyNumTensor, RustyNumQTensor};
