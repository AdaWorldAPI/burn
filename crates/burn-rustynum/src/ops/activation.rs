use burn_backend::ops::ActivationOps;

use crate::backend::RustyNum;

/// All activation functions have default implementations in the trait that compose
/// from FloatTensorOps primitives. Phase 2 will override relu/gelu/sigmoid with
/// rustynum SIMD kernels for AVX-512 vectorized activation.
impl ActivationOps<Self> for RustyNum {}
