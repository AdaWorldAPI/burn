//! Single dispatch point for SIMD / AMX / MKL acceleration through `ndarray`.
//!
//! All paths fall back to `ndarray::linalg::general_mat_mul` (matrixmultiply
//! under the hood) when the relevant feature is off or the input doesn't
//! match the fast path's preconditions. This keeps the default build identical
//! to upstream burn-ndarray semantics.
//!
//! Feature flags (defined in `crates/burn-ndarray/Cargo.toml`):
//! - `simd-runtime-dispatch` — opts into the fork's `LazyLock<Tier>` polyfill
//!   for elementwise/reduction kernels.
//! - `simd-amx` — Intel AMX matmul fast path; gated additionally on
//!   `target_arch = "x86_64"` + `target_os = "linux"` and runtime
//!   `amx_available()`.
//! - `mkl` — routes f32/f64 GEMM through `ndarray::backend::mkl::sgemm/dgemm`.
//!
//! When the workspace `ndarray` source is the AdaWorldAPI/ndarray fork, all
//! the fork-provided symbols below are present. Against upstream crates.io
//! ndarray they are not — but feature flags keep those code paths gated, so
//! the default build still compiles.

use crate::{NdArrayElement, SharedArray};
use burn_backend::{ElementConversion, Shape};
use core::any::TypeId;
use ndarray::s;

/// Returns `true` when the running CPU has Intel AMX hardware enabled and the
/// `simd-amx` feature is on. Always `false` outside x86_64 Linux.
#[inline]
#[allow(dead_code)]
pub fn amx_available() -> bool {
    #[cfg(all(feature = "simd-amx", target_arch = "x86_64", target_os = "linux"))]
    {
        ndarray::hpc::amx_matmul::amx_available()
    }
    #[cfg(not(all(feature = "simd-amx", target_arch = "x86_64", target_os = "linux")))]
    {
        false
    }
}

// ============================================================================
// Type-specialization helpers
// ============================================================================
// burn-side ops are generic over `E: NdArrayElement`. When we want to reach
// for an f32-only fast path (AMX, MKL sgemm), we need to convert the generic
// view to a concrete `ArrayView2<f32>`. A `TypeId` check guards the cast; the
// transmute is sound when `E == f32` because `ArrayView2<E>` and
// `ArrayView2<f32>` then refer to the exact same monomorphization.

#[allow(dead_code)] // used only when fork-feature flags are enabled
#[inline]
fn try_view_as_f32<'a, E: 'static>(
    view: &ndarray::ArrayView2<'a, E>,
) -> Option<ndarray::ArrayView2<'a, f32>> {
    if TypeId::of::<E>() == TypeId::of::<f32>() {
        // SAFETY: TypeId equality means E and f32 are the same type.
        Some(unsafe { core::mem::transmute_copy::<_, ndarray::ArrayView2<'a, f32>>(view) })
    } else {
        None
    }
}

#[allow(dead_code)]
#[inline]
fn try_view_as_f64<'a, E: 'static>(
    view: &ndarray::ArrayView2<'a, E>,
) -> Option<ndarray::ArrayView2<'a, f64>> {
    if TypeId::of::<E>() == TypeId::of::<f64>() {
        Some(unsafe { core::mem::transmute_copy::<_, ndarray::ArrayView2<'a, f64>>(view) })
    } else {
        None
    }
}

#[allow(dead_code)]
#[inline]
fn try_view_mut_as_f32<'a, E: 'static>(
    view: &mut ndarray::ArrayViewMut2<'a, E>,
) -> Option<ndarray::ArrayViewMut2<'a, f32>> {
    if TypeId::of::<E>() == TypeId::of::<f32>() {
        // SAFETY: TypeId equality means E and f32 are the same type.
        // `transmute_copy` on a mutable view aliases the underlying storage,
        // but because the original `view` is reborrowed for the duration of
        // the AMX/MKL call and then dropped, no concurrent access occurs.
        Some(unsafe { core::mem::transmute_copy::<_, ndarray::ArrayViewMut2<'a, f32>>(view) })
    } else {
        None
    }
}

#[allow(dead_code)]
#[inline]
fn try_view_mut_as_f64<'a, E: 'static>(
    view: &mut ndarray::ArrayViewMut2<'a, E>,
) -> Option<ndarray::ArrayViewMut2<'a, f64>> {
    if TypeId::of::<E>() == TypeId::of::<f64>() {
        Some(unsafe { core::mem::transmute_copy::<_, ndarray::ArrayViewMut2<'a, f64>>(view) })
    } else {
        None
    }
}

// ============================================================================
// Matmul dispatcher
// ============================================================================

/// Single-batch matmul dispatcher.
///
/// Decision tree (top-down, first hit wins):
///   1. f32 + AMX feature + `amx_available()` → `hpc::amx_matmul::matmul_f32`.
///   2. f32 + MKL feature → `backend::mkl::sgemm`.
///   3. f64 + MKL feature → `backend::mkl::dgemm`.
///   4. Default: `ndarray::linalg::general_mat_mul`.
///
/// If a fast path returns an error (shape mismatch, etc.), we fall through to
/// the default path so behavior is always defined.
pub(crate) fn matmul_2d<E: NdArrayElement>(
    lhs: ndarray::ArrayView2<'_, E>,
    rhs: ndarray::ArrayView2<'_, E>,
    out: &mut ndarray::ArrayViewMut2<'_, E>,
) {
    // (1) AMX f32 — x86_64 Linux only.
    #[cfg(all(feature = "simd-amx", target_arch = "x86_64", target_os = "linux"))]
    {
        if amx_available() {
            if let (Some(l), Some(r), Some(o)) = (
                try_view_as_f32(&lhs),
                try_view_as_f32(&rhs),
                try_view_mut_as_f32(out),
            ) {
                if ndarray::hpc::amx_matmul::matmul_f32(l, r, o).is_ok() {
                    return;
                }
            }
        }
    }

    // (2) MKL f32.
    #[cfg(feature = "mkl")]
    {
        if let (Some(l), Some(r), Some(o)) = (
            try_view_as_f32(&lhs),
            try_view_as_f32(&rhs),
            try_view_mut_as_f32(out),
        ) {
            if ndarray::backend::mkl::sgemm(l, r, o, 1.0, 0.0).is_ok() {
                return;
            }
        }
        if let (Some(l), Some(r), Some(o)) = (
            try_view_as_f64(&lhs),
            try_view_as_f64(&rhs),
            try_view_mut_as_f64(out),
        ) {
            if ndarray::backend::mkl::dgemm(l, r, o, 1.0, 0.0).is_ok() {
                return;
            }
        }
    }

    // (4) Default.
    let alpha: E = 1.0.elem();
    let beta: E = 0.0.elem();
    ndarray::linalg::general_mat_mul(alpha, &lhs, &rhs, beta, out);
}

/// Reshape + dispatch helper preserving the original 3D batched-matmul loop.
pub(crate) fn matmul_batched<E: NdArrayElement>(
    lhs: SharedArray<E>,
    rhs: SharedArray<E>,
    num_l_batches: usize,
    num_r_batches: usize,
    num_out_batches: usize,
    m: usize,
    k: usize,
    n: usize,
    strides_lhs: &[usize],
    strides_rhs: &[usize],
    strides_out: &[usize],
) -> ndarray::Array3<E> {
    use crate::{UnsafeSharedRef, iter_range_par, ops::NdArrayOps, run_par};

    let lhs_array = NdArrayOps::reshape(lhs, Shape::new([num_l_batches, m, k]));
    let rhs_array = NdArrayOps::reshape(rhs, Shape::new([num_r_batches, k, n]));

    run_par!(|| {
        let mut out_array = ndarray::Array3::<E>::zeros((num_out_batches, m, n));
        let unsafe_shared_out_array = UnsafeSharedRef::new(&mut out_array);

        iter_range_par!(0, num_out_batches).for_each(|out_batch| {
            let out_index = unflatten(strides_out, out_batch);
            let l_batch = flatten(strides_lhs, &out_index);
            let r_batch = flatten(strides_rhs, &out_index);

            let lhs_slice = lhs_array.slice(s!(l_batch, .., ..));
            let rhs_slice = rhs_array.slice(s!(r_batch, .., ..));

            unsafe {
                let mut out_slice = unsafe_shared_out_array
                    .get()
                    .slice_mut(s!(out_batch, .., ..));

                matmul_2d::<E>(lhs_slice, rhs_slice, &mut out_slice);
            }
        });

        out_array
    })
}

fn unflatten(strides: &[usize], linear_index: usize) -> alloc::vec::Vec<usize> {
    let mut coord = alloc::vec::Vec::with_capacity(strides.len());
    let mut rem = linear_index;
    for &stride in strides {
        coord.push(rem / stride);
        rem %= stride;
    }
    coord
}

fn flatten(strides: &[usize], index: &[usize]) -> usize {
    debug_assert_eq!(strides.len(), index.len());
    strides.iter().zip(index.iter()).map(|(s, i)| s * i).sum()
}

// ============================================================================
// Reductions (f32) — routed through ndarray::hpc::reductions when available.
// ============================================================================
//
// `ndarray::hpc::reductions::*` runs through the polyfill SIMD layer
// (AVX-512/AVX2/NEON/scalar runtime-dispatched per #117/#118/#122). Falls
// back to a scalar implementation when the `simd-runtime-dispatch` feature
// is off, preserving the default build path.

#[inline]
#[allow(dead_code)]
pub fn sum_f32(s: &[f32]) -> f32 {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::reductions::sum_f32(s)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        s.iter().sum()
    }
}

#[inline]
#[allow(dead_code)]
pub fn mean_f32(s: &[f32]) -> Option<f32> {
    if s.is_empty() {
        return None;
    }
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::reductions::mean_f32(s)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        Some(sum_f32(s) / s.len() as f32)
    }
}

#[inline]
#[allow(dead_code)]
pub fn max_f32(s: &[f32]) -> Option<f32> {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::reductions::max_f32(s)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        s.iter().copied().reduce(f32::max)
    }
}

#[inline]
#[allow(dead_code)]
pub fn min_f32(s: &[f32]) -> Option<f32> {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::reductions::min_f32(s)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        s.iter().copied().reduce(f32::min)
    }
}

#[inline]
#[allow(dead_code)]
pub fn argmax_f32(s: &[f32]) -> Option<usize> {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::reductions::argmax_f32(s)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        s.iter()
            .enumerate()
            .reduce(|a, b| if b.1 > a.1 { b } else { a })
            .map(|(i, _)| i)
    }
}

#[inline]
#[allow(dead_code)]
pub fn argmin_f32(s: &[f32]) -> Option<usize> {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::reductions::argmin_f32(s)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        s.iter()
            .enumerate()
            .reduce(|a, b| if b.1 < a.1 { b } else { a })
            .map(|(i, _)| i)
    }
}

#[inline]
#[allow(dead_code)]
pub fn nrm2_f32(s: &[f32]) -> f32 {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::reductions::nrm2_f32(s)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        s.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

// ============================================================================
// Quantization helpers (Q4_0, I4, I2, I8) — routed through ndarray::hpc::quantized
// ============================================================================
//
// These wrap the fork's GGUF-compatible quantization APIs (#120) plus the
// legacy I8/I4/I2 paths. The wrappers exist so burn-ndarray's `QTensorOps`
// can swap between the fork and a scalar fallback via a single feature gate.

#[inline]
#[allow(dead_code)]
pub fn quantize_f32_to_q4_0(data: &[f32]) -> (alloc::vec::Vec<u8>, alloc::vec::Vec<f32>) {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::quantized::quantize_f32_to_q4_0(data)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        let _ = data;
        unimplemented!(
            "Q4_0 quantization requires the `simd-runtime-dispatch` feature on \
             burn-ndarray, which routes to ndarray::hpc::quantized."
        );
    }
}

#[inline]
#[allow(dead_code)]
pub fn dequantize_q4_0_to_f32(packed: &[u8], scales: &[f32]) -> alloc::vec::Vec<f32> {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        ndarray::hpc::quantized::dequantize_q4_0_to_f32(packed, scales)
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        let _ = (packed, scales);
        unimplemented!(
            "Q4_0 dequantization requires the `simd-runtime-dispatch` feature \
             on burn-ndarray, which routes to ndarray::hpc::quantized."
        );
    }
}
