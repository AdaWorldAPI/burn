//! Single dispatch point for SIMD / AMX / GEMM acceleration through the
//! AdaWorldAPI/ndarray fork.
//!
//! All dispatch is **unconditional** — the fork's `backend::native`
//! polyfill provides runtime `LazyLock<Tier>` dispatch (AVX-512 / AVX2+FMA /
//! scalar), and `ndarray::backend::{gemm_f32, gemm_f64, dot_f32, ...}`
//! routes to MKL → OpenBLAS → native depending on which (optional) FFI
//! features are enabled at the ndarray level. With no FFI features, the
//! native polyfill carries the load.
//!
//! AMX f32/i8/bf16 matmul (Intel Sapphire Rapids+) is taken whenever
//! available at runtime; otherwise we fall through to the polyfilled
//! GEMM. INT8 GEMM goes AMX → VNNI → polyfilled scalar fallback.
//!
//! This module is the single file to edit if dispatch policy changes.

use crate::{NdArrayElement, SharedArray};
use burn_backend::{ElementConversion, Shape};
use core::any::TypeId;
use ndarray::s;

// ============================================================================
// Capability detection
// ============================================================================

/// Returns `true` when the running CPU has Intel AMX hardware enabled and
/// usable. Always `false` outside x86_64 Linux.
#[inline]
pub fn amx_available() -> bool {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        ndarray::hpc::amx_matmul::amx_available()
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        false
    }
}

// ============================================================================
// Type-specialization helpers
// ============================================================================
// burn-side ops are generic over `E: NdArrayElement`. When we want to reach
// for an f32-only fast path (AMX, sgemm), we cast the generic view to a
// concrete `ArrayView2<f32>`. A `TypeId` check guards the cast; the
// transmute is sound when `E == f32` because `ArrayView2<E>` and
// `ArrayView2<f32>` are then the same monomorphization.

#[inline]
fn try_view_as_f32<'a, E: 'static>(
    view: &ndarray::ArrayView2<'a, E>,
) -> Option<ndarray::ArrayView2<'a, f32>> {
    if TypeId::of::<E>() == TypeId::of::<f32>() {
        Some(unsafe { core::mem::transmute_copy::<_, ndarray::ArrayView2<'a, f32>>(view) })
    } else {
        None
    }
}

#[inline]
fn try_view_mut_as_f32<'a, E: 'static>(
    view: &mut ndarray::ArrayViewMut2<'a, E>,
) -> Option<ndarray::ArrayViewMut2<'a, f32>> {
    if TypeId::of::<E>() == TypeId::of::<f32>() {
        Some(unsafe { core::mem::transmute_copy::<_, ndarray::ArrayViewMut2<'a, f32>>(view) })
    } else {
        None
    }
}

// ============================================================================
// Matmul dispatcher
// ============================================================================

/// Single-batch matmul dispatcher. Decision tree (top-down, first hit wins):
///   1. f32 + AMX hardware (x86_64 Linux + `amx_available`) → AMX tile matmul.
///   2. Default → `ndarray::linalg::general_mat_mul` (matrixmultiply,
///      always-fast path used by upstream ndarray).
///
/// The ndarray fork's `backend::native` polyfill (AVX-512/AVX2/scalar) is
/// invoked transparently inside `general_mat_mul`'s GEMM path on supported
/// hardware. AMX returning `Err` falls through to the default — behavior is
/// always defined.
pub(crate) fn matmul_2d<E: NdArrayElement>(
    lhs: ndarray::ArrayView2<'_, E>,
    rhs: ndarray::ArrayView2<'_, E>,
    out: &mut ndarray::ArrayViewMut2<'_, E>,
) {
    // (1) AMX f32 — runtime gated.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        if amx_available()
            && let (Some(l), Some(r), Some(o)) = (
                try_view_as_f32(&lhs),
                try_view_as_f32(&rhs),
                try_view_mut_as_f32(out),
            )
            && ndarray::hpc::amx_matmul::matmul_f32(l, r, o).is_ok()
        {
            return;
        }
    }

    // (2) Default — runs through matrixmultiply, which itself gets SIMD
    //     codegen from rustc on f32/f64 in release builds.
    let alpha: E = 1.0.elem();
    let beta: E = 0.0.elem();
    ndarray::linalg::general_mat_mul(alpha, &lhs, &rhs, beta, out);
}

/// Reshape + dispatch helper preserving the original 3D batched-matmul loop.
#[allow(clippy::too_many_arguments)]
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
// INT8 GEMM dispatcher
// ============================================================================

/// INT8 GEMM dispatcher. Decision tree internally:
///   1. AMX hardware (x86_64 Linux + `amx_available`) → TDPBUSD tile (256 MACs/insn).
///   2. AVX-512-VNNI (`vpdpbusd`, 64 MACs/insn) → `vnni_gemm`.
///   3. Polyfill scalar fallback.
///
/// Routes through `ndarray::hpc::quantized::int8_gemm_f32`, which dispatches
/// AMX/VNNI/scalar internally.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn int8_gemm_f32(
    a: &[u8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale_a: f32,
    zero_point_a: i32,
    scale_b: f32,
) {
    ndarray::hpc::quantized::int8_gemm_f32(a, b, c, m, n, k, scale_a, zero_point_a, scale_b);
}

// ============================================================================
// Reductions (f32) — runtime-dispatched SIMD via the polyfill.
// ============================================================================

#[inline]
#[allow(dead_code)]
pub fn sum_f32(s: &[f32]) -> f32 {
    ndarray::hpc::reductions::sum_f32(s)
}

#[inline]
#[allow(dead_code)]
pub fn mean_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::mean_f32(s)
}

#[inline]
#[allow(dead_code)]
pub fn max_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::max_f32(s)
}

#[inline]
#[allow(dead_code)]
pub fn min_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::min_f32(s)
}

#[inline]
#[allow(dead_code)]
pub fn argmax_f32(s: &[f32]) -> Option<usize> {
    ndarray::hpc::reductions::argmax_f32(s)
}

#[inline]
#[allow(dead_code)]
pub fn argmin_f32(s: &[f32]) -> Option<usize> {
    ndarray::hpc::reductions::argmin_f32(s)
}

#[inline]
#[allow(dead_code)]
pub fn nrm2_f32(s: &[f32]) -> f32 {
    ndarray::hpc::reductions::nrm2_f32(s)
}

// ============================================================================
// Quantization helpers — Q4_0, I4, I2, I8 routed through the fork.
// ============================================================================

#[inline]
#[allow(dead_code)]
pub fn quantize_f32_to_q4_0(data: &[f32]) -> (alloc::vec::Vec<u8>, alloc::vec::Vec<f32>) {
    ndarray::hpc::quantized::quantize_f32_to_q4_0(data)
}

#[inline]
#[allow(dead_code)]
pub fn dequantize_q4_0_to_f32(packed: &[u8], scales: &[f32]) -> alloc::vec::Vec<f32> {
    ndarray::hpc::quantized::dequantize_q4_0_to_f32(packed, scales)
}

#[inline]
#[allow(dead_code)]
pub fn quantize_f32_to_i8(
    data: &[f32],
) -> (alloc::vec::Vec<i8>, ndarray::hpc::quantized::QuantParams) {
    ndarray::hpc::quantized::quantize_f32_to_i8(data)
}

#[inline]
#[allow(dead_code)]
pub fn dequantize_i8_to_f32(
    codes: &[i8],
    params: &ndarray::hpc::quantized::QuantParams,
    n: usize,
) -> alloc::vec::Vec<f32> {
    ndarray::hpc::quantized::dequantize_i8_to_f32(codes, params, n)
}

#[inline]
#[allow(dead_code)]
pub fn quantize_f32_to_i4(
    data: &[f32],
) -> (alloc::vec::Vec<u8>, ndarray::hpc::quantized::QuantParams) {
    ndarray::hpc::quantized::quantize_f32_to_i4(data)
}

#[inline]
#[allow(dead_code)]
pub fn dequantize_i4_to_f32(
    packed: &[u8],
    params: &ndarray::hpc::quantized::QuantParams,
    len: usize,
) -> alloc::vec::Vec<f32> {
    ndarray::hpc::quantized::dequantize_i4_to_f32(packed, params, len)
}

#[inline]
#[allow(dead_code)]
pub fn quantize_f32_to_i2(
    data: &[f32],
) -> (alloc::vec::Vec<u8>, ndarray::hpc::quantized::QuantParams) {
    ndarray::hpc::quantized::quantize_f32_to_i2(data)
}

#[inline]
#[allow(dead_code)]
pub fn dequantize_i2_to_f32(
    packed: &[u8],
    params: &ndarray::hpc::quantized::QuantParams,
    n: usize,
) -> alloc::vec::Vec<f32> {
    ndarray::hpc::quantized::dequantize_i2_to_f32(packed, params, n)
}
