//! Single dispatch point for SIMD / AMX / GEMM acceleration through the
//! AdaWorldAPI/ndarray fork.
//!
//! All dispatch is **unconditional** — the fork's `backend::native`
//! provides a pure-Rust BLAS surface (CBLAS-compatible API names; the
//! impl is reverse-engineered, not an Intel library). It runs on the
//! fork's hand-rolled SIMD polyfill `ndarray::simd::F32x16`/`F64x8`/etc.
//! — built on stable-Rust `__m512` / `__m256` / `float32x4_t` intrinsics
//! organised by ISA in `simd_avx512.rs` / `simd_avx2.rs` / `simd_neon.rs`
//! / `simd_wasm.rs`, dispatched once at startup via `LazyLock<Tier>` in
//! `simd.rs`. (Not `std::simd` — that's nightly-only via `portable_simd`;
//! the polyfill is forward-compat: when `std::simd` stabilises, `simd.rs`
//! gets swapped with zero consumer changes.)
//!
//! `ndarray::backend::{gemm_f32, gemm_f64, dot_f32, ...}` is the
//! universally-available entry point.
//!
//! `hpc::vml::*` (vectorized math: exp, log, sqrt, erf, etc.) is also
//! reverse-engineered pure-Rust SIMD on `F32x16` / `F64x8`, not Intel VML.
//!
//! AMX f32/i8/bf16 matmul (Intel Sapphire Rapids+ hardware, but software
//! is stable inline asm — no Intel library involvement) is taken whenever
//! available at runtime; otherwise we fall through to the reverse-engineered
//! BLAS. INT8 GEMM goes AMX → AVX-512-VNNI → scalar fallback inside the
//! fork's `int8_gemm_f32`.
//!
//! This module is the single file to edit if dispatch policy changes.

use crate::{NdArrayElement, SharedArray};
use burn_backend::{ElementConversion, Shape};
#[cfg(feature = "std")] // only the std-gated f32 view casts use it
use core::any::TypeId;
use ndarray::s;

// ============================================================================
// Capability detection
// ============================================================================

/// Returns `true` when the running CPU has Intel AMX hardware enabled and
/// usable. Always `false` outside x86_64 Linux.
#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)] // used only under `feature = "amx-f32"`
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
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)] // used only under `feature = "amx-f32"`
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
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)] // used only under `feature = "amx-f32"`
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
    // (1) AMX f32 — opt-in (`amx-f32`) AND runtime gated.
    //
    // OFF BY DEFAULT, deliberately. AMX has no native f32 tile op; the fork's
    // `matmul_f32` rounds through bf16 (TDPBF16PS) and still returns `Ok(())`,
    // so the precision loss is silent. Measured on this workspace's Xeon
    // (`amx_probe` below): ~1e-3 relative error, stable across 4x4x4,
    // 16x16x16, 32x64x32 and ragged 17x33x15 — i.e. bf16 mantissa (eps 7.8e-3)
    // with f32 accumulation, not a tile-shape bug.
    //
    // That is outside the tolerance `burn-backend-tests`' linalg suite asserts:
    // taking this path unconditionally fails 50 tests (25 qr, 13 lu, 7 svd,
    // 3 det, 1 attention) that pass at 1826/1826 without it. Enable only for
    // workloads that have measured bf16-grade matmul as acceptable.
    #[cfg(all(
        target_arch = "x86_64",
        target_os = "linux",
        feature = "std",
        feature = "amx-f32"
    ))]
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
                let mut out_array = unsafe_shared_out_array.get();
                let mut out_slice = out_array.slice_mut(s!(out_batch, .., ..));

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
#[cfg(feature = "std")]
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
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn sum_f32(s: &[f32]) -> f32 {
    ndarray::hpc::reductions::sum_f32(ndarray::ArrayView1::from(s))
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn mean_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::mean_f32(ndarray::ArrayView1::from(s))
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn max_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::max_f32(ndarray::ArrayView1::from(s))
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn min_f32(s: &[f32]) -> Option<f32> {
    ndarray::hpc::reductions::min_f32(ndarray::ArrayView1::from(s))
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn argmax_f32(s: &[f32]) -> Option<usize> {
    ndarray::hpc::reductions::argmax_f32(ndarray::ArrayView1::from(s))
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn argmin_f32(s: &[f32]) -> Option<usize> {
    ndarray::hpc::reductions::argmin_f32(ndarray::ArrayView1::from(s))
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn nrm2_f32(s: &[f32]) -> f32 {
    ndarray::hpc::reductions::nrm2_f32(ndarray::ArrayView1::from(s))
}

// ============================================================================
// Quantization helpers — Q4_0, I4, I2, I8 routed through the fork.
// ============================================================================

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn quantize_f32_to_q4_0(data: &[f32]) -> (alloc::vec::Vec<u8>, alloc::vec::Vec<f32>) {
    ndarray::hpc::quantized::quantize_f32_to_q4_0(data)
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn dequantize_q4_0_to_f32(packed: &[u8], scales: &[f32]) -> alloc::vec::Vec<f32> {
    ndarray::hpc::quantized::dequantize_q4_0_to_f32(packed, scales)
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn quantize_f32_to_i8(
    data: &[f32],
) -> (alloc::vec::Vec<i8>, ndarray::hpc::quantized::QuantParams) {
    ndarray::hpc::quantized::quantize_f32_to_i8(data)
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn dequantize_i8_to_f32(
    codes: &[i8],
    params: &ndarray::hpc::quantized::QuantParams,
    n: usize,
) -> alloc::vec::Vec<f32> {
    ndarray::hpc::quantized::dequantize_i8_to_f32(codes, params, n)
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn quantize_f32_to_i4(
    data: &[f32],
) -> (alloc::vec::Vec<u8>, ndarray::hpc::quantized::QuantParams) {
    ndarray::hpc::quantized::quantize_f32_to_i4(data)
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn dequantize_i4_to_f32(
    packed: &[u8],
    params: &ndarray::hpc::quantized::QuantParams,
    len: usize,
) -> alloc::vec::Vec<f32> {
    ndarray::hpc::quantized::dequantize_i4_to_f32(packed, params, len)
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn quantize_f32_to_i2(
    data: &[f32],
) -> (alloc::vec::Vec<u8>, ndarray::hpc::quantized::QuantParams) {
    ndarray::hpc::quantized::quantize_f32_to_i2(data)
}

#[inline]
#[cfg(feature = "std")] // `ndarray::hpc` / `::backend` exist only with std
#[allow(dead_code)]
pub fn dequantize_i2_to_f32(
    packed: &[u8],
    params: &ndarray::hpc::quantized::QuantParams,
    n: usize,
) -> alloc::vec::Vec<f32> {
    ndarray::hpc::quantized::dequantize_i2_to_f32(packed, params, n)
}

/// Pins the measured precision of the fork's AMX f32 path, two-sided.
///
/// Fails if it degrades past bf16 (a real tile/shape bug), and ALSO fails if
/// it becomes exact — because exactness would mean the fork gained a true f32
/// path, and the `amx-f32` gate above should then be reconsidered rather than
/// left off on stale evidence.
#[cfg(all(test, feature = "std"))]
mod amx_probe {
    #[test]
    fn amx_f32_is_bf16_grade_not_exact_and_not_garbage() {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        {
            use ndarray::Array2;
            if !super::amx_available() {
                eprintln!("AMX unavailable");
                return;
            }
            for (m, k, n) in [
                (4usize, 4usize, 4usize),
                (16, 16, 16),
                (32, 64, 32),
                (17, 33, 15),
            ] {
                let a = Array2::from_shape_fn((m, k), |(i, j)| {
                    (((i * 7 + j * 3) % 97) as f32).sqrt() * 0.3137 - 1.0
                });
                let b = Array2::from_shape_fn((k, n), |(i, j)| {
                    (((i * 5 + j * 2) % 89) as f32).sqrt() * 0.2713 - 0.5
                });
                let mut got = Array2::<f32>::zeros((m, n));
                let ok = ndarray::hpc::amx_matmul::matmul_f32(a.view(), b.view(), got.view_mut())
                    .is_ok();
                let mut want = Array2::<f32>::zeros((m, n));
                ndarray::linalg::general_mat_mul(
                    1.0f32,
                    &a.view(),
                    &b.view(),
                    0.0f32,
                    &mut want.view_mut(),
                );
                let max_abs = got
                    .iter()
                    .zip(want.iter())
                    .map(|(g, w)| (g - w).abs())
                    .fold(0.0f32, f32::max);
                let scale = want
                    .iter()
                    .map(|w| w.abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-6);
                let rel = max_abs / scale;
                eprintln!("{m}x{k}x{n}: amx_ok={ok} max_abs_err={max_abs:.6} rel={rel:.6}");
                assert!(ok, "{m}x{k}x{n}: matmul_f32 refused");
                assert!(
                    rel < 5e-3,
                    "{m}x{k}x{n}: rel {rel:.6} is worse than bf16 — tile/shape bug, not rounding"
                );
                assert!(
                    rel > 1e-5,
                    "{m}x{k}x{n}: rel {rel:.6} is f32-exact — the fork may have a real f32 path now; re-evaluate the `amx-f32` gate"
                );
            }
        }
    }
}
