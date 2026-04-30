//! Single dispatch point for SIMD / AMX / MKL acceleration through `ndarray`.
//!
//! Today this module forwards everything to the existing `ndarray::linalg`
//! and elementwise paths (which are themselves backed by `matrixmultiply` and
//! the macerator-based SIMD layer in `crate::ops::simd`). When the workspace
//! `ndarray` dependency is switched to `AdaWorldAPI/ndarray`, swap the bodies
//! below to call `ndarray::simd::ops::*`, `ndarray::hpc::amx_matmul::*`, and
//! `ndarray::backend::*` — no other file in `burn-ndarray` should need to
//! change.
//!
//! Feature flags:
//! - `simd-runtime-dispatch`: route through `ndarray::simd` LazyLock<Tier>.
//! - `simd-amx`: enable Intel AMX matmul fast path (x86_64 Linux only).
//! - `mkl`: route GEMM through Intel MKL via ndarray's CBLAS backend.

use crate::{NdArrayElement, SharedArray};
use burn_backend::{ElementConversion, Shape};
use ndarray::s;

/// Returns true when the running CPU has Intel AMX hardware enabled by the OS.
///
/// Today: always `false`. When the fork lands, this should call
/// `ndarray::simd_amx::amx_available()` which performs the full CPUID +
/// XCR0 + Linux prctl(ARCH_REQ_XCOMP_PERM) check.
#[inline]
#[allow(dead_code)]
pub fn amx_available() -> bool {
    #[cfg(all(feature = "simd-amx", target_arch = "x86_64", target_os = "linux"))]
    {
        // TODO: replace with `ndarray::simd_amx::amx_available()` once the
        // workspace ndarray points at the AdaWorldAPI fork.
        false
    }
    #[cfg(not(all(feature = "simd-amx", target_arch = "x86_64", target_os = "linux")))]
    {
        false
    }
}

/// Returns the active SIMD tier description for diagnostics.
///
/// When the fork lands, mirror `ndarray::simd::Tier` here.
#[inline]
#[allow(dead_code)]
pub fn simd_tier() -> &'static str {
    #[cfg(feature = "simd-runtime-dispatch")]
    {
        // TODO: re-export `ndarray::simd::tier_name()` once available.
        "compile-time"
    }
    #[cfg(not(feature = "simd-runtime-dispatch"))]
    {
        "compile-time"
    }
}

/// Single-batch matmul dispatcher.
///
/// Decision tree (top-down, first match wins):
///   1. `simd-amx` + `amx_available()` + f32 + shapes are AMX-friendly → AMX tile.
///   2. `mkl` + f32/f64 → CBLAS sgemm/dgemm via ndarray's `blas` feature.
///   3. Default: `ndarray::linalg::general_mat_mul` (scalar / matrixmultiply).
///
/// Today only branch (3) is active. Branches (1) and (2) are stubs that will
/// be filled in when the AdaWorldAPI/ndarray fork is wired into the workspace.
pub(crate) fn matmul_2d<E: NdArrayElement>(
    lhs: ndarray::ArrayView2<'_, E>,
    rhs: ndarray::ArrayView2<'_, E>,
    out: &mut ndarray::ArrayViewMut2<'_, E>,
) {
    let alpha: E = 1.0.elem();
    let beta: E = 0.0.elem();

    // (1) AMX fast path — x86_64 Linux only, f32 only, shapes that map to
    //     16x16 tiles. TODO: dispatch when fork lands.
    #[cfg(all(feature = "simd-amx", target_arch = "x86_64", target_os = "linux"))]
    {
        if amx_available() && core::any::TypeId::of::<E>() == core::any::TypeId::of::<f32>() {
            // TODO: ndarray::hpc::amx_matmul::matmul_f32_amx(lhs, rhs, out);
            //       return;
        }
    }

    // (2) MKL fast path. ndarray's `blas` feature already routes
    //     `general_mat_mul` through CBLAS when the matching `blas-*` features
    //     are also enabled on burn-ndarray. The dispatch below is a no-op
    //     placeholder for future direct sgemm/dgemm calls.
    #[cfg(feature = "mkl")]
    {
        // TODO: explicit MKL sgemm/dgemm call once the fork exposes a Rust API.
    }

    // (3) Default path — unchanged from upstream behavior.
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
    strides
        .iter()
        .zip(index.iter())
        .map(|(s, i)| s * i)
        .sum()
}
