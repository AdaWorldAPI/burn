# Burn + RustyNum Backend Integration Plan

**Date:** 2026-02-25
**Status:** PLAN — read-only exploration complete, no code changes yet

---

## 1. Executive Summary

This document specifies how to implement `burn::Backend` backed by rustynum's AVX-512 SIMD/VNNI/BF16 microkernels, replacing the generic ndarray/OpenBLAS path. The goal is a single-binary Cognitive Engine where burn neural networks execute inside rustynum's cache-blocked loops, with zero-copy data flow from CogRecord containers on the Blackboard.

**Key architectural decisions harvested from upstream repos:**

| Source | Pattern Harvested | Applied Where |
|--------|------------------|---------------|
| **burn-ndarray** | `NdArrayTensor` enum + `execute_with_dtype!` macro dispatch | `RustyNumTensor` enum design |
| **candle** | Minimal `Storage` + contiguous/strided duality | Contiguity enforcement in rustynum tensor primitive |
| **ort** | `_backing: Box<dyn Any>` guard for zero-copy lifetime management | Blackboard → Burn tensor path |
| **polars** | Arrow-backed `ChunkedArray` with zero-copy `rechunk()` | DataFusion → Burn input pipeline |

---

## 2. Burn Backend Trait Anatomy (Extracted)

The `Backend` trait (`burn-backend/src/backend/base.rs:65-163`) requires:

```
FloatTensorOps<Self>     — 80+ methods (matmul, add, reshape, etc.)
BoolTensorOps<Self>      — boolean tensor operations
IntTensorOps<Self>       — integer tensor operations
ModuleOps<Self>          — conv2d, pool, attention, etc.
ActivationOps<Self>      — relu, gelu, sigmoid, etc. (has defaults)
QTensorOps<Self>         — quantize/dequantize
TransactionOps<Self>     — batch read/write (has defaults)
```

**Associated types we must define:**

```rust
type Device = RustyNumDevice;                    // CPU-only (for now)
type FloatTensorPrimitive = RustyNumTensor;      // Wraps rustynum data
type FloatElem = f32;                            // Default float type
type IntTensorPrimitive = RustyNumTensor;        // Same enum, int variants
type IntElem = i64;                              // Default int type
type BoolTensorPrimitive = RustyNumTensor;       // Bool variant
type BoolElem = bool;
type QuantizedTensorPrimitive = RustyNumQTensor; // INT8/BF16 quantized
```

---

## 3. New Crate: `burn-rustynum`

### 3.1 Crate Location

```
burn/crates/burn-rustynum/
├── Cargo.toml
└── src/
    ├── lib.rs          — RustyNum backend struct + Backend impl
    ├── tensor.rs       — RustyNumTensor enum + TensorMetadata
    ├── storage.rs      — Zero-copy bridge to Blackboard
    ├── element.rs      — Element trait impls for rustynum types
    ├── ops/
    │   ├── mod.rs
    │   ├── tensor.rs   — FloatTensorOps (matmul, add, reshape, etc.)
    │   ├── int.rs      — IntTensorOps
    │   ├── bool.rs     — BoolTensorOps
    │   ├── module.rs   — ModuleOps (conv, pool via im2col+GEMM)
    │   ├── activation.rs — ActivationOps (VML-backed sigmoid, etc.)
    │   └── qtensor.rs  — QTensorOps (INT8 VNNI, BF16)
    └── parallel.rs     — rayon integration for batch dims
```

### 3.2 Dependencies

```toml
[dependencies]
burn-backend = { path = "../burn-backend" }
rustynum-rs = { path = "../../../../rustynum/rustynum-rs" }
rustynum-core = { path = "../../../../rustynum/rustynum-core" }
rustyblas = { path = "../../../../rustynum/rustyblas" }
rustymkl = { path = "../../../../rustynum/rustymkl" }

[features]
default = ["avx512"]
avx512 = ["rustynum-core/avx512"]
avx2 = ["rustynum-core/avx2"]
mkl = ["rustynum-core/mkl"]         # Optional real-MKL FFI
libxsmm = ["rustynum-core/libxsmm"] # Optional LIBXSMM JIT
```

---

## 4. Tensor Primitive Design

### 4.1 Core Type

Inspired by `NdArrayTensor` enum pattern (burn-ndarray/src/tensor.rs:23-35):

```rust
/// Tensor primitive backed by rustynum flat arrays + shape metadata.
#[derive(Debug, Clone)]
pub enum RustyNumTensor {
    F32 { data: Arc<Vec<f32>>, shape: Vec<usize>, strides: Vec<usize> },
    F64 { data: Arc<Vec<f64>>, shape: Vec<usize>, strides: Vec<usize> },
    I64 { data: Arc<Vec<i64>>, shape: Vec<usize>, strides: Vec<usize> },
    I32 { data: Arc<Vec<i32>>, shape: Vec<usize>, strides: Vec<usize> },
    U8  { data: Arc<Vec<u8>>,  shape: Vec<usize>, strides: Vec<usize> },
    Bool { data: Arc<Vec<bool>>, shape: Vec<usize>, strides: Vec<usize> },
    /// Zero-copy view into a Blackboard allocation.
    /// The `_backing` guard keeps the Blackboard alive.
    BlackboardView {
        ptr: *const u8,
        len: usize,
        dtype: DType,
        shape: Vec<usize>,
        strides: Vec<usize>,
        _backing: Arc<dyn Any + Send + Sync>,
    },
}
```

**Key design decisions:**

1. **`Arc<Vec<T>>`** for COW semantics (same as ndarray's `ArcArray`). Mutations check `Arc::get_mut()` — if unique, mutate in-place; if shared, clone-then-mutate.

2. **`BlackboardView`** variant enables zero-copy from rustynum-core's `Blackboard` allocator. The `_backing: Arc<dyn Any>` pattern is directly from ort (`/home/user/ort/src/value/mod.rs:60`).

3. **Contiguous-only** for V1. Rustynum's SIMD kernels require contiguous memory. Strides are metadata-only (for reshape/transpose tracking) and any non-contiguous operation triggers an explicit `to_contiguous()` copy. This mirrors candle's approach where the CPU backend enforces contiguous layout at kernel boundaries.

### 4.2 Dispatch Macro

Following burn-ndarray's `execute_with_dtype!` pattern:

```rust
macro_rules! rustynum_dispatch {
    ($tensor:expr, $elem:ident, $op:expr) => {
        match $tensor {
            RustyNumTensor::F32 { data, shape, strides } => {
                type $elem = f32;
                $op(data, shape, strides)
            }
            RustyNumTensor::F64 { data, shape, strides } => {
                type $elem = f64;
                $op(data, shape, strides)
            }
            // ... other variants
        }
    };
}
```

---

## 5. GEMM Call Path (The Critical Path)

### 5.1 Current ndarray Path (What We Replace)

```
Tensor::matmul()
  → B::float_matmul(lhs, rhs)           [burn-backend/src/backend/ops/tensor.rs:337]
    → matmul(lhs_arr, rhs_arr)           [burn-ndarray/src/ops/matmul.rs:9]
      → ndarray::linalg::general_mat_mul  [calls OpenBLAS sgemm/dgemm]
```

### 5.2 RustyNum Path (Target)

```
Tensor::matmul()
  → B::float_matmul(lhs, rhs)
    → rustynum_matmul(lhs, rhs)
      → ensure_contiguous(lhs), ensure_contiguous(rhs)
      → broadcast_batch_dims(lhs_shape, rhs_shape)
      → for each batch:
          → rustyblas::sgemm(...)           [6x16 AVX-512 microkernel]
          OR if BF16 dtype:
          → rustyblas::bf16_gemm(...)       [vdpbf16ps mixed-precision]
          OR if quantized:
          → rustyblas::int8_gemm_vnni(...)  [vpdpbusd + per-channel scale]
```

### 5.3 Implementation Steps for float_matmul

1. Extract `data`, `shape`, `strides` from both tensors
2. Ensure contiguous layout (copy if needed)
3. Compute broadcast batch dimensions (reuse burn-ndarray's `output_shape()` logic)
4. Allocate output buffer: `vec![0.0f32; batch_size * m * n]`
5. Loop over batches, calling `rustyblas::level3::sgemm()` with:
   - `order = RowMajor`, `transa = NoTrans`, `transb = NoTrans`
   - `alpha = 1.0`, `beta = 0.0`
   - Direct pointer into batch slice
6. Wrap output in `RustyNumTensor::F32 { ... }`

---

## 6. QTensorOps: Quantized Path (INT8 VNNI + BF16)

### 6.1 Quantization Storage

```rust
#[derive(Debug, Clone)]
pub struct RustyNumQTensor {
    /// Quantized data (i8 or u8 for INT8, u16 for BF16)
    pub data: RustyNumTensor,
    /// Quantization scheme
    pub scheme: QuantScheme,
    /// Per-tensor or per-channel scale factors
    pub scales: Vec<f32>,
    /// Zero points (for asymmetric quantization)
    pub zero_points: Vec<i32>,
}
```

### 6.2 INT8 VNNI GEMM Integration

```rust
fn quantize(tensor: FloatTensor<Self>, scheme: &QuantScheme, qparams: ...) -> QuantizedTensor<Self> {
    // 1. Extract f32 data
    // 2. Compute per-tensor scale = max(abs(data)) / 127
    // 3. Quantize: i8_data[i] = round(f32_data[i] / scale)
    // 4. Return RustyNumQTensor with i8 data + scale
}

fn qmatmul(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> FloatTensor<Self> {
    // Calls rustyblas::int8_gemm_vnni_512() or _256() based on CPU caps
    // Returns f32 output (already dequantized by per-channel scales)
}
```

### 6.3 BF16 GEMM Path

```rust
// When Burn creates a BF16 tensor:
fn float_matmul_bf16(lhs_bf16: &[u16], rhs_bf16: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Calls rustyblas::bf16_gemm::bf16_gemm_avx512()
    // Uses vdpbf16ps for mixed-precision accumulation
}
```

---

## 7. Zero-Copy Blackboard → Burn Tensor Path

### 7.1 The Problem

CogRecord containers sit on rustynum-core's Blackboard (64-byte aligned arena). Burn needs to read embeddings from these containers without copying into a new allocation.

### 7.2 The Solution (Inspired by ort's `_backing` Guard)

```rust
/// Create a Burn tensor that borrows directly from a Blackboard allocation.
pub fn tensor_from_blackboard(
    blackboard: Arc<Blackboard>,
    offset: usize,
    shape: Vec<usize>,
    dtype: DType,
) -> RustyNumTensor {
    let total_bytes = shape.iter().product::<usize>() * dtype.size();
    let ptr = blackboard.as_ptr().add(offset);

    RustyNumTensor::BlackboardView {
        ptr,
        len: total_bytes,
        dtype,
        shape: shape.clone(),
        strides: compute_c_strides(&shape),
        _backing: blackboard,  // Keeps Blackboard alive
    }
}
```

**Mutating a BlackboardView** triggers copy-on-write: the data is copied to a new `Arc<Vec<T>>` and the variant changes from `BlackboardView` to the owned variant.

### 7.3 CogRecord → Burn Tensor Extraction

```rust
/// Extract a specific field from a CogRecord as a Burn tensor.
/// Zero-copy when the CogRecord lives on the Blackboard.
pub fn cogrecord_embedding_to_tensor(
    cogrecord: &CogRecord,
    blackboard: Arc<Blackboard>,
) -> RustyNumTensor {
    // CogRecord stores 1024-D f32 embedding at known offset
    let embed_offset = cogrecord.embed_offset();
    let embed_dim = cogrecord.embed_dim();
    tensor_from_blackboard(blackboard, embed_offset, vec![embed_dim], DType::F32)
}
```

---

## 8. PackedQualia Hydration Bridge (Continuous ↔ Discrete)

### 8.1 The Bridge Function

The `hydrate_qualia_to_bf16()` function (to be added to `rustynum-core/src/bf16_hamming.rs`) converts an 18-byte packed phenomenological state into 16 continuous BF16 values in a single AVX-512 pass.

### 8.2 Burn Integration Point

```rust
/// Convert a batch of PackedQualia into a Burn tensor ready for the forward pass.
pub fn qualia_batch_to_tensor(
    qualia_batch: &[PackedQualia],
    device: &RustyNumDevice,
) -> FloatTensor<RustyNum> {
    let batch_size = qualia_batch.len();
    let mut out_bf16 = vec![0u16; batch_size * 16]; // 16 dimensions per qualia

    for (i, q) in qualia_batch.iter().enumerate() {
        unsafe {
            hydrate_qualia_to_bf16(
                q as *const PackedQualia,
                out_bf16[i * 16..].as_mut_ptr(),
            );
        }
    }

    // Return as BF16 tensor for BF16 GEMM path, or widen to f32
    RustyNumTensor::from_bf16(out_bf16, vec![batch_size, 16])
}
```

### 8.3 SIMD Thresholding (Continuous → Discrete)

When the burn model outputs a continuous embedding, convert to a binary hypervector:

```rust
/// Threshold a continuous Burn tensor output into a discrete 16K-bit hypervector.
/// Uses rustynum's AVX-512 comparison to avoid heap allocation.
pub fn threshold_to_hypervector(
    tensor: FloatTensor<RustyNum>,  // e.g., [1, 16384] f32
) -> Vec<u8> {
    let data = extract_f32_data(&tensor);
    // rustynum_core::simd::f32_to_binary_avx512(data, threshold=0.0)
    // Returns packed bits: 16384 / 8 = 2048 bytes
    rustynum_core::simd::threshold_f32_to_bits(&data, 0.0)
}
```

---

## 9. DataFusion → Burn Input Pipeline (Polars-Inspired)

### 9.1 Architecture

```
DataFusion SQL Query
  → RecordBatch (Arrow columnar)
    → arrow_to_rustynum_tensor()  [zero-copy via Arrow buffer reference]
      → RustyNumTensor::BlackboardView or RustyNumTensor::F32
        → Burn Tensor input
```

### 9.2 Arrow Buffer → RustyNum Tensor

Using the existing `rustynum-arrow` crate's bridge:

```rust
/// Convert an Arrow Float32Array column into a Burn-compatible RustyNum tensor.
/// Zero-copy when the Arrow buffer is properly aligned.
pub fn arrow_column_to_tensor(
    column: &Float32Array,
    target_shape: Vec<usize>,
) -> RustyNumTensor {
    let buffer = column.values();
    let slice: &[f32] = buffer.typed_data::<f32>();

    // If aligned to 64 bytes, we can reference directly
    if (slice.as_ptr() as usize) % 64 == 0 {
        RustyNumTensor::BlackboardView {
            ptr: slice.as_ptr() as *const u8,
            len: slice.len() * 4,
            dtype: DType::F32,
            shape: target_shape.clone(),
            strides: compute_c_strides(&target_shape),
            _backing: Arc::new(buffer.clone()), // Arrow buffer keeps data alive
        }
    } else {
        // Fallback: copy to aligned buffer
        RustyNumTensor::F32 {
            data: Arc::new(slice.to_vec()),
            shape: target_shape,
            strides: compute_c_strides(&target_shape),
        }
    }
}
```

---

## 10. Cranelift JIT Planner Hook

### 10.1 Role in the Architecture

The Cranelift JIT (already in rustynum via the wasmtime fork) compiles data-dependent query plans at runtime:

```
SQL WHERE clause → Cranelift IR → Native code
  → Scans Blackboard PackedQualia array
  → Applies filter predicates as immediate bitmask checks
  → For surviving records, calls hydrate_qualia_to_bf16()
  → Hands BF16 chunks directly to burn tensor input buffer
```

### 10.2 Integration with JITSON

The `rustynum-core/src/jitson.rs` engine already generates AVX-512 pipeline descriptors from JSON/YAML templates. The burn integration extends this:

```json
{
  "pipeline": "qualia_inference",
  "stages": [
    { "type": "scan", "source": "blackboard", "filter": "gestalt == 'I' AND volition > 0" },
    { "type": "hydrate", "kernel": "hydrate_qualia_to_bf16" },
    { "type": "burn_forward", "model": "qualia_classifier", "dtype": "bf16" },
    { "type": "threshold", "output": "hypervector_16k" }
  ]
}
```

---

## 11. Implementation Phases

### Phase 1: Skeleton Backend (Week 1)
- [ ] Create `burn-rustynum` crate with `Backend` impl
- [ ] Implement `FloatTensorOps` with rustynum GEMM for matmul
- [ ] Implement basic element-wise ops (add, sub, mul, div) via SIMD dispatch
- [ ] Implement reshape, transpose, slice, permute
- [ ] Pass `burn-backend-tests` float tensor test suite

### Phase 2: Full Ops Coverage (Week 2)
- [ ] Implement `IntTensorOps` and `BoolTensorOps`
- [ ] Implement `ActivationOps` (sigmoid via VML, relu via SIMD max)
- [ ] Implement `ModuleOps` (conv2d via im2col + sgemm, pooling)
- [ ] Implement `TransactionOps` defaults
- [ ] Pass full `burn-backend-tests` suite

### Phase 3: Quantized Path (Week 3)
- [ ] Implement `QTensorOps` with INT8 VNNI quantization
- [ ] Add BF16 GEMM path for mixed-precision inference
- [ ] Add `PackedQualia` hydration bridge
- [ ] Add SIMD thresholding (continuous → discrete)

### Phase 4: Zero-Copy Integration (Week 4)
- [ ] Implement `BlackboardView` tensor variant
- [ ] Add CogRecord → Burn tensor zero-copy path
- [ ] Add Arrow → Burn tensor zero-copy path
- [ ] Integrate with DataFusion query pipeline
- [ ] Add Cranelift JIT → Burn forward pass hook

### Phase 5: Autodiff (Optional, Week 5+)
- [ ] Implement `AutodiffBackend` via `burn-autodiff` wrapper
- [ ] Enable training on rustynum backend
- [ ] Backprop through quantized layers

---

## 12. FloatTensorOps Method Inventory

Methods requiring implementation (extracted from `burn-backend/src/backend/ops/tensor.rs`):

### Required (no default impl):
| Method | RustyNum Kernel | Priority |
|--------|----------------|----------|
| `float_from_data` | `TensorData` → `RustyNumTensor` | P0 |
| `float_into_data` | `RustyNumTensor` → `TensorData` | P0 |
| `float_device` | Always `RustyNumDevice::Cpu` | P0 |
| `float_to_device` | No-op for CPU | P0 |
| `float_into_int` | Element-wise cast | P0 |
| `float_empty` | `vec![0.0; n]` | P0 |
| `float_add` | `SimdOps::add` | P0 |
| `float_sub` | `SimdOps::sub` | P0 |
| `float_mul_scalar` | `SimdOps::scale` | P0 |
| `float_div` | Element-wise SIMD | P0 |
| `float_matmul` | `rustyblas::sgemm` | **P0 (critical)** |
| `float_random` | `rustynum_core::rng` | P0 |
| `float_reshape` | Shape metadata change | P0 |
| `float_swap_dims` | Transpose/permute | P0 |
| `float_gather` | Index-based gather | P1 |
| `float_scatter` | Index-based scatter | P1 |
| `float_select` | Dim-based selection | P1 |
| `float_slice` | Contiguous subrange | P0 |
| `float_equal` | Element-wise compare | P1 |
| `float_argmax` | SIMD reduction | P1 |
| `float_argmin` | SIMD reduction | P1 |
| `float_sum` | `SimdOps::sum` | P0 |
| `float_sum_dim` | Axis reduction | P1 |
| `float_mean_dim` | `SimdOps::sum` / n | P1 |
| `float_exp` | `rustymkl::vml::vsexp` | P1 |
| `float_log` | `rustymkl::vml::vsln` | P1 |
| `float_powf_scalar` | `rustymkl::vml::vspowx` | P1 |
| `float_sqrt` | `rustymkl::vml::vssqrt` | P1 |
| `float_abs` | SIMD bitwise AND mask | P1 |
| `float_neg` | SIMD XOR sign bit | P1 |
| `float_cat` | Concatenation | P1 |
| `float_clamp` | SIMD min/max | P1 |
| `float_into_bool` | Threshold compare | P2 |

### Have default implementations (override for performance):
| Method | Worth Overriding? |
|--------|------------------|
| `float_zeros` | No (default is fine) |
| `float_ones` | No |
| `float_full` | No |
| `float_remainder` | Maybe (SIMD fmod) |
| `float_prod` | Maybe (SIMD reduction) |

---

## 13. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| ModuleOps complexity (conv2d, attention) | High | Start with im2col → GEMM; attention = batched matmul + softmax |
| Contiguous-only limitation | Medium | Document clearly; add `to_contiguous()` at kernel boundaries |
| burn API churn (upstream updates) | Medium | Pin to upstream commit; periodic rebase |
| MSRV conflict (burn uses stable, rustynum needs nightly) | Low | `burn-rustynum` compiled with nightly; other burn crates stay stable |
| Thread safety of BlackboardView | Medium | `Arc<dyn Any + Send + Sync>` enforces it; raw pointer is immutable |

---

## 14. Success Criteria

1. `cargo test -p burn-rustynum` passes all `burn-backend-tests` (400+ tests)
2. GEMM throughput matches or exceeds rustyblas standalone benchmarks (138+ GFLOPS at 1024x1024)
3. Zero-copy CogRecord → Burn tensor path verified with valgrind (no unnecessary allocations)
4. INT8 VNNI inference path functional with per-channel dequantization
5. End-to-end: DataFusion SQL → filter → hydrate → burn forward → threshold → hypervector

---

## Appendix A: Upstream References

- **Burn Backend trait:** `/home/user/burn/crates/burn-backend/src/backend/base.rs:65-163`
- **Burn FloatTensorOps:** `/home/user/burn/crates/burn-backend/src/backend/ops/tensor.rs:13-337`
- **NdArray Backend impl:** `/home/user/burn/crates/burn-ndarray/src/backend.rs:60-132`
- **NdArray matmul:** `/home/user/burn/crates/burn-ndarray/src/ops/matmul.rs:9-70`
- **NdArray tensor:** `/home/user/burn/crates/burn-ndarray/src/tensor.rs:23-35`
- **Ort zero-copy backing:** `/home/user/ort/src/value/mod.rs:54-89`
- **Ort TensorRef::from_array_view:** `/home/user/ort/src/value/impl_tensor/create.rs:214-221`
- **RustyNum BLAS GEMM:** `/home/user/rustynum/rustyblas/src/level3.rs` (6x16 f32 microkernel)
- **RustyNum INT8 VNNI:** `/home/user/rustynum/rustyblas/src/int8_gemm.rs`
- **RustyNum BF16 GEMM:** `/home/user/rustynum/rustyblas/src/bf16_gemm.rs`
- **RustyNum Blackboard:** `/home/user/rustynum/rustynum-core/src/blackboard.rs`
- **RustyNum VML:** `/home/user/rustynum/rustymkl/src/vml.rs`

## Appendix B: Agent Directive (For Future Sessions)

```
@Claude, we are building the ultimate "One-Binary" Cognitive Engine.
We are integrating our burn fork directly into ladybug-rs, using rustynum
as the absolute bare-metal hardware execution backend.

Your task is to implement the burn-rustynum backend crate following the
plan in docs/RUSTYNUM_BACKEND_PLAN.md.

Directives:
1. The rustynum Backend for Burn (No Generic BLAS):
   Do not let burn default to generic CPU NdArray backends. Implement
   burn::Backend where GEMM directly calls rustynum's AVX-512 vpdpbusd
   (INT8 VNNI) and vdpbf16ps (BF16) microkernels.

2. The Upstream Harvest (candle, ort, polars):
   - From candle: Minimalist tensor contraction with rustynum AVX-512 popcount
   - From ort: Pre-allocated arena memory via Blackboard _backing guard
   - From polars: Arrow-backed chunking for DataFusion → tensor input

3. Continuous to Discrete Bridge:
   When burn outputs a continuous tensor, pass through rustynum's SIMD
   thresholding to become a discrete 16K-bit hypervector. No intermediate
   heap allocations.
```
