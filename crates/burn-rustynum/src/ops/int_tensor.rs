use alloc::vec;
use alloc::vec::Vec;

use burn_backend::backend::ExecutionError;
use burn_backend::ops::IntTensorOps;
use burn_backend::tensor::IntTensor;
use burn_backend::{Distribution, Scalar, Shape, TensorData};
use burn_std::tensor::Slice;
use burn_std::IntDType;

use crate::backend::{RustyNum, RustyNumDevice, SEED};
use crate::tensor::{RustyNumTensor, c_strides};

use rand::rngs::StdRng;
use rand::SeedableRng;

impl IntTensorOps<Self> for RustyNum {
    fn int_empty(shape: Shape, _device: &RustyNumDevice, dtype: IntDType) -> IntTensor<Self> {
        RustyNumTensor::zeros(shape.to_vec(), dtype.into())
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn int_from_data(data: TensorData, _device: &RustyNumDevice) -> IntTensor<Self> {
        RustyNumTensor::from_data(data)
    }

    fn int_device(_tensor: &IntTensor<Self>) -> RustyNumDevice {
        RustyNumDevice::Cpu
    }

    fn int_to_device(tensor: IntTensor<Self>, _device: &RustyNumDevice) -> IntTensor<Self> {
        tensor
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        tensor.reshape(shape.to_vec())
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        int_slice_impl(&tensor, slices)
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        int_slice_assign_impl(tensor, slices, &value)
    }

    fn int_into_float(tensor: IntTensor<Self>) -> <Self as burn_backend::Backend>::FloatTensorPrimitive {
        tensor.cast_to(burn_backend::DType::F32)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: <Self as burn_backend::Backend>::BoolTensorPrimitive,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        int_mask_where_impl(&tensor, &mask, &value)
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: <Self as burn_backend::Backend>::BoolTensorPrimitive,
        value: Scalar,
    ) -> IntTensor<Self> {
        int_mask_fill_impl(&tensor, &mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        int_gather_impl(&tensor, dim, &indices)
    }

    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        int_scatter_add_impl(tensor, dim, &indices, &value)
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        int_select_impl(&tensor, dim, &indices)
    }

    fn int_select_add(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        int_select_add_impl(tensor, dim, &indices, &value)
    }

    fn int_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_op(&lhs, &rhs, |a, b| a == b)
    }

    fn int_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_elem_op(&lhs, rhs, |a, s| a == s)
    }

    fn int_greater(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_op(&lhs, &rhs, |a, b| a > b)
    }

    fn int_greater_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_elem_op(&lhs, rhs, |a, s| a > s)
    }

    fn int_greater_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_op(&lhs, &rhs, |a, b| a >= b)
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_elem_op(&lhs, rhs, |a, s| a >= s)
    }

    fn int_lower(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_op(&lhs, &rhs, |a, b| a < b)
    }

    fn int_lower_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_elem_op(&lhs, rhs, |a, s| a < s)
    }

    fn int_lower_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_op(&lhs, &rhs, |a, b| a <= b)
    }

    fn int_lower_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
    ) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_int_elem_op(&lhs, rhs, |a, s| a <= s)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a + b)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a + s)
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a - b)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a - s)
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a * b)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a * s)
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a / b)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a / s)
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a % b)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a % s)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        int_matmul_impl(&lhs, &rhs)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let data = tensor.as_i64();
        let sum: i64 = data.iter().sum();
        RustyNumTensor::from_i64(vec![sum], vec![1])
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        int_reduce_dim_impl(&tensor, dim, 0i64, |acc, x| acc + x)
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let data = tensor.as_i64();
        let prod: i64 = data.iter().product();
        RustyNumTensor::from_i64(vec![prod], vec![1])
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        int_reduce_dim_impl(&tensor, dim, 1i64, |acc, x| acc * x)
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let dim_size = tensor.shape[dim] as i64;
        let sum = int_reduce_dim_impl(&tensor, dim, 0i64, |acc, x| acc + x);
        scalar_int_op(&sum, Scalar::Int(dim_size), |a, s| a / s)
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        int_cum_op_impl(&tensor, dim, 0i64, |acc, x| acc + x)
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        int_cum_op_impl(&tensor, dim, 1i64, |acc, x| acc * x)
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        int_cum_op_impl(&tensor, dim, i64::MAX, |acc, x| acc.min(x))
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        int_cum_op_impl(&tensor, dim, i64::MIN, |acc, x| acc.max(x))
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        int_argmax_impl(&tensor, dim)
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        int_argmin_impl(&tensor, dim)
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let data = tensor.as_i64();
        let out: Vec<i64> = data.iter().map(|&x| x.abs()).collect();
        RustyNumTensor::from_i64(out, tensor.shape.clone())
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        if dim1 == dim2 {
            return tensor;
        }
        let mut axes: Vec<usize> = (0..tensor.shape.len()).collect();
        axes.swap(dim1, dim2);
        int_permute_impl(tensor, &axes)
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        int_permute_impl(tensor, axes)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        int_flip_impl(tensor, axes)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        int_expand_impl(tensor, shape)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        _device: &RustyNumDevice,
    ) -> IntTensor<Self> {
        let mut seed = SEED.lock().unwrap();
        let mut rng = seed.take().unwrap_or_else(|| StdRng::from_rng(&mut rand::rng()));
        let data = TensorData::random::<i64, _, _>(shape, distribution, &mut rng);
        *seed = Some(rng);
        RustyNumTensor::from_data(data)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a & b)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a & s)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a | b)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a | s)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a ^ b)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a ^ s)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let data = tensor.as_i64();
        let out: Vec<i64> = data.iter().map(|&x| !x).collect();
        RustyNumTensor::from_i64(out, tensor.shape.clone())
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a << b)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a << s)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int_op(&lhs, &rhs, |a, b| a >> b)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        scalar_int_op(&lhs, rhs, |a, s| a >> s)
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        tensor.cast_to(dtype.into())
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        int_unfold_impl(&tensor, dim, size, step)
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Element-wise binary int operation.
fn binary_int_op(lhs: &RustyNumTensor, rhs: &RustyNumTensor, op: impl Fn(i64, i64) -> i64) -> RustyNumTensor {
    let lhs_data = lhs.as_i64();
    let rhs_data = rhs.as_i64();

    if lhs.shape == rhs.shape {
        let out: Vec<i64> = lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| op(a, b)).collect();
        RustyNumTensor::from_i64(out, lhs.shape.clone())
    } else {
        broadcast_binary_int_op(lhs, rhs, op)
    }
}

/// Broadcast two int tensors and apply a binary op.
fn broadcast_binary_int_op(lhs: &RustyNumTensor, rhs: &RustyNumTensor, op: impl Fn(i64, i64) -> i64) -> RustyNumTensor {
    let out_shape = broadcast_shape(&lhs.shape, &rhs.shape);
    let n: usize = out_shape.iter().product();
    let lhs_data = lhs.as_i64();
    let rhs_data = rhs.as_i64();
    let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
    let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);
    let out_strides = c_strides(&out_shape);

    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut lhs_idx = 0usize;
        let mut rhs_idx = 0usize;
        let mut remaining = i;
        for d in 0..out_shape.len() {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            lhs_idx += coord * lhs_strides[d];
            rhs_idx += coord * rhs_strides[d];
        }
        out[i] = op(lhs_data[lhs_idx], rhs_data[rhs_idx]);
    }
    RustyNumTensor::from_i64(out, out_shape)
}

/// Scalar int operation.
fn scalar_int_op(tensor: &RustyNumTensor, scalar: Scalar, op: impl Fn(i64, i64) -> i64) -> RustyNumTensor {
    let s: i64 = scalar.elem();
    let data = tensor.as_i64();
    let out: Vec<i64> = data.iter().map(|&x| op(x, s)).collect();
    RustyNumTensor::from_i64(out, tensor.shape.clone())
}

/// Compare two int tensors element-wise, producing a bool tensor.
fn compare_int_op(lhs: &RustyNumTensor, rhs: &RustyNumTensor, op: impl Fn(i64, i64) -> bool) -> RustyNumTensor {
    let lhs_data = lhs.as_i64();
    let rhs_data = rhs.as_i64();
    if lhs.shape == rhs.shape {
        let out: Vec<bool> = lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| op(a, b)).collect();
        RustyNumTensor::from_bool(out, lhs.shape.clone())
    } else {
        let out_shape = broadcast_shape(&lhs.shape, &rhs.shape);
        let n: usize = out_shape.iter().product();
        let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
        let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);
        let out_strides = c_strides(&out_shape);
        let mut out = vec![false; n];
        for i in 0..n {
            let mut li = 0usize;
            let mut ri = 0usize;
            let mut rem = i;
            for d in 0..out_shape.len() {
                let coord = rem / out_strides[d];
                rem %= out_strides[d];
                li += coord * lhs_strides[d];
                ri += coord * rhs_strides[d];
            }
            out[i] = op(lhs_data[li], rhs_data[ri]);
        }
        RustyNumTensor::from_bool(out, out_shape)
    }
}

/// Compare int tensor with scalar, producing a bool tensor.
fn compare_int_elem_op(tensor: &RustyNumTensor, scalar: Scalar, op: impl Fn(i64, i64) -> bool) -> RustyNumTensor {
    let s: i64 = scalar.elem();
    let data = tensor.as_i64();
    let out: Vec<bool> = data.iter().map(|&x| op(x, s)).collect();
    RustyNumTensor::from_bool(out, tensor.shape.clone())
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![0usize; max_rank];
    for i in 0..max_rank {
        let da = if i < max_rank - a.len() { 1 } else { a[i - (max_rank - a.len())] };
        let db = if i < max_rank - b.len() { 1 } else { b[i - (max_rank - b.len())] };
        result[i] = if da == db { da } else if da == 1 { db } else if db == 1 { da } else {
            panic!("Cannot broadcast shapes {:?} and {:?}", a, b);
        };
    }
    result
}

fn broadcast_strides(src_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let rank = out_shape.len();
    let src_rank = src_shape.len();
    let mut strides = vec![0usize; rank];
    let src_strides = c_strides(src_shape);

    for i in 0..rank {
        let src_i = i as isize - (rank as isize - src_rank as isize);
        if src_i >= 0 {
            let si = src_i as usize;
            if src_shape[si] == out_shape[i] {
                strides[i] = src_strides[si];
            } else {
                strides[i] = 0;
            }
        }
    }
    strides
}

/// Matrix multiply for i64 tensors: supports batched matmul.
fn int_matmul_impl(lhs: &RustyNumTensor, rhs: &RustyNumTensor) -> RustyNumTensor {
    let lhs_data = lhs.as_i64();
    let rhs_data = rhs.as_i64();

    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();
    assert!(lhs_rank >= 2 && rhs_rank >= 2, "matmul requires at least 2D tensors");

    let m = lhs.shape[lhs_rank - 2];
    let k = lhs.shape[lhs_rank - 1];
    let k2 = rhs.shape[rhs_rank - 2];
    let n = rhs.shape[rhs_rank - 1];
    assert_eq!(k, k2, "matmul inner dimensions must match: {} vs {}", k, k2);

    let lhs_batch: usize = lhs.shape[..lhs_rank - 2].iter().product();
    let rhs_batch: usize = rhs.shape[..rhs_rank - 2].iter().product();
    let batch = lhs_batch.max(rhs_batch);

    let mut out_shape = if lhs_rank > rhs_rank {
        lhs.shape[..lhs_rank - 2].to_vec()
    } else {
        rhs.shape[..rhs_rank - 2].to_vec()
    };
    out_shape.push(m);
    out_shape.push(n);

    let lhs_stride = m * k;
    let rhs_stride = k * n;
    let out_stride = m * n;

    let mut out = vec![0i64; batch * out_stride];

    for b in 0..batch {
        let lb = if lhs_batch == 1 { 0 } else { b };
        let rb = if rhs_batch == 1 { 0 } else { b };
        let a = &lhs_data[lb * lhs_stride..(lb + 1) * lhs_stride];
        let b_mat = &rhs_data[rb * rhs_stride..(rb + 1) * rhs_stride];
        let c = &mut out[b * out_stride..(b + 1) * out_stride];

        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += a_ip * b_mat[p * n + j];
                }
            }
        }
    }

    RustyNumTensor::from_i64(out, out_shape)
}

/// Reduce along a dimension for i64 tensors.
fn int_reduce_dim_impl(tensor: &RustyNumTensor, dim: usize, init: i64, op: impl Fn(i64, i64) -> i64) -> RustyNumTensor {
    let data = tensor.as_i64();
    let shape = &tensor.shape;

    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let mut out = vec![init; outer * inner];
    for o in 0..outer {
        for d in 0..dim_size {
            for i in 0..inner {
                let src_idx = o * dim_size * inner + d * inner + i;
                let dst_idx = o * inner + i;
                out[dst_idx] = op(out[dst_idx], data[src_idx]);
            }
        }
    }

    let mut new_shape = shape.clone();
    new_shape[dim] = 1;
    RustyNumTensor::from_i64(out, new_shape)
}

/// Cumulative operation along a dimension for i64 tensors.
fn int_cum_op_impl(tensor: &RustyNumTensor, dim: usize, init: i64, op: impl Fn(i64, i64) -> i64) -> RustyNumTensor {
    let data = tensor.as_i64();
    let shape = &tensor.shape;

    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let mut out = vec![0i64; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = o * dim_size * inner + d * inner + i;
                acc = op(acc, data[idx]);
                out[idx] = acc;
            }
        }
    }

    RustyNumTensor::from_i64(out, shape.clone())
}

/// Argmax along a dimension for i64 tensors.
fn int_argmax_impl(tensor: &RustyNumTensor, dim: usize) -> RustyNumTensor {
    let data = tensor.as_i64();
    let shape = &tensor.shape;
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = i64::MIN;
            for d in 0..dim_size {
                let val = data[o * dim_size * inner + d * inner + i];
                if val > best_val {
                    best_val = val;
                    best_idx = d;
                }
            }
            out[o * inner + i] = best_idx as i64;
        }
    }
    let mut new_shape = shape.clone();
    new_shape[dim] = 1;
    RustyNumTensor::from_i64(out, new_shape)
}

/// Argmin along a dimension for i64 tensors.
fn int_argmin_impl(tensor: &RustyNumTensor, dim: usize) -> RustyNumTensor {
    let data = tensor.as_i64();
    let shape = &tensor.shape;
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = i64::MAX;
            for d in 0..dim_size {
                let val = data[o * dim_size * inner + d * inner + i];
                if val < best_val {
                    best_val = val;
                    best_idx = d;
                }
            }
            out[o * inner + i] = best_idx as i64;
        }
    }
    let mut new_shape = shape.clone();
    new_shape[dim] = 1;
    RustyNumTensor::from_i64(out, new_shape)
}

/// Permute dimensions for i64 tensors.
fn int_permute_impl(tensor: RustyNumTensor, axes: &[usize]) -> RustyNumTensor {
    let data = tensor.as_i64();
    let old_shape = &tensor.shape;
    let rank = old_shape.len();
    let old_strides = c_strides(old_shape);

    let new_shape: Vec<usize> = axes.iter().map(|&a| old_shape[a]).collect();
    let new_strides = c_strides(&new_shape);
    let n = tensor.num_elements();

    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut old_idx = 0usize;
        let mut rem = i;
        for d in 0..rank {
            let coord = rem / new_strides[d];
            rem %= new_strides[d];
            old_idx += coord * old_strides[axes[d]];
        }
        out[i] = data[old_idx];
    }
    RustyNumTensor::from_i64(out, new_shape)
}

/// Flip along specified axes for i64 tensors.
fn int_flip_impl(tensor: RustyNumTensor, axes: &[usize]) -> RustyNumTensor {
    let data = tensor.as_i64();
    let shape = &tensor.shape;
    let strides = c_strides(shape);
    let n = tensor.num_elements();
    let rank = shape.len();

    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut src_idx = 0usize;
        let mut rem = i;
        for d in 0..rank {
            let coord = rem / strides[d];
            rem %= strides[d];
            let flipped_coord = if axes.contains(&d) {
                shape[d] - 1 - coord
            } else {
                coord
            };
            src_idx += flipped_coord * strides[d];
        }
        out[i] = data[src_idx];
    }
    RustyNumTensor::from_i64(out, shape.clone())
}

/// Slice tensor along multiple dimensions for i64 tensors.
fn int_slice_impl(tensor: &RustyNumTensor, slices: &[Slice]) -> RustyNumTensor {
    let data = tensor.as_i64();
    let shape = &tensor.shape;
    let rank = shape.len();
    let strides = c_strides(shape);

    let ranges: Vec<(usize, usize, usize)> = (0..rank).map(|d| {
        if d < slices.len() {
            let s = &slices[d];
            let dim_len = shape[d] as isize;
            let start = if s.start < 0 { (dim_len + s.start) as usize } else { s.start as usize };
            let end = match s.end {
                Some(e) => if e < 0 { (dim_len + e) as usize } else { e as usize },
                None => shape[d],
            };
            let step = s.step as usize;
            (start, end, step)
        } else {
            (0, shape[d], 1)
        }
    }).collect();

    let new_shape: Vec<usize> = ranges.iter().map(|&(s, e, step)| (e - s + step - 1) / step).collect();
    let n: usize = new_shape.iter().product();
    let new_strides = c_strides(&new_shape);

    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut src_idx = 0usize;
        let mut rem = i;
        for d in 0..rank {
            let coord = rem / new_strides[d];
            rem %= new_strides[d];
            let src_coord = ranges[d].0 + coord * ranges[d].2;
            src_idx += src_coord * strides[d];
        }
        out[i] = data[src_idx];
    }
    RustyNumTensor::from_i64(out, new_shape)
}

/// Assign values to a slice of a tensor for i64 tensors.
fn int_slice_assign_impl(tensor: RustyNumTensor, slices: &[Slice], value: &RustyNumTensor) -> RustyNumTensor {
    let shape = tensor.shape.clone();
    let rank = shape.len();
    let strides = c_strides(&shape);
    let val_data = value.as_i64();
    let val_strides = c_strides(&value.shape);

    let ranges: Vec<(usize, usize, usize)> = (0..rank).map(|d| {
        if d < slices.len() {
            let s = &slices[d];
            let dim_len = shape[d] as isize;
            let start = if s.start < 0 { (dim_len + s.start) as usize } else { s.start as usize };
            let end = match s.end {
                Some(e) => if e < 0 { (dim_len + e) as usize } else { e as usize },
                None => shape[d],
            };
            (start, end, s.step as usize)
        } else {
            (0, shape[d], 1)
        }
    }).collect();

    let val_shape: Vec<usize> = ranges.iter().map(|&(s, e, step)| (e - s + step - 1) / step).collect();
    let n: usize = val_shape.iter().product();

    // Clone the data for mutation
    let mut out_data: Vec<i64> = tensor.as_i64().to_vec();
    for i in 0..n {
        let mut dst_idx = 0usize;
        let mut rem = i;
        for d in 0..rank {
            let coord = rem / val_strides[d];
            rem %= val_strides[d];
            let dst_coord = ranges[d].0 + coord * ranges[d].2;
            dst_idx += dst_coord * strides[d];
        }
        out_data[dst_idx] = val_data[i];
    }
    RustyNumTensor::from_i64(out_data, shape)
}

/// Gather elements along a dimension for i64 tensors.
fn int_gather_impl(tensor: &RustyNumTensor, dim: usize, indices: &RustyNumTensor) -> RustyNumTensor {
    let data = tensor.as_i64();
    let idx_data = indices.as_i64();
    let shape = &tensor.shape;
    let strides = c_strides(shape);
    let idx_shape = &indices.shape;
    let idx_strides = c_strides(idx_shape);
    let n: usize = idx_shape.iter().product();

    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut src_idx = 0usize;
        let mut rem = i;
        for d in 0..idx_shape.len() {
            let coord = rem / idx_strides[d];
            rem %= idx_strides[d];
            if d == dim {
                src_idx += (idx_data[i] as usize) * strides[d];
            } else {
                src_idx += coord * strides[d];
            }
        }
        out[i] = data[src_idx];
    }
    RustyNumTensor::from_i64(out, idx_shape.clone())
}

/// Scatter add for i64 tensors.
fn int_scatter_add_impl(tensor: RustyNumTensor, dim: usize, indices: &RustyNumTensor, value: &RustyNumTensor) -> RustyNumTensor {
    let idx_data = indices.as_i64();
    let val_data = value.as_i64();
    let shape = tensor.shape.clone();
    let strides = c_strides(&shape);
    let idx_shape = &indices.shape;
    let idx_strides = c_strides(idx_shape);
    let n: usize = idx_shape.iter().product();

    let mut out_data: Vec<i64> = tensor.as_i64().to_vec();
    for i in 0..n {
        let mut dst_idx = 0usize;
        let mut rem = i;
        for d in 0..idx_shape.len() {
            let coord = rem / idx_strides[d];
            rem %= idx_strides[d];
            if d == dim {
                dst_idx += (idx_data[i] as usize) * strides[d];
            } else {
                dst_idx += coord * strides[d];
            }
        }
        out_data[dst_idx] += val_data[i];
    }
    RustyNumTensor::from_i64(out_data, shape)
}

/// Select along a dimension for i64 tensors.
fn int_select_impl(tensor: &RustyNumTensor, dim: usize, indices: &RustyNumTensor) -> RustyNumTensor {
    let data = tensor.as_i64();
    let idx_data = indices.as_i64();
    let shape = &tensor.shape;
    let strides = c_strides(shape);

    let num_indices = idx_data.len();
    let mut new_shape = shape.clone();
    new_shape[dim] = num_indices;
    let n: usize = new_shape.iter().product();
    let new_strides = c_strides(&new_shape);

    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut src_idx = 0usize;
        let mut rem = i;
        for d in 0..shape.len() {
            let coord = rem / new_strides[d];
            rem %= new_strides[d];
            if d == dim {
                src_idx += (idx_data[coord] as usize) * strides[d];
            } else {
                src_idx += coord * strides[d];
            }
        }
        out[i] = data[src_idx];
    }
    RustyNumTensor::from_i64(out, new_shape)
}

/// Select add for i64 tensors.
fn int_select_add_impl(tensor: RustyNumTensor, dim: usize, indices: &RustyNumTensor, value: &RustyNumTensor) -> RustyNumTensor {
    let idx_data = indices.as_i64();
    let val_data = value.as_i64();
    let shape = tensor.shape.clone();
    let strides = c_strides(&shape);

    let num_indices = idx_data.len();
    let mut val_shape = shape.clone();
    val_shape[dim] = num_indices;
    let val_strides = c_strides(&val_shape);
    let n: usize = val_shape.iter().product();

    let mut out_data: Vec<i64> = tensor.as_i64().to_vec();
    for i in 0..n {
        let mut dst_idx = 0usize;
        let mut rem = i;
        for d in 0..shape.len() {
            let coord = rem / val_strides[d];
            rem %= val_strides[d];
            if d == dim {
                dst_idx += (idx_data[coord] as usize) * strides[d];
            } else {
                dst_idx += coord * strides[d];
            }
        }
        out_data[dst_idx] += val_data[i];
    }
    RustyNumTensor::from_i64(out_data, shape)
}

/// Mask where for i64 tensors.
fn int_mask_where_impl(tensor: &RustyNumTensor, mask: &RustyNumTensor, value: &RustyNumTensor) -> RustyNumTensor {
    let data = tensor.as_i64();
    let mask_data = mask.as_bool();
    let val_data = value.as_i64();
    let out: Vec<i64> = data.iter().zip(mask_data.iter()).zip(val_data.iter())
        .map(|((&t, &m), &v)| if m { v } else { t })
        .collect();
    RustyNumTensor::from_i64(out, tensor.shape.clone())
}

/// Mask fill for i64 tensors.
fn int_mask_fill_impl(tensor: &RustyNumTensor, mask: &RustyNumTensor, value: Scalar) -> RustyNumTensor {
    let data = tensor.as_i64();
    let mask_data = mask.as_bool();
    let fill: i64 = value.elem();
    let out: Vec<i64> = data.iter().zip(mask_data.iter())
        .map(|(&t, &m)| if m { fill } else { t })
        .collect();
    RustyNumTensor::from_i64(out, tensor.shape.clone())
}

/// Expand (broadcast) tensor to a target shape for i64 tensors.
fn int_expand_impl(tensor: RustyNumTensor, shape: Shape) -> RustyNumTensor {
    let target = shape.to_vec();
    if tensor.shape == target {
        return tensor;
    }

    let data = tensor.as_i64();
    let src_strides = broadcast_strides(&tensor.shape, &target);
    let out_strides = c_strides(&target);
    let n: usize = target.iter().product();

    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut src_idx = 0usize;
        let mut rem = i;
        for d in 0..target.len() {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            src_idx += coord * src_strides[d];
        }
        out[i] = data[src_idx];
    }
    RustyNumTensor::from_i64(out, target)
}

/// Unfold windows along a dimension for i64 tensors.
fn int_unfold_impl(tensor: &RustyNumTensor, dim: usize, size: usize, step: usize) -> RustyNumTensor {
    let data = tensor.as_i64();
    let shape = &tensor.shape;
    let rank = shape.len();
    let strides = c_strides(shape);

    let dim_size = shape[dim];
    let num_windows = if dim_size >= size {
        (dim_size - size) / step + 1
    } else {
        0
    };

    // Output shape: [..., num_windows, size, ...]
    // Insert num_windows at dim, size at dim+1, removing original dim
    let mut new_shape = Vec::with_capacity(rank + 1);
    for d in 0..rank {
        if d == dim {
            new_shape.push(num_windows);
            new_shape.push(size);
        } else {
            new_shape.push(shape[d]);
        }
    }

    let n: usize = new_shape.iter().product();
    let new_strides = c_strides(&new_shape);

    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut rem = i;
        let mut src_idx = 0usize;
        let mut new_d = 0;
        for d in 0..rank {
            if d == dim {
                // window index
                let win_coord = rem / new_strides[new_d];
                rem %= new_strides[new_d];
                new_d += 1;
                // position within window
                let pos_coord = rem / new_strides[new_d];
                rem %= new_strides[new_d];
                new_d += 1;
                let src_coord = win_coord * step + pos_coord;
                src_idx += src_coord * strides[d];
            } else {
                let coord = rem / new_strides[new_d];
                rem %= new_strides[new_d];
                new_d += 1;
                src_idx += coord * strides[d];
            }
        }
        out[i] = data[src_idx];
    }
    RustyNumTensor::from_i64(out, new_shape)
}
