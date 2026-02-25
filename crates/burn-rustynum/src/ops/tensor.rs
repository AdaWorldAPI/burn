use alloc::vec;
use alloc::vec::Vec;

use burn_backend::backend::ExecutionError;
use burn_backend::ops::FloatTensorOps;
use burn_backend::tensor::FloatTensor;
use burn_backend::{Distribution, FloatDType, Scalar, Shape, TensorData};
use burn_std::tensor::Slice;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;
use libm::erf;

use crate::backend::{RustyNum, RustyNumDevice, SEED};
use crate::tensor::{RustyNumTensor, TensorStorage};

use rand::rngs::StdRng;
use rand::SeedableRng;

impl FloatTensorOps<Self> for RustyNum {
    fn float_from_data(data: TensorData, _device: &RustyNumDevice) -> FloatTensor<Self> {
        RustyNumTensor::from_data(data)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &RustyNumDevice,
    ) -> FloatTensor<Self> {
        let mut seed = SEED.lock().unwrap();
        let mut rng = seed.take().unwrap_or_else(|| StdRng::from_rng(&mut rand::rng()));
        let data = TensorData::random::<f32, _, _>(shape, distribution, &mut rng);
        *seed = Some(rng);
        RustyNumTensor::from_data(data)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> RustyNumDevice {
        RustyNumDevice::Cpu
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &RustyNumDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(
        shape: Shape,
        _device: &RustyNumDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        let burn_dtype = match dtype {
            FloatDType::F64 => burn_backend::DType::F64,
            FloatDType::F32 | FloatDType::Flex32 => burn_backend::DType::F32,
            _ => burn_backend::DType::F32,
        };
        RustyNumTensor::zeros(shape.to_vec(), burn_dtype)
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> <Self as burn_backend::Backend>::IntTensorPrimitive {
        tensor.cast_to(burn_backend::DType::I64)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        { simd_add_f32(&lhs, &rhs) }
        #[cfg(not(feature = "simd"))]
        { binary_float_op(&lhs, &rhs, |a, b| a + b) }
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        { simd_add_scalar_f32(&lhs, rhs.elem()) }
        #[cfg(not(feature = "simd"))]
        { scalar_float_op(&lhs, rhs, |a, s| a + s) }
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        { simd_sub_f32(&lhs, &rhs) }
        #[cfg(not(feature = "simd"))]
        { binary_float_op(&lhs, &rhs, |a, b| a - b) }
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        { simd_sub_scalar_f32(&lhs, rhs.elem()) }
        #[cfg(not(feature = "simd"))]
        { scalar_float_op(&lhs, rhs, |a, s| a - s) }
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        { simd_mul_f32(&lhs, &rhs) }
        #[cfg(not(feature = "simd"))]
        { binary_float_op(&lhs, &rhs, |a, b| a * b) }
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        { simd_mul_scalar_f32(&lhs, rhs.elem()) }
        #[cfg(not(feature = "simd"))]
        { scalar_float_op(&lhs, rhs, |a, s| a * s) }
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        { simd_div_f32(&lhs, &rhs) }
        #[cfg(not(feature = "simd"))]
        { binary_float_op(&lhs, &rhs, |a, b| a / b) }
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        #[cfg(feature = "simd")]
        { simd_div_scalar_f32(&lhs, rhs.elem()) }
        #[cfg(not(feature = "simd"))]
        { scalar_float_op(&lhs, rhs, |a, s| a / s) }
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_op(&lhs, &rhs, |a, b| a % b)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        scalar_float_op(&lhs, rhs, |a, s| a % s)
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        matmul_impl(&lhs, &rhs)
    }

    fn float_cross(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        // Cross product requires dim of size 3
        let lhs_data = lhs.as_f32();
        let rhs_data = rhs.as_f32();
        let shape = lhs.shape.clone();
        let n = lhs.num_elements();
        let stride = shape[dim];
        assert_eq!(stride, 3, "Cross product requires dimension of size 3");

        // Compute strides for the cross dimension
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let mut out = vec![0.0f32; n];

        for o in 0..outer {
            for i in 0..inner {
                let idx = |d: usize| o * 3 * inner + d * inner + i;
                let a0 = lhs_data[idx(0)];
                let a1 = lhs_data[idx(1)];
                let a2 = lhs_data[idx(2)];
                let b0 = rhs_data[idx(0)];
                let b1 = rhs_data[idx(1)];
                let b2 = rhs_data[idx(2)];
                out[idx(0)] = a1 * b2 - a2 * b1;
                out[idx(1)] = a2 * b0 - a0 * b2;
                out[idx(2)] = a0 * b1 - a1 * b0;
            }
        }

        RustyNumTensor::from_f32(out, shape)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| 1.0 / x)
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        swap_dims_impl(tensor, dim1, dim2)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        permute_impl(tensor, axes)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        flip_impl(tensor, axes)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        tensor.reshape(shape.to_vec())
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: <Self as burn_backend::Backend>::IntTensorPrimitive,
    ) -> FloatTensor<Self> {
        gather_impl(&tensor, dim, &indices)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: <Self as burn_backend::Backend>::IntTensorPrimitive,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        scatter_add_impl(tensor, dim, &indices, &value)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: <Self as burn_backend::Backend>::IntTensorPrimitive,
    ) -> FloatTensor<Self> {
        select_impl(&tensor, dim, &indices)
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: <Self as burn_backend::Backend>::IntTensorPrimitive,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        select_add_impl(tensor, dim, &indices, &value)
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        slice_impl(&tensor, slices)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        slice_assign_impl(tensor, slices, &value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: <Self as burn_backend::Backend>::BoolTensorPrimitive,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        mask_where_impl(&tensor, &mask, &value)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: <Self as burn_backend::Backend>::BoolTensorPrimitive,
        value: Scalar,
    ) -> FloatTensor<Self> {
        mask_fill_impl(&tensor, &mask, value)
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_float_op(&lhs, &rhs, |a, b| a == b)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_scalar_float_op(&lhs, rhs, |a, s| a == s)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_float_op(&lhs, &rhs, |a, b| a > b)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_scalar_float_op(&lhs, rhs, |a, s| a > s)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_float_op(&lhs, &rhs, |a, b| a >= b)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_scalar_float_op(&lhs, rhs, |a, s| a >= s)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_float_op(&lhs, &rhs, |a, b| a < b)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_scalar_float_op(&lhs, rhs, |a, s| a < s)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_float_op(&lhs, &rhs, |a, b| a <= b)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> <Self as burn_backend::Backend>::BoolTensorPrimitive {
        compare_scalar_float_op(&lhs, rhs, |a, s| a <= s)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let sum: f64 = match &tensor.storage {
            TensorStorage::F32(v) => v.iter().map(|&x| x as f64).sum(),
            TensorStorage::F64(v) => v.iter().sum(),
            _ => panic!("Expected float tensor"),
        };
        match &tensor.storage {
            TensorStorage::F32(_) => RustyNumTensor::from_f32(vec![sum as f32], vec![1]),
            TensorStorage::F64(_) => RustyNumTensor::from_f64(vec![sum], vec![1]),
            _ => unreachable!(),
        }
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_dim_impl(&tensor, dim, 0.0, |acc, x| acc + x)
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let n = tensor.num_elements() as f64;
        let sum: f64 = match &tensor.storage {
            TensorStorage::F32(v) => v.iter().map(|&x| x as f64).sum(),
            TensorStorage::F64(v) => v.iter().sum(),
            _ => panic!("Expected float tensor"),
        };
        match &tensor.storage {
            TensorStorage::F32(_) => RustyNumTensor::from_f32(vec![(sum / n) as f32], vec![1]),
            TensorStorage::F64(_) => RustyNumTensor::from_f64(vec![sum / n], vec![1]),
            _ => unreachable!(),
        }
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let dim_size = tensor.shape[dim] as f32;
        let sum = reduce_dim_impl(&tensor, dim, 0.0, |acc, x| acc + x);
        scalar_float_op(&sum, Scalar::Float(dim_size as f64), |a, s| a / s)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.exp())
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.ln())
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.ln_1p())
    }

    fn float_powf_scalar(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        let exp: f32 = value.elem();
        unary_float_op(&tensor, |x| x.powf(exp))
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.sqrt())
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.abs())
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.cos())
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.sin())
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.tan())
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.cosh())
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.sinh())
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.tanh())
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.acos())
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.acosh())
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.asin())
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.asinh())
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.atan())
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.atanh())
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_op(&lhs, &rhs, |a, b| a.atan2(b))
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.round())
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.floor())
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.ceil())
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| x.trunc())
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_op(&tensor, |x| erf(x as f64) as f32)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> <Self as burn_backend::Backend>::IntTensorPrimitive {
        argmax_impl(&tensor, dim)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> <Self as burn_backend::Backend>::IntTensorPrimitive {
        argmin_impl(&tensor, dim)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        expand_impl(tensor, shape)
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let target = match dtype {
            FloatDType::F64 => burn_backend::DType::F64,
            FloatDType::F32 | FloatDType::Flex32 => burn_backend::DType::F32,
            _ => burn_backend::DType::F32,
        };
        tensor.cast_to(target)
    }

    fn float_unfold(tensor: FloatTensor<Self>, dim: usize, size: usize, step: usize) -> FloatTensor<Self> {
        unfold_impl(tensor, dim, size, step)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        cum_op_impl(&tensor, dim, 0.0, |acc, x| acc + x)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        cum_op_impl(&tensor, dim, 1.0, |acc, x| acc * x)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        cum_op_impl(&tensor, dim, f32::INFINITY, |acc, x| acc.min(x))
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        cum_op_impl(&tensor, dim, f32::NEG_INFINITY, |acc, x| acc.max(x))
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_op(&lhs, &rhs, |a, b| a.powf(b))
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        let exp: f32 = value.elem();
        unary_float_op(&tensor, |x| x.powf(exp))
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let prod: f64 = match &tensor.storage {
            TensorStorage::F32(v) => v.iter().map(|&x| x as f64).product(),
            TensorStorage::F64(v) => v.iter().product(),
            _ => panic!("Expected float tensor"),
        };
        match &tensor.storage {
            TensorStorage::F32(_) => RustyNumTensor::from_f32(vec![prod as f32], vec![1]),
            TensorStorage::F64(_) => RustyNumTensor::from_f64(vec![prod], vec![1]),
            _ => unreachable!(),
        }
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_dim_impl(&tensor, dim, 1.0, |acc, x| acc * x)
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Element-wise binary float operation.
fn binary_float_op(lhs: &RustyNumTensor, rhs: &RustyNumTensor, op: impl Fn(f32, f32) -> f32) -> RustyNumTensor {
    let lhs_data = lhs.as_f32();
    let rhs_data = rhs.as_f32();

    // Handle broadcasting
    if lhs.shape == rhs.shape {
        let out: Vec<f32> = lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| op(a, b)).collect();
        RustyNumTensor::from_f32(out, lhs.shape.clone())
    } else {
        broadcast_binary_op(lhs, rhs, op)
    }
}

/// SIMD-accelerated element-wise add (same-shape fast path).
#[cfg(feature = "simd")]
fn simd_add_f32(lhs: &RustyNumTensor, rhs: &RustyNumTensor) -> RustyNumTensor {
    if lhs.shape == rhs.shape {
        let out = rustynum_core::simd::add_f32_vec(lhs.as_f32(), rhs.as_f32());
        RustyNumTensor::from_f32(out, lhs.shape.clone())
    } else {
        broadcast_binary_op(lhs, rhs, |a, b| a + b)
    }
}

/// SIMD-accelerated element-wise subtract (same-shape fast path).
#[cfg(feature = "simd")]
fn simd_sub_f32(lhs: &RustyNumTensor, rhs: &RustyNumTensor) -> RustyNumTensor {
    if lhs.shape == rhs.shape {
        let out = rustynum_core::simd::sub_f32_vec(lhs.as_f32(), rhs.as_f32());
        RustyNumTensor::from_f32(out, lhs.shape.clone())
    } else {
        broadcast_binary_op(lhs, rhs, |a, b| a - b)
    }
}

/// SIMD-accelerated element-wise multiply (same-shape fast path).
#[cfg(feature = "simd")]
fn simd_mul_f32(lhs: &RustyNumTensor, rhs: &RustyNumTensor) -> RustyNumTensor {
    if lhs.shape == rhs.shape {
        let out = rustynum_core::simd::mul_f32_vec(lhs.as_f32(), rhs.as_f32());
        RustyNumTensor::from_f32(out, lhs.shape.clone())
    } else {
        broadcast_binary_op(lhs, rhs, |a, b| a * b)
    }
}

/// SIMD-accelerated element-wise divide (same-shape fast path).
#[cfg(feature = "simd")]
fn simd_div_f32(lhs: &RustyNumTensor, rhs: &RustyNumTensor) -> RustyNumTensor {
    if lhs.shape == rhs.shape {
        let out = rustynum_core::simd::div_f32_vec(lhs.as_f32(), rhs.as_f32());
        RustyNumTensor::from_f32(out, lhs.shape.clone())
    } else {
        broadcast_binary_op(lhs, rhs, |a, b| a / b)
    }
}

/// SIMD-accelerated scalar add.
#[cfg(feature = "simd")]
fn simd_add_scalar_f32(tensor: &RustyNumTensor, scalar: f32) -> RustyNumTensor {
    let out = rustynum_core::simd::add_f32_scalar(tensor.as_f32(), scalar);
    RustyNumTensor::from_f32(out, tensor.shape.clone())
}

/// SIMD-accelerated scalar subtract.
#[cfg(feature = "simd")]
fn simd_sub_scalar_f32(tensor: &RustyNumTensor, scalar: f32) -> RustyNumTensor {
    let out = rustynum_core::simd::sub_f32_scalar(tensor.as_f32(), scalar);
    RustyNumTensor::from_f32(out, tensor.shape.clone())
}

/// SIMD-accelerated scalar multiply.
#[cfg(feature = "simd")]
fn simd_mul_scalar_f32(tensor: &RustyNumTensor, scalar: f32) -> RustyNumTensor {
    let out = rustynum_core::simd::mul_f32_scalar(tensor.as_f32(), scalar);
    RustyNumTensor::from_f32(out, tensor.shape.clone())
}

/// SIMD-accelerated scalar divide.
#[cfg(feature = "simd")]
fn simd_div_scalar_f32(tensor: &RustyNumTensor, scalar: f32) -> RustyNumTensor {
    let out = rustynum_core::simd::div_f32_scalar(tensor.as_f32(), scalar);
    RustyNumTensor::from_f32(out, tensor.shape.clone())
}

/// Broadcast two tensors and apply a binary op.
fn broadcast_binary_op(lhs: &RustyNumTensor, rhs: &RustyNumTensor, op: impl Fn(f32, f32) -> f32) -> RustyNumTensor {
    let out_shape = broadcast_shape(&lhs.shape, &rhs.shape);
    let n: usize = out_shape.iter().product();
    let lhs_data = lhs.as_f32();
    let rhs_data = rhs.as_f32();
    let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
    let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);
    let out_strides = crate::tensor::c_strides(&out_shape);

    let mut out = vec![0.0f32; n];
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
    RustyNumTensor::from_f32(out, out_shape)
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
    let src_strides = crate::tensor::c_strides(src_shape);

    for i in 0..rank {
        let src_i = i as isize - (rank as isize - src_rank as isize);
        if src_i >= 0 {
            let si = src_i as usize;
            if src_shape[si] == out_shape[i] {
                strides[i] = src_strides[si];
            } else {
                // Broadcast dim: stride = 0
                strides[i] = 0;
            }
        }
        // else: implicitly broadcast (stride stays 0)
    }
    strides
}

/// Scalar float operation.
fn scalar_float_op(tensor: &RustyNumTensor, scalar: Scalar, op: impl Fn(f32, f32) -> f32) -> RustyNumTensor {
    let s: f32 = scalar.elem();
    let data = tensor.as_f32();
    let out: Vec<f32> = data.iter().map(|&x| op(x, s)).collect();
    RustyNumTensor::from_f32(out, tensor.shape.clone())
}

/// Unary float operation.
fn unary_float_op(tensor: &RustyNumTensor, op: impl Fn(f32) -> f32) -> RustyNumTensor {
    let data = tensor.as_f32();
    let out: Vec<f32> = data.iter().map(|&x| op(x)).collect();
    RustyNumTensor::from_f32(out, tensor.shape.clone())
}

/// Compare two float tensors element-wise, producing a bool tensor.
fn compare_float_op(lhs: &RustyNumTensor, rhs: &RustyNumTensor, op: impl Fn(f32, f32) -> bool) -> RustyNumTensor {
    let lhs_data = lhs.as_f32();
    let rhs_data = rhs.as_f32();
    if lhs.shape == rhs.shape {
        let out: Vec<bool> = lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| op(a, b)).collect();
        RustyNumTensor::from_bool(out, lhs.shape.clone())
    } else {
        let out_shape = broadcast_shape(&lhs.shape, &rhs.shape);
        let n: usize = out_shape.iter().product();
        let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
        let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);
        let out_strides = crate::tensor::c_strides(&out_shape);
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

fn compare_scalar_float_op(tensor: &RustyNumTensor, scalar: Scalar, op: impl Fn(f32, f32) -> bool) -> RustyNumTensor {
    let s: f32 = scalar.elem();
    let data = tensor.as_f32();
    let out: Vec<bool> = data.iter().map(|&x| op(x, s)).collect();
    RustyNumTensor::from_bool(out, tensor.shape.clone())
}

/// Matrix multiply: supports batched matmul.
fn matmul_impl(lhs: &RustyNumTensor, rhs: &RustyNumTensor) -> RustyNumTensor {
    let lhs_data = lhs.as_f32();
    let rhs_data = rhs.as_f32();

    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();
    assert!(lhs_rank >= 2 && rhs_rank >= 2, "matmul requires at least 2D tensors");

    let m = lhs.shape[lhs_rank - 2];
    let k = lhs.shape[lhs_rank - 1];
    let k2 = rhs.shape[rhs_rank - 2];
    let n = rhs.shape[rhs_rank - 1];
    assert_eq!(k, k2, "matmul inner dimensions must match: {} vs {}", k, k2);

    // Batch dimensions
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

    let mut out = vec![0.0f32; batch * out_stride];

    for b in 0..batch {
        let lb = if lhs_batch == 1 { 0 } else { b };
        let rb = if rhs_batch == 1 { 0 } else { b };
        let a = &lhs_data[lb * lhs_stride..(lb + 1) * lhs_stride];
        let b_mat = &rhs_data[rb * rhs_stride..(rb + 1) * rhs_stride];
        let c = &mut out[b * out_stride..(b + 1) * out_stride];

        #[cfg(feature = "simd")]
        {
            // AVX-512 Goto BLAS sgemm: cache-blocked microkernel, multi-threaded for large matrices
            rustyblas::level3::sgemm(
                rustyblas::Layout::RowMajor,
                rustyblas::Transpose::NoTrans,
                rustyblas::Transpose::NoTrans,
                m, n, k,
                1.0, a, k,
                b_mat, n,
                0.0, c, n,
            );
        }
        #[cfg(not(feature = "simd"))]
        {
            for i in 0..m {
                for p in 0..k {
                    let a_ip = a[i * k + p];
                    for j in 0..n {
                        c[i * n + j] += a_ip * b_mat[p * n + j];
                    }
                }
            }
        }
    }

    RustyNumTensor::from_f32(out, out_shape)
}

/// Reduce along a dimension.
fn reduce_dim_impl(tensor: &RustyNumTensor, dim: usize, init: f32, op: impl Fn(f32, f32) -> f32) -> RustyNumTensor {
    let data = tensor.as_f32();
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
    RustyNumTensor::from_f32(out, new_shape)
}

/// Argmax along a dimension.
fn argmax_impl(tensor: &RustyNumTensor, dim: usize) -> RustyNumTensor {
    let data = tensor.as_f32();
    let shape = &tensor.shape;
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
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

/// Argmin along a dimension.
fn argmin_impl(tensor: &RustyNumTensor, dim: usize) -> RustyNumTensor {
    let data = tensor.as_f32();
    let shape = &tensor.shape;
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = f32::INFINITY;
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

/// Swap two dimensions (generalized transpose).
fn swap_dims_impl(tensor: RustyNumTensor, dim1: usize, dim2: usize) -> RustyNumTensor {
    if dim1 == dim2 {
        return tensor;
    }
    let mut axes: Vec<usize> = (0..tensor.shape.len()).collect();
    axes.swap(dim1, dim2);
    permute_impl(tensor, &axes)
}

/// Permute dimensions.
fn permute_impl(tensor: RustyNumTensor, axes: &[usize]) -> RustyNumTensor {
    let data = tensor.as_f32();
    let old_shape = &tensor.shape;
    let rank = old_shape.len();
    let old_strides = crate::tensor::c_strides(old_shape);

    let new_shape: Vec<usize> = axes.iter().map(|&a| old_shape[a]).collect();
    let new_strides = crate::tensor::c_strides(&new_shape);
    let n = tensor.num_elements();

    let mut out = vec![0.0f32; n];
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
    RustyNumTensor::from_f32(out, new_shape)
}

/// Flip along specified axes.
fn flip_impl(tensor: RustyNumTensor, axes: &[usize]) -> RustyNumTensor {
    let data = tensor.as_f32();
    let shape = &tensor.shape;
    let strides = crate::tensor::c_strides(shape);
    let n = tensor.num_elements();
    let rank = shape.len();

    let mut out = vec![0.0f32; n];
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
    RustyNumTensor::from_f32(out, shape.clone())
}

/// Slice tensor along multiple dimensions.
fn slice_impl(tensor: &RustyNumTensor, slices: &[Slice]) -> RustyNumTensor {
    let data = tensor.as_f32();
    let shape = &tensor.shape;
    let rank = shape.len();
    let strides = crate::tensor::c_strides(shape);

    // Resolve slice ranges
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
    let new_strides = crate::tensor::c_strides(&new_shape);

    let mut out = vec![0.0f32; n];
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
    RustyNumTensor::from_f32(out, new_shape)
}

/// Assign values to a slice of a tensor.
fn slice_assign_impl(mut tensor: RustyNumTensor, slices: &[Slice], value: &RustyNumTensor) -> RustyNumTensor {
    let shape = tensor.shape.clone();
    let rank = shape.len();
    let strides = crate::tensor::c_strides(&shape);
    let val_data = value.as_f32();
    let val_strides = crate::tensor::c_strides(&value.shape);

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

    let data = tensor.as_f32_mut();
    for i in 0..n {
        let mut dst_idx = 0usize;
        let mut rem = i;
        for d in 0..rank {
            let coord = rem / val_strides[d];
            rem %= val_strides[d];
            let dst_coord = ranges[d].0 + coord * ranges[d].2;
            dst_idx += dst_coord * strides[d];
        }
        data[dst_idx] = val_data[i];
    }
    tensor
}

/// Gather elements along a dimension.
fn gather_impl(tensor: &RustyNumTensor, dim: usize, indices: &RustyNumTensor) -> RustyNumTensor {
    let data = tensor.as_f32();
    let idx_data = indices.as_i64();
    let shape = &tensor.shape;
    let strides = crate::tensor::c_strides(shape);
    let idx_shape = &indices.shape;
    let idx_strides = crate::tensor::c_strides(idx_shape);
    let n: usize = idx_shape.iter().product();

    let mut out = vec![0.0f32; n];
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
    RustyNumTensor::from_f32(out, idx_shape.clone())
}

/// Scatter add.
fn scatter_add_impl(mut tensor: RustyNumTensor, dim: usize, indices: &RustyNumTensor, value: &RustyNumTensor) -> RustyNumTensor {
    let idx_data = indices.as_i64();
    let val_data = value.as_f32();
    let shape = tensor.shape.clone();
    let strides = crate::tensor::c_strides(&shape);
    let idx_shape = &indices.shape;
    let idx_strides = crate::tensor::c_strides(idx_shape);
    let n: usize = idx_shape.iter().product();

    let data = tensor.as_f32_mut();
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
        data[dst_idx] += val_data[i];
    }
    tensor
}

/// Select along a dimension.
fn select_impl(tensor: &RustyNumTensor, dim: usize, indices: &RustyNumTensor) -> RustyNumTensor {
    let data = tensor.as_f32();
    let idx_data = indices.as_i64();
    let shape = &tensor.shape;
    let strides = crate::tensor::c_strides(shape);

    let num_indices = idx_data.len();
    let mut new_shape = shape.clone();
    new_shape[dim] = num_indices;
    let n: usize = new_shape.iter().product();
    let new_strides = crate::tensor::c_strides(&new_shape);

    let mut out = vec![0.0f32; n];
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
    RustyNumTensor::from_f32(out, new_shape)
}

/// Select add.
fn select_add_impl(mut tensor: RustyNumTensor, dim: usize, indices: &RustyNumTensor, value: &RustyNumTensor) -> RustyNumTensor {
    let idx_data = indices.as_i64();
    let val_data = value.as_f32();
    let shape = tensor.shape.clone();
    let strides = crate::tensor::c_strides(&shape);

    let num_indices = idx_data.len();
    let mut val_shape = shape.clone();
    val_shape[dim] = num_indices;
    let val_strides = crate::tensor::c_strides(&val_shape);
    let n: usize = val_shape.iter().product();

    let data = tensor.as_f32_mut();
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
        data[dst_idx] += val_data[i];
    }
    tensor
}

/// Mask where.
fn mask_where_impl(tensor: &RustyNumTensor, mask: &RustyNumTensor, value: &RustyNumTensor) -> RustyNumTensor {
    let data = tensor.as_f32();
    let mask_data = mask.as_bool();
    let val_data = value.as_f32();
    let out: Vec<f32> = data.iter().zip(mask_data.iter()).zip(val_data.iter())
        .map(|((&t, &m), &v)| if m { v } else { t })
        .collect();
    RustyNumTensor::from_f32(out, tensor.shape.clone())
}

/// Mask fill.
fn mask_fill_impl(tensor: &RustyNumTensor, mask: &RustyNumTensor, value: Scalar) -> RustyNumTensor {
    let data = tensor.as_f32();
    let mask_data = mask.as_bool();
    let fill: f32 = value.elem();
    let out: Vec<f32> = data.iter().zip(mask_data.iter())
        .map(|(&t, &m)| if m { fill } else { t })
        .collect();
    RustyNumTensor::from_f32(out, tensor.shape.clone())
}

/// Expand (broadcast) tensor to a target shape.
fn expand_impl(tensor: RustyNumTensor, shape: Shape) -> RustyNumTensor {
    let target = shape.to_vec();
    if tensor.shape == target {
        return tensor;
    }

    let data = tensor.as_f32();
    let src_strides = broadcast_strides(&tensor.shape, &target);
    let out_strides = crate::tensor::c_strides(&target);
    let n: usize = target.iter().product();

    let mut out = vec![0.0f32; n];
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
    RustyNumTensor::from_f32(out, target)
}

/// Cumulative operation along a dimension (cumsum, cumprod, cummin, cummax).
fn cum_op_impl(tensor: &RustyNumTensor, dim: usize, init: f32, op: impl Fn(f32, f32) -> f32) -> RustyNumTensor {
    let data = tensor.as_f32();
    let shape = &tensor.shape;
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let mut out = vec![0.0f32; data.len()];
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
    RustyNumTensor::from_f32(out, shape.clone())
}

/// Unfold along a dimension: extracts sliding windows of given size with given step.
fn unfold_impl(tensor: RustyNumTensor, dim: usize, size: usize, step: usize) -> RustyNumTensor {
    let data = tensor.as_f32();
    let shape = &tensor.shape;
    let rank = shape.len();
    let strides = crate::tensor::c_strides(shape);

    let dim_size = shape[dim];
    let num_windows = (dim_size.saturating_sub(size)) / step + 1;

    // Output shape: [..., num_windows, ..., size] where the new dim is appended at the end
    let mut new_shape = shape.clone();
    new_shape[dim] = num_windows;
    new_shape.push(size);

    let n: usize = new_shape.iter().product();
    let new_strides = crate::tensor::c_strides(&new_shape);

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut rem = i;
        let mut src_idx = 0usize;
        let mut window_offset = 0usize;
        for d in 0..new_shape.len() {
            let coord = rem / new_strides[d];
            rem %= new_strides[d];
            if d == dim {
                // This is the window index
                src_idx += coord * step * strides[dim];
            } else if d == rank {
                // This is the position within the window
                window_offset = coord;
            } else {
                src_idx += coord * strides[d];
            }
        }
        src_idx += window_offset * strides[dim];
        out[i] = data[src_idx];
    }
    RustyNumTensor::from_f32(out, new_shape)
}
