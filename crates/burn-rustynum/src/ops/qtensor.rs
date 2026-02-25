use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use burn_backend::backend::ExecutionError;
use burn_backend::ops::QTensorOps;
use burn_backend::quantization::{
    QParams, QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue,
    QuantizationParametersPrimitive, QuantizedBytes,
};
use burn_backend::tensor::{FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{DType, Shape, TensorData, TensorMetadata};
use burn_std::tensor::Slice;

use crate::backend::{RustyNum, RustyNumDevice};
use crate::tensor::{RustyNumQTensor, RustyNumTensor, TensorStorage};

impl QTensorOps<Self> for RustyNum {
    fn q_from_data(data: TensorData, _device: &RustyNumDevice) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(scheme) => {
                let shape = data.shape.clone();
                let num_elements = data.num_elements();
                let q_bytes = QuantizedBytes {
                    bytes: data.into_bytes(),
                    scheme,
                    num_elements,
                };

                match scheme {
                    QuantScheme {
                        level: QuantLevel::Tensor | QuantLevel::Block(_),
                        mode: QuantMode::Symmetric,
                        value: QuantValue::Q8F | QuantValue::Q8S,
                        ..
                    } => {
                        let (values, qparams) = q_bytes.into_vec_i8();
                        let data = TensorData::new(values, shape);
                        let scheme = scheme.with_store(QuantStore::Native);

                        let qparams = qparams
                            .scales
                            .into_iter()
                            .map(|scales| QParams { scales })
                            .collect();

                        RustyNumQTensor {
                            qtensor: RustyNumTensor::from_data(data),
                            scheme,
                            qparams,
                        }
                    }
                    _ => unimplemented!(
                        "q_from_data not supported for scheme {:?}",
                        scheme
                    ),
                }
            }
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let shape = tensor.shape.clone();
        let float_data = tensor.as_f32();
        let scale_data = qparams.scales.as_f32();

        match scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                value: QuantValue::Q8F | QuantValue::Q8S,
                store: QuantStore::Native,
                ..
            } => {
                let scale = scale_data[0];
                let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

                let quantized: Vec<i8> = float_data
                    .iter()
                    .map(|&v| {
                        let q = (v * inv_scale).round();
                        q.clamp(-128.0, 127.0) as i8
                    })
                    .collect();

                let q_tensor = RustyNumTensor {
                    storage: TensorStorage::I8(Arc::new(quantized)),
                    shape,
                };

                RustyNumQTensor {
                    qtensor: q_tensor,
                    scheme: *scheme,
                    qparams: vec![QParams { scales: scale }],
                }
            }
            QuantScheme {
                level: QuantLevel::Block(block_size),
                mode: QuantMode::Symmetric,
                value: QuantValue::Q8F | QuantValue::Q8S,
                store: QuantStore::Native,
                ..
            } => {
                let scales = scale_data;
                let block_n = block_size.num_elements();
                let mut quantized = Vec::with_capacity(float_data.len());
                let mut all_qparams = Vec::new();

                for (block_idx, chunk) in float_data.chunks(block_n).enumerate() {
                    let scale = if block_idx < scales.len() {
                        scales[block_idx]
                    } else {
                        scales[scales.len() - 1]
                    };
                    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

                    for &v in chunk {
                        let q = (v * inv_scale).round();
                        quantized.push(q.clamp(-128.0, 127.0) as i8);
                    }
                    all_qparams.push(QParams { scales: scale });
                }

                let q_tensor = RustyNumTensor {
                    storage: TensorStorage::I8(Arc::new(quantized)),
                    shape,
                };

                RustyNumQTensor {
                    qtensor: q_tensor,
                    scheme: *scheme,
                    qparams: all_qparams,
                }
            }
            _ => unimplemented!("Quantization not supported for scheme {:?}", scheme),
        }
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let shape = tensor.qtensor.shape.clone();

        let i8_data = match &tensor.qtensor.storage {
            TensorStorage::I8(v) => v.as_slice(),
            _ => panic!(
                "Expected I8 storage for quantized tensor, got {:?}",
                tensor.qtensor.dtype()
            ),
        };

        match tensor.scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                ..
            } => {
                let scale = tensor.qparams[0].scales;
                let dequantized: Vec<f32> =
                    i8_data.iter().map(|&q| q as f32 * scale).collect();
                RustyNumTensor::from_f32(dequantized, shape)
            }
            QuantScheme {
                level: QuantLevel::Block(block_size),
                mode: QuantMode::Symmetric,
                ..
            } => {
                let block_n = block_size.num_elements();
                let mut dequantized = Vec::with_capacity(i8_data.len());

                for (block_idx, chunk) in i8_data.chunks(block_n).enumerate() {
                    let scale = if block_idx < tensor.qparams.len() {
                        tensor.qparams[block_idx].scales
                    } else {
                        tensor.qparams[tensor.qparams.len() - 1].scales
                    };
                    for &q in chunk {
                        dequantized.push(q as f32 * scale);
                    }
                }

                RustyNumTensor::from_f32(dequantized, shape)
            }
        }
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> RustyNumDevice {
        RustyNumDevice::Cpu
    }

    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        _device: &RustyNumDevice,
    ) -> QuantizedTensor<Self> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        RustyNumQTensor {
            qtensor: tensor.qtensor.reshape(shape.to_vec()),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    async fn q_into_data(
        tensor: QuantizedTensor<Self>,
    ) -> Result<TensorData, ExecutionError> {
        let shape = tensor.qtensor.shape();
        let scales: Vec<f32> = tensor.qparams.iter().map(|q| q.scales).collect();

        let i8_data = match &tensor.qtensor.storage {
            TensorStorage::I8(v) => (**v).clone(),
            _ => panic!(
                "Expected I8 storage for quantized tensor, got {:?}",
                tensor.qtensor.dtype()
            ),
        };

        Ok(TensorData::quantized(i8_data, shape, tensor.scheme, &scales))
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        RustyNumQTensor {
            qtensor: swap_dims_i8(tensor.qtensor, dim1, dim2),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        RustyNumQTensor {
            qtensor: permute_i8(tensor.qtensor, axes),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        RustyNumQTensor {
            qtensor: flip_i8(tensor.qtensor, axes),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        RustyNumQTensor {
            qtensor: select_i8(tensor.qtensor, dim, &indices),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_slice(
        tensor: QuantizedTensor<Self>,
        slices: &[Slice],
    ) -> QuantizedTensor<Self> {
        RustyNumQTensor {
            qtensor: slice_i8(&tensor.qtensor, slices),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        RustyNumQTensor {
            qtensor: expand_i8(tensor.qtensor, shape),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }
}

// =============================================================================
// i8 tensor operation helpers for quantized data manipulation
// =============================================================================

/// Swap two dimensions of an i8 tensor.
fn swap_dims_i8(tensor: RustyNumTensor, dim1: usize, dim2: usize) -> RustyNumTensor {
    if dim1 == dim2 {
        return tensor;
    }
    let mut axes: Vec<usize> = (0..tensor.shape.len()).collect();
    axes.swap(dim1, dim2);
    permute_i8(tensor, &axes)
}

/// Permute dimensions of an i8 tensor.
fn permute_i8(tensor: RustyNumTensor, axes: &[usize]) -> RustyNumTensor {
    let data = get_i8_data(&tensor);
    let old_shape = &tensor.shape;
    let rank = old_shape.len();
    let old_strides = crate::tensor::c_strides(old_shape);

    let new_shape: Vec<usize> = axes.iter().map(|&a| old_shape[a]).collect();
    let new_strides = crate::tensor::c_strides(&new_shape);
    let n = tensor.num_elements();

    let mut out = vec![0i8; n];
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

    RustyNumTensor {
        storage: TensorStorage::I8(Arc::new(out)),
        shape: new_shape,
    }
}

/// Flip an i8 tensor along specified axes.
fn flip_i8(tensor: RustyNumTensor, axes: &[usize]) -> RustyNumTensor {
    let data = get_i8_data(&tensor);
    let shape = &tensor.shape;
    let strides = crate::tensor::c_strides(shape);
    let n = tensor.num_elements();
    let rank = shape.len();

    let mut out = vec![0i8; n];
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

    RustyNumTensor {
        storage: TensorStorage::I8(Arc::new(out)),
        shape: shape.clone(),
    }
}

/// Select elements from an i8 tensor along a dimension.
fn select_i8(
    tensor: RustyNumTensor,
    dim: usize,
    indices: &RustyNumTensor,
) -> RustyNumTensor {
    let data = get_i8_data(&tensor);
    let idx_data = indices.as_i64();
    let shape = &tensor.shape;
    let strides = crate::tensor::c_strides(shape);

    let num_indices = idx_data.len();
    let mut new_shape = shape.clone();
    new_shape[dim] = num_indices;
    let n: usize = new_shape.iter().product();
    let new_strides = crate::tensor::c_strides(&new_shape);

    let mut out = vec![0i8; n];
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

    RustyNumTensor {
        storage: TensorStorage::I8(Arc::new(out)),
        shape: new_shape,
    }
}

/// Slice an i8 tensor.
fn slice_i8(tensor: &RustyNumTensor, slices: &[Slice]) -> RustyNumTensor {
    let data = get_i8_data(tensor);
    let shape = &tensor.shape;
    let rank = shape.len();
    let strides = crate::tensor::c_strides(shape);

    let ranges: Vec<(usize, usize, usize)> = (0..rank)
        .map(|d| {
            if d < slices.len() {
                let s = &slices[d];
                let dim_len = shape[d] as isize;
                let start = if s.start < 0 {
                    (dim_len + s.start) as usize
                } else {
                    s.start as usize
                };
                let end = match s.end {
                    Some(e) => {
                        if e < 0 {
                            (dim_len + e) as usize
                        } else {
                            e as usize
                        }
                    }
                    None => shape[d],
                };
                let step = s.step as usize;
                (start, end, step)
            } else {
                (0, shape[d], 1)
            }
        })
        .collect();

    let new_shape: Vec<usize> = ranges
        .iter()
        .map(|&(s, e, step)| (e - s + step - 1) / step)
        .collect();
    let n: usize = new_shape.iter().product();
    let new_strides = crate::tensor::c_strides(&new_shape);

    let mut out = vec![0i8; n];
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

    RustyNumTensor {
        storage: TensorStorage::I8(Arc::new(out)),
        shape: new_shape,
    }
}

/// Expand (broadcast) an i8 tensor to a target shape.
fn expand_i8(tensor: RustyNumTensor, shape: Shape) -> RustyNumTensor {
    let target = shape.to_vec();
    if tensor.shape == target {
        return tensor;
    }

    let data = get_i8_data(&tensor);
    let src_strides = broadcast_strides_q(&tensor.shape, &target);
    let out_strides = crate::tensor::c_strides(&target);
    let n: usize = target.iter().product();

    let mut out = vec![0i8; n];
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

    RustyNumTensor {
        storage: TensorStorage::I8(Arc::new(out)),
        shape: target,
    }
}

/// Get i8 data from a RustyNumTensor, panicking if the storage is not I8.
fn get_i8_data(tensor: &RustyNumTensor) -> &[i8] {
    match &tensor.storage {
        TensorStorage::I8(v) => v.as_slice(),
        _ => panic!(
            "Expected I8 tensor for quantized operation, got {:?}",
            tensor.dtype()
        ),
    }
}

/// Compute broadcast strides from a source shape to an output shape.
fn broadcast_strides_q(src_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
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
            }
            // else: broadcast dim, stride stays 0
        }
        // else: implicitly broadcast (stride stays 0)
    }
    strides
}
