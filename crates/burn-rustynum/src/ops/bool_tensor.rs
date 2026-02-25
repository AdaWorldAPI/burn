use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use burn_backend::backend::ExecutionError;
use burn_backend::ops::BoolTensorOps;
use burn_backend::tensor::{BoolElem, BoolTensor, FloatTensor, IntTensor};
use burn_backend::{Shape, TensorData};
use burn_std::tensor::Slice;

use crate::backend::{RustyNum, RustyNumDevice};
use crate::tensor::{RustyNumTensor, TensorStorage, c_strides};

impl BoolTensorOps<Self> for RustyNum {
    fn bool_empty(shape: Shape, device: &RustyNumDevice) -> BoolTensor<Self> {
        Self::bool_zeros(shape, device)
    }

    fn bool_zeros(shape: Shape, _device: &RustyNumDevice) -> BoolTensor<Self> {
        let n: usize = shape.num_elements();
        RustyNumTensor::from_bool(vec![false; n], shape.to_vec())
    }

    fn bool_ones(shape: Shape, _device: &RustyNumDevice) -> BoolTensor<Self> {
        let n: usize = shape.num_elements();
        RustyNumTensor::from_bool(vec![true; n], shape.to_vec())
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn bool_from_data(data: TensorData, _device: &RustyNumDevice) -> BoolTensor<Self> {
        RustyNumTensor::from_data(data)
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        let bools = tensor.as_bool();
        let int_data: Vec<i64> = bools.iter().map(|&b| if b { 1i64 } else { 0i64 }).collect();
        RustyNumTensor::from_i64(int_data, tensor.shape.clone())
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        let bools = tensor.as_bool();
        let float_data: Vec<f32> = bools
            .iter()
            .map(|&b| if b { 1.0f32 } else { 0.0f32 })
            .collect();
        RustyNumTensor::from_f32(float_data, tensor.shape.clone())
    }

    fn bool_device(_tensor: &BoolTensor<Self>) -> RustyNumDevice {
        RustyNumDevice::Cpu
    }

    fn bool_to_device(tensor: BoolTensor<Self>, _device: &RustyNumDevice) -> BoolTensor<Self> {
        tensor
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        tensor.reshape(shape.to_vec())
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        let data = tensor.as_bool();
        let shape = &tensor.shape;
        let rank = shape.len();
        let strides = c_strides(shape);

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
        let new_strides = c_strides(&new_shape);

        let mut out = vec![false; n];
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
        RustyNumTensor::from_bool(out, new_shape)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let shape = tensor.shape.clone();
        let rank = shape.len();
        let strides = c_strides(&shape);
        let val_data = value.as_bool();
        let val_strides = c_strides(&value.shape);

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
                    (start, end, s.step as usize)
                } else {
                    (0, shape[d], 1)
                }
            })
            .collect();

        let val_shape: Vec<usize> = ranges
            .iter()
            .map(|&(s, e, step)| (e - s + step - 1) / step)
            .collect();
        let n: usize = val_shape.iter().product();

        // Clone tensor for mutation
        let mut tensor = tensor;
        let data = match &mut tensor.storage {
            TensorStorage::Bool(v) => Arc::make_mut(v),
            _ => panic!("Expected Bool tensor"),
        };

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

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let data = tensor.as_bool();
        let mask_data = mask.as_bool();
        let val_data = value.as_bool();
        let out: Vec<bool> = data
            .iter()
            .zip(mask_data.iter())
            .zip(val_data.iter())
            .map(|((&t, &m), &v)| if m { v } else { t })
            .collect();
        RustyNumTensor::from_bool(out, tensor.shape.clone())
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolElem<Self>,
    ) -> BoolTensor<Self> {
        let data = tensor.as_bool();
        let mask_data = mask.as_bool();
        let out: Vec<bool> = data
            .iter()
            .zip(mask_data.iter())
            .map(|(&t, &m)| if m { value } else { t })
            .collect();
        RustyNumTensor::from_bool(out, tensor.shape.clone())
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        let data = tensor.as_bool();
        let idx_data = indices.as_i64();
        let shape = &tensor.shape;
        let strides = c_strides(shape);
        let idx_shape = &indices.shape;
        let idx_strides = c_strides(idx_shape);
        let n: usize = idx_shape.iter().product();

        let mut out = vec![false; n];
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
        RustyNumTensor::from_bool(out, idx_shape.clone())
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let idx_data = indices.as_i64();
        let val_data = value.as_bool();
        let shape = tensor.shape.clone();
        let strides = c_strides(&shape);
        let idx_shape = &indices.shape;
        let idx_strides = c_strides(idx_shape);
        let n: usize = idx_shape.iter().product();

        let mut tensor = tensor;
        let data = match &mut tensor.storage {
            TensorStorage::Bool(v) => Arc::make_mut(v),
            _ => panic!("Expected Bool tensor"),
        };

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
            data[dst_idx] = data[dst_idx] || val_data[i];
        }
        tensor
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let lhs_data = lhs.as_bool();
        let rhs_data = rhs.as_bool();
        let out: Vec<bool> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a == b)
            .collect();
        RustyNumTensor::from_bool(out, lhs.shape.clone())
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: BoolElem<Self>) -> BoolTensor<Self> {
        let data = lhs.as_bool();
        let out: Vec<bool> = data.iter().map(|&a| a == rhs).collect();
        RustyNumTensor::from_bool(out, lhs.shape.clone())
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let data = tensor.as_bool();
        let out: Vec<bool> = data.iter().map(|&a| !a).collect();
        RustyNumTensor::from_bool(out, tensor.shape.clone())
    }

    fn bool_and(tensor: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let lhs_data = tensor.as_bool();
        let rhs_data = rhs.as_bool();
        let out: Vec<bool> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        RustyNumTensor::from_bool(out, tensor.shape.clone())
    }

    fn bool_or(tensor: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let lhs_data = tensor.as_bool();
        let rhs_data = rhs.as_bool();
        let out: Vec<bool> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        RustyNumTensor::from_bool(out, tensor.shape.clone())
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        if dim1 == dim2 {
            return tensor;
        }
        let mut axes: Vec<usize> = (0..tensor.shape.len()).collect();
        axes.swap(dim1, dim2);
        Self::bool_permute(tensor, &axes)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let data = tensor.as_bool();
        let old_shape = &tensor.shape;
        let rank = old_shape.len();
        let old_strides = c_strides(old_shape);

        let new_shape: Vec<usize> = axes.iter().map(|&a| old_shape[a]).collect();
        let new_strides = c_strides(&new_shape);
        let n = tensor.num_elements();

        let mut out = vec![false; n];
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
        RustyNumTensor::from_bool(out, new_shape)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let data = tensor.as_bool();
        let shape = &tensor.shape;
        let strides = c_strides(shape);
        let n = tensor.num_elements();
        let rank = shape.len();

        let mut out = vec![false; n];
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
        RustyNumTensor::from_bool(out, shape.clone())
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let target = shape.to_vec();
        if tensor.shape == target {
            return tensor;
        }

        let data = tensor.as_bool();
        let src_strides = broadcast_strides(&tensor.shape, &target);
        let out_strides = c_strides(&target);
        let n: usize = target.iter().product();

        let mut out = vec![false; n];
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
        RustyNumTensor::from_bool(out, target)
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        let data = tensor.as_bool();
        let shape = &tensor.shape;
        let rank = shape.len();
        let strides = c_strides(shape);

        let dim_size = shape[dim];
        // Number of windows: max(0, ceil((dim_size - size) / step) + 1)
        // which is (dim_size - size) / step + 1 when dim_size >= size
        let num_windows = if dim_size >= size {
            (dim_size - size) / step + 1
        } else {
            0
        };

        // Output shape: [..., num_windows, size, ...]
        // The dim dimension is replaced by num_windows, and size is inserted after it.
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

        let mut out = vec![false; n];
        for i in 0..n {
            let mut rem = i;
            let mut src_idx = 0usize;

            let mut new_d = 0;
            for d in 0..rank {
                if d == dim {
                    // Window index
                    let window_coord = rem / new_strides[new_d];
                    rem %= new_strides[new_d];
                    new_d += 1;
                    // Position within window
                    let pos_coord = rem / new_strides[new_d];
                    rem %= new_strides[new_d];
                    new_d += 1;

                    let src_coord = window_coord * step + pos_coord;
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
        RustyNumTensor::from_bool(out, new_shape)
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Compute broadcast strides mapping source shape into target (broadcast) shape.
/// Dimensions of size 1 in the source get stride 0 (broadcast).
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
            }
            // else: broadcast dim, stride stays 0
        }
        // else: implicitly broadcast (stride stays 0)
    }
    strides
}
