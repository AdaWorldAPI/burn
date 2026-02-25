use alloc::vec;

use burn_backend::ops::{
    AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
    DeformConvOptions, InterpolateMode, InterpolateOptions, MaxPool2dBackward,
    MaxPool2dWithIndices, ModuleOps,
};
use burn_backend::ops::attention::attention_fallback;
use burn_backend::tensor::FloatTensor;

use crate::backend::RustyNum;
use crate::tensor::RustyNumTensor;

// ---------------------------------------------------------------------------
// Helper: compute pooling output size
// ---------------------------------------------------------------------------

fn pool_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> usize {
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let numerator = input_size + 2 * padding - effective_kernel;
    if ceil_mode {
        (numerator + stride - 1) / stride + 1
    } else {
        numerator / stride + 1
    }
}

// ---------------------------------------------------------------------------
// impl ModuleOps
// ---------------------------------------------------------------------------

impl ModuleOps<Self> for RustyNum {
    // ========================================================================
    // 1. conv2d -- im2col + matmul
    // ========================================================================
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let w_data = weight.as_f32();

        let batch = x.shape[0];
        let c_in = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let c_out = weight.shape[0];
        let c_in_per_group = weight.shape[1];
        let kh = weight.shape[2];
        let kw = weight.shape[3];

        let [stride_h, stride_w] = options.stride;
        let [pad_h, pad_w] = options.padding;
        let [dil_h, dil_w] = options.dilation;
        let groups = options.groups;

        let h_out = pool_output_size(h_in, kh, stride_h, pad_h, dil_h, false);
        let w_out = pool_output_size(w_in, kw, stride_w, pad_w, dil_w, false);

        let c_out_per_group = c_out / groups;

        let mut output = vec![0.0f32; batch * c_out * h_out * w_out];

        for b in 0..batch {
            for g in 0..groups {
                for oc in 0..c_out_per_group {
                    let abs_oc = g * c_out_per_group + oc;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut sum = 0.0f32;
                            for ic in 0..c_in_per_group {
                                let abs_ic = g * c_in_per_group + ic;
                                for fh in 0..kh {
                                    for fw in 0..kw {
                                        let ih =
                                            oh as isize * stride_h as isize + fh as isize * dil_h as isize - pad_h as isize;
                                        let iw =
                                            ow as isize * stride_w as isize + fw as isize * dil_w as isize - pad_w as isize;
                                        if ih >= 0
                                            && ih < h_in as isize
                                            && iw >= 0
                                            && iw < w_in as isize
                                        {
                                            let ih = ih as usize;
                                            let iw = iw as usize;
                                            let x_idx = ((b * c_in + abs_ic) * h_in + ih) * w_in + iw;
                                            let w_idx = ((abs_oc * c_in_per_group + ic) * kh + fh) * kw + fw;
                                            sum += x_data[x_idx] * w_data[w_idx];
                                        }
                                    }
                                }
                            }
                            let out_idx = ((b * c_out + abs_oc) * h_out + oh) * w_out + ow;
                            output[out_idx] = sum;
                        }
                    }
                }
            }
        }

        // Add bias
        if let Some(bias_t) = bias {
            let bias_data = bias_t.as_f32();
            for b in 0..batch {
                for oc in 0..c_out {
                    let bias_val = bias_data[oc];
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let idx = ((b * c_out + oc) * h_out + oh) * w_out + ow;
                            output[idx] += bias_val;
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, c_out, h_out, w_out])
    }

    // ========================================================================
    // 2. deform_conv2d -- bilinear sampling with offsets
    // ========================================================================
    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let offset_data = offset.as_f32();
        let w_data = weight.as_f32();

        let batch = x.shape[0];
        let c_in = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let c_out = weight.shape[0];
        let c_in_per_group = weight.shape[1];
        let kh = weight.shape[2];
        let kw = weight.shape[3];

        let [stride_h, stride_w] = options.stride;
        let [pad_h, pad_w] = options.padding;
        let [dil_h, dil_w] = options.dilation;
        let weight_groups = options.weight_groups;
        let offset_groups = options.offset_groups;

        let h_out = pool_output_size(h_in, kh, stride_h, pad_h, dil_h, false);
        let w_out = pool_output_size(w_in, kw, stride_w, pad_w, dil_w, false);

        let c_out_per_wg = c_out / weight_groups;
        let c_in_per_offset_group = c_in / offset_groups;

        let mask_data = mask.as_ref().map(|m| m.as_f32());

        let mut output = vec![0.0f32; batch * c_out * h_out * w_out];

        for b in 0..batch {
            for g in 0..weight_groups {
                for oc in 0..c_out_per_wg {
                    let abs_oc = g * c_out_per_wg + oc;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut sum = 0.0f32;
                            for ic in 0..c_in_per_group {
                                let abs_ic = g * c_in_per_group + ic;
                                let og = abs_ic / c_in_per_offset_group;
                                for fh in 0..kh {
                                    for fw in 0..kw {
                                        let offset_idx_base = ((b * offset_groups + og) * kh * kw + fh * kw + fw) * 2;
                                        let offset_h_idx = (offset_idx_base) * h_out * w_out
                                            + oh * w_out
                                            + ow;
                                        let offset_w_idx = (offset_idx_base + 1) * h_out * w_out
                                            + oh * w_out
                                            + ow;

                                        // Compute offset from flat [B, offset_groups*kh*kw*2, h_out, w_out]
                                        let off_h = offset_data
                                            .get(offset_h_idx)
                                            .copied()
                                            .unwrap_or(0.0);
                                        let off_w = offset_data
                                            .get(offset_w_idx)
                                            .copied()
                                            .unwrap_or(0.0);

                                        let ih_f = oh as f32 * stride_h as f32
                                            + fh as f32 * dil_h as f32
                                            - pad_h as f32
                                            + off_h;
                                        let iw_f = ow as f32 * stride_w as f32
                                            + fw as f32 * dil_w as f32
                                            - pad_w as f32
                                            + off_w;

                                        // Bilinear interpolation
                                        let val = bilinear_sample(
                                            x_data,
                                            b,
                                            abs_ic,
                                            h_in,
                                            w_in,
                                            c_in,
                                            ih_f,
                                            iw_f,
                                        );

                                        // Mask
                                        let m = if let Some(md) = &mask_data {
                                            let mask_idx = ((b * offset_groups + og) * kh * kw
                                                + fh * kw
                                                + fw)
                                                * h_out
                                                * w_out
                                                + oh * w_out
                                                + ow;
                                            md.get(mask_idx).copied().unwrap_or(1.0)
                                        } else {
                                            1.0
                                        };

                                        let w_idx = ((abs_oc * c_in_per_group + ic) * kh + fh) * kw + fw;
                                        sum += val * m * w_data[w_idx];
                                    }
                                }
                            }
                            let out_idx = ((b * c_out + abs_oc) * h_out + oh) * w_out + ow;
                            output[out_idx] = sum;
                        }
                    }
                }
            }
        }

        if let Some(bias_t) = bias {
            let bias_data = bias_t.as_f32();
            for b in 0..batch {
                for oc in 0..c_out {
                    let bv = bias_data[oc];
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let idx = ((b * c_out + oc) * h_out + oh) * w_out + ow;
                            output[idx] += bv;
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, c_out, h_out, w_out])
    }

    // ========================================================================
    // 3. deform_conv2d_backward
    // ========================================================================
    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let x_data = x.as_f32();
        let offset_data = offset.as_f32();
        let w_data = weight.as_f32();
        let grad_data = output_grad.as_f32();

        let batch = x.shape[0];
        let c_in = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let c_out = weight.shape[0];
        let c_in_per_group = weight.shape[1];
        let kh = weight.shape[2];
        let kw = weight.shape[3];

        let [stride_h, stride_w] = options.stride;
        let [pad_h, pad_w] = options.padding;
        let [dil_h, dil_w] = options.dilation;
        let weight_groups = options.weight_groups;
        let offset_groups = options.offset_groups;

        let h_out = output_grad.shape[2];
        let w_out = output_grad.shape[3];

        let c_out_per_wg = c_out / weight_groups;
        let c_in_per_offset_group = c_in / offset_groups;

        let mask_data = mask.as_ref().map(|m| m.as_f32());

        let mut x_grad = vec![0.0f32; batch * c_in * h_in * w_in];
        let mut offset_grad = vec![0.0f32; offset.shape.iter().product::<usize>()];
        let mut weight_grad = vec![0.0f32; weight.shape.iter().product::<usize>()];
        let mut mask_grad_data = mask
            .as_ref()
            .map(|m| vec![0.0f32; m.shape.iter().product::<usize>()]);
        let mut bias_grad_data = bias.as_ref().map(|_| vec![0.0f32; c_out]);

        for b in 0..batch {
            for g in 0..weight_groups {
                for oc in 0..c_out_per_wg {
                    let abs_oc = g * c_out_per_wg + oc;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let grad_val = grad_data[((b * c_out + abs_oc) * h_out + oh) * w_out + ow];

                            // bias grad
                            if let Some(ref mut bg) = bias_grad_data {
                                bg[abs_oc] += grad_val;
                            }

                            for ic in 0..c_in_per_group {
                                let abs_ic = g * c_in_per_group + ic;
                                let og = abs_ic / c_in_per_offset_group;
                                for fh in 0..kh {
                                    for fw in 0..kw {
                                        let offset_idx_base = ((b * offset_groups + og) * kh * kw + fh * kw + fw) * 2;
                                        let off_h_idx = offset_idx_base * h_out * w_out + oh * w_out + ow;
                                        let off_w_idx = (offset_idx_base + 1) * h_out * w_out + oh * w_out + ow;

                                        let off_h = offset_data.get(off_h_idx).copied().unwrap_or(0.0);
                                        let off_w = offset_data.get(off_w_idx).copied().unwrap_or(0.0);

                                        let ih_f = oh as f32 * stride_h as f32
                                            + fh as f32 * dil_h as f32
                                            - pad_h as f32
                                            + off_h;
                                        let iw_f = ow as f32 * stride_w as f32
                                            + fw as f32 * dil_w as f32
                                            - pad_w as f32
                                            + off_w;

                                        let m_val = if let Some(md) = &mask_data {
                                            let mask_idx = ((b * offset_groups + og) * kh * kw + fh * kw + fw)
                                                * h_out * w_out + oh * w_out + ow;
                                            md.get(mask_idx).copied().unwrap_or(1.0)
                                        } else {
                                            1.0
                                        };

                                        let val = bilinear_sample(x_data, b, abs_ic, h_in, w_in, c_in, ih_f, iw_f);
                                        let w_idx = ((abs_oc * c_in_per_group + ic) * kh + fh) * kw + fw;

                                        // weight grad
                                        weight_grad[w_idx] += grad_val * val * m_val;

                                        // x grad via bilinear backward
                                        let w_val = w_data[w_idx];
                                        bilinear_sample_backward_add(
                                            &mut x_grad,
                                            b, abs_ic, h_in, w_in, c_in,
                                            ih_f, iw_f,
                                            grad_val * w_val * m_val,
                                        );

                                        // offset grad (finite difference approximation)
                                        let eps = 0.1;
                                        let val_dh = bilinear_sample(x_data, b, abs_ic, h_in, w_in, c_in, ih_f + eps, iw_f);
                                        let val_dw = bilinear_sample(x_data, b, abs_ic, h_in, w_in, c_in, ih_f, iw_f + eps);
                                        let dval_dh = (val_dh - val) / eps;
                                        let dval_dw = (val_dw - val) / eps;

                                        if off_h_idx < offset_grad.len() {
                                            offset_grad[off_h_idx] += grad_val * w_val * m_val * dval_dh;
                                        }
                                        if off_w_idx < offset_grad.len() {
                                            offset_grad[off_w_idx] += grad_val * w_val * m_val * dval_dw;
                                        }

                                        // mask grad
                                        if let Some(ref mut mg) = mask_grad_data {
                                            let mask_idx = ((b * offset_groups + og) * kh * kw + fh * kw + fw)
                                                * h_out * w_out + oh * w_out + ow;
                                            if mask_idx < mg.len() {
                                                mg[mask_idx] += grad_val * val * w_data[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let x_grad_t = RustyNumTensor::from_f32(x_grad, x.shape.clone());
        let offset_grad_t = RustyNumTensor::from_f32(offset_grad, offset.shape.clone());
        let weight_grad_t = RustyNumTensor::from_f32(weight_grad, weight.shape.clone());
        let mask_grad_t = mask_grad_data.map(|mg| {
            RustyNumTensor::from_f32(mg, mask.as_ref().unwrap().shape.clone())
        });
        let bias_grad_t = bias_grad_data.map(|bg| {
            RustyNumTensor::from_f32(bg, vec![c_out])
        });

        DeformConv2dBackward::new(x_grad_t, offset_grad_t, weight_grad_t, mask_grad_t, bias_grad_t)
    }

    // ========================================================================
    // 4. conv3d -- direct nested loops
    // ========================================================================
    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let w_data = weight.as_f32();

        let batch = x.shape[0];
        let c_in = x.shape[1];
        let d_in = x.shape[2];
        let h_in = x.shape[3];
        let w_in = x.shape[4];

        let c_out = weight.shape[0];
        let c_in_per_group = weight.shape[1];
        let kd = weight.shape[2];
        let kh = weight.shape[3];
        let kw = weight.shape[4];

        let [stride_d, stride_h, stride_w] = options.stride;
        let [pad_d, pad_h, pad_w] = options.padding;
        let [dil_d, dil_h, dil_w] = options.dilation;
        let groups = options.groups;

        let d_out = pool_output_size(d_in, kd, stride_d, pad_d, dil_d, false);
        let h_out = pool_output_size(h_in, kh, stride_h, pad_h, dil_h, false);
        let w_out = pool_output_size(w_in, kw, stride_w, pad_w, dil_w, false);

        let c_out_per_group = c_out / groups;

        let mut output = vec![0.0f32; batch * c_out * d_out * h_out * w_out];

        for b in 0..batch {
            for g in 0..groups {
                for oc in 0..c_out_per_group {
                    let abs_oc = g * c_out_per_group + oc;
                    for od in 0..d_out {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let mut sum = 0.0f32;
                                for ic in 0..c_in_per_group {
                                    let abs_ic = g * c_in_per_group + ic;
                                    for fd in 0..kd {
                                        let id = od as isize * stride_d as isize
                                            + fd as isize * dil_d as isize
                                            - pad_d as isize;
                                        if id < 0 || id >= d_in as isize {
                                            continue;
                                        }
                                        let id = id as usize;
                                        for fh in 0..kh {
                                            let ih = oh as isize * stride_h as isize
                                                + fh as isize * dil_h as isize
                                                - pad_h as isize;
                                            if ih < 0 || ih >= h_in as isize {
                                                continue;
                                            }
                                            let ih = ih as usize;
                                            for fw in 0..kw {
                                                let iw = ow as isize * stride_w as isize
                                                    + fw as isize * dil_w as isize
                                                    - pad_w as isize;
                                                if iw < 0 || iw >= w_in as isize {
                                                    continue;
                                                }
                                                let iw = iw as usize;
                                                let x_idx = (((b * c_in + abs_ic) * d_in + id) * h_in + ih) * w_in + iw;
                                                let w_idx = (((abs_oc * c_in_per_group + ic) * kd + fd) * kh + fh) * kw + fw;
                                                sum += x_data[x_idx] * w_data[w_idx];
                                            }
                                        }
                                    }
                                }
                                let out_idx = (((b * c_out + abs_oc) * d_out + od) * h_out + oh) * w_out + ow;
                                output[out_idx] = sum;
                            }
                        }
                    }
                }
            }
        }

        if let Some(bias_t) = bias {
            let bias_data = bias_t.as_f32();
            for b in 0..batch {
                for oc in 0..c_out {
                    let bv = bias_data[oc];
                    for od in 0..d_out {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let idx = (((b * c_out + oc) * d_out + od) * h_out + oh) * w_out + ow;
                                output[idx] += bv;
                            }
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, c_out, d_out, h_out, w_out])
    }

    // ========================================================================
    // 5. conv_transpose2d
    // ========================================================================
    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let w_data = weight.as_f32();

        let batch = x.shape[0];
        let c_in = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        // weight shape: [c_in, c_out_per_group, kh, kw]
        let c_out_per_group = weight.shape[1];
        let kh = weight.shape[2];
        let kw = weight.shape[3];

        let [stride_h, stride_w] = options.stride;
        let [pad_h, pad_w] = options.padding;
        let [pad_out_h, pad_out_w] = options.padding_out;
        let [dil_h, dil_w] = options.dilation;
        let groups = options.groups;

        let c_in_per_group = c_in / groups;
        let c_out = c_out_per_group * groups;

        let h_out = (h_in - 1) * stride_h + dil_h * (kh - 1) + 1 - 2 * pad_h + pad_out_h;
        let w_out = (w_in - 1) * stride_w + dil_w * (kw - 1) + 1 - 2 * pad_w + pad_out_w;

        let mut output = vec![0.0f32; batch * c_out * h_out * w_out];

        for b in 0..batch {
            for g in 0..groups {
                for ic in 0..c_in_per_group {
                    let abs_ic = g * c_in_per_group + ic;
                    for ih in 0..h_in {
                        for iw in 0..w_in {
                            let x_val = x_data[((b * c_in + abs_ic) * h_in + ih) * w_in + iw];
                            for oc in 0..c_out_per_group {
                                let abs_oc = g * c_out_per_group + oc;
                                for fh in 0..kh {
                                    let oh_s = ih as isize * stride_h as isize
                                        + fh as isize * dil_h as isize
                                        - pad_h as isize;
                                    if oh_s < 0 || oh_s >= h_out as isize {
                                        continue;
                                    }
                                    let oh = oh_s as usize;
                                    for fw in 0..kw {
                                        let ow_s = iw as isize * stride_w as isize
                                            + fw as isize * dil_w as isize
                                            - pad_w as isize;
                                        if ow_s < 0 || ow_s >= w_out as isize {
                                            continue;
                                        }
                                        let ow = ow_s as usize;
                                        let w_idx = ((abs_ic * c_out_per_group + oc) * kh + fh) * kw + fw;
                                        let out_idx = ((b * c_out + abs_oc) * h_out + oh) * w_out + ow;
                                        output[out_idx] += x_val * w_data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(bias_t) = bias {
            let bias_data = bias_t.as_f32();
            for b in 0..batch {
                for oc in 0..c_out {
                    let bv = bias_data[oc];
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let idx = ((b * c_out + oc) * h_out + oh) * w_out + ow;
                            output[idx] += bv;
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, c_out, h_out, w_out])
    }

    // ========================================================================
    // 6. conv_transpose3d
    // ========================================================================
    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let w_data = weight.as_f32();

        let batch = x.shape[0];
        let c_in = x.shape[1];
        let d_in = x.shape[2];
        let h_in = x.shape[3];
        let w_in = x.shape[4];

        // weight: [c_in, c_out_per_group, kd, kh, kw]
        let c_out_per_group = weight.shape[1];
        let kd = weight.shape[2];
        let kh = weight.shape[3];
        let kw = weight.shape[4];

        let [stride_d, stride_h, stride_w] = options.stride;
        let [pad_d, pad_h, pad_w] = options.padding;
        let [pad_out_d, pad_out_h, pad_out_w] = options.padding_out;
        let [dil_d, dil_h, dil_w] = options.dilation;
        let groups = options.groups;

        let c_in_per_group = c_in / groups;
        let c_out = c_out_per_group * groups;

        let d_out = (d_in - 1) * stride_d + dil_d * (kd - 1) + 1 - 2 * pad_d + pad_out_d;
        let h_out = (h_in - 1) * stride_h + dil_h * (kh - 1) + 1 - 2 * pad_h + pad_out_h;
        let w_out = (w_in - 1) * stride_w + dil_w * (kw - 1) + 1 - 2 * pad_w + pad_out_w;

        let mut output = vec![0.0f32; batch * c_out * d_out * h_out * w_out];

        for b in 0..batch {
            for g in 0..groups {
                for ic in 0..c_in_per_group {
                    let abs_ic = g * c_in_per_group + ic;
                    for id in 0..d_in {
                        for ih in 0..h_in {
                            for iw in 0..w_in {
                                let x_val = x_data
                                    [(((b * c_in + abs_ic) * d_in + id) * h_in + ih) * w_in + iw];
                                for oc in 0..c_out_per_group {
                                    let abs_oc = g * c_out_per_group + oc;
                                    for fd in 0..kd {
                                        let od_s = id as isize * stride_d as isize
                                            + fd as isize * dil_d as isize
                                            - pad_d as isize;
                                        if od_s < 0 || od_s >= d_out as isize {
                                            continue;
                                        }
                                        let od = od_s as usize;
                                        for fh in 0..kh {
                                            let oh_s = ih as isize * stride_h as isize
                                                + fh as isize * dil_h as isize
                                                - pad_h as isize;
                                            if oh_s < 0 || oh_s >= h_out as isize {
                                                continue;
                                            }
                                            let oh = oh_s as usize;
                                            for fw in 0..kw {
                                                let ow_s = iw as isize * stride_w as isize
                                                    + fw as isize * dil_w as isize
                                                    - pad_w as isize;
                                                if ow_s < 0 || ow_s >= w_out as isize {
                                                    continue;
                                                }
                                                let ow = ow_s as usize;
                                                let w_idx = (((abs_ic * c_out_per_group + oc) * kd + fd) * kh + fh) * kw + fw;
                                                let out_idx = (((b * c_out + abs_oc) * d_out + od) * h_out + oh) * w_out + ow;
                                                output[out_idx] += x_val * w_data[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(bias_t) = bias {
            let bias_data = bias_t.as_f32();
            for b in 0..batch {
                for oc in 0..c_out {
                    let bv = bias_data[oc];
                    for od in 0..d_out {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let idx = (((b * c_out + oc) * d_out + od) * h_out + oh) * w_out + ow;
                                output[idx] += bv;
                            }
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, c_out, d_out, h_out, w_out])
    }

    // ========================================================================
    // 7. avg_pool2d
    // ========================================================================
    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let [kh, kw] = kernel_size;
        let [sh, sw] = stride;
        let [ph, pw] = padding;

        let h_out = pool_output_size(h_in, kh, sh, ph, 1, ceil_mode);
        let w_out = pool_output_size(w_in, kw, sw, pw, 1, ceil_mode);

        let mut output = vec![0.0f32; batch * channels * h_out * w_out];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        let mut count = 0usize;
                        for fh in 0..kh {
                            let ih = oh as isize * sh as isize + fh as isize - ph as isize;
                            if ih < 0 || ih >= h_in as isize {
                                if count_include_pad {
                                    count += kw; // count pad pixels in this row
                                }
                                continue;
                            }
                            let ih = ih as usize;
                            for fw in 0..kw {
                                let iw = ow as isize * sw as isize + fw as isize - pw as isize;
                                if iw < 0 || iw >= w_in as isize {
                                    if count_include_pad {
                                        count += 1;
                                    }
                                    continue;
                                }
                                let iw = iw as usize;
                                let idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                                sum += x_data[idx];
                                count += 1;
                            }
                        }
                        let divisor = if count_include_pad {
                            kh * kw
                        } else if count > 0 {
                            count
                        } else {
                            1
                        };
                        let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                        output[out_idx] = sum / divisor as f32;
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, channels, h_out, w_out])
    }

    // ========================================================================
    // 8. avg_pool2d_backward
    // ========================================================================
    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        _ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let grad_data = grad.as_f32();

        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let [kh, kw] = kernel_size;
        let [sh, sw] = stride;
        let [ph, pw] = padding;

        let h_out = grad.shape[2];
        let w_out = grad.shape[3];

        let mut x_grad = vec![0.0f32; batch * channels * h_in * w_in];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        // Count valid positions for this output element
                        let mut count = 0usize;
                        for fh in 0..kh {
                            let ih = oh as isize * sh as isize + fh as isize - ph as isize;
                            if ih < 0 || ih >= h_in as isize {
                                if count_include_pad {
                                    count += kw;
                                }
                                continue;
                            }
                            for fw in 0..kw {
                                let iw = ow as isize * sw as isize + fw as isize - pw as isize;
                                if iw < 0 || iw >= w_in as isize {
                                    if count_include_pad {
                                        count += 1;
                                    }
                                    continue;
                                }
                                count += 1;
                            }
                        }

                        let divisor = if count_include_pad {
                            kh * kw
                        } else if count > 0 {
                            count
                        } else {
                            1
                        };

                        let g_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                        let grad_val = grad_data[g_idx] / divisor as f32;

                        for fh in 0..kh {
                            let ih = oh as isize * sh as isize + fh as isize - ph as isize;
                            if ih < 0 || ih >= h_in as isize {
                                continue;
                            }
                            let ih = ih as usize;
                            for fw in 0..kw {
                                let iw = ow as isize * sw as isize + fw as isize - pw as isize;
                                if iw < 0 || iw >= w_in as isize {
                                    continue;
                                }
                                let iw = iw as usize;
                                let x_idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                                x_grad[x_idx] += grad_val;
                            }
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(x_grad, vec![batch, channels, h_in, w_in])
    }

    // ========================================================================
    // 9. adaptive_avg_pool2d
    // ========================================================================
    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let [h_out, w_out] = output_size;

        let mut output = vec![0.0f32; batch * channels * h_out * w_out];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    let ih_start = (oh * h_in) / h_out;
                    let ih_end = ((oh + 1) * h_in) / h_out;
                    for ow in 0..w_out {
                        let iw_start = (ow * w_in) / w_out;
                        let iw_end = ((ow + 1) * w_in) / w_out;

                        let mut sum = 0.0f32;
                        let count = (ih_end - ih_start) * (iw_end - iw_start);
                        for ih in ih_start..ih_end {
                            for iw in iw_start..iw_end {
                                let idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                                sum += x_data[idx];
                            }
                        }
                        let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                        output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, channels, h_out, w_out])
    }

    // ========================================================================
    // 10. adaptive_avg_pool2d_backward
    // ========================================================================
    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let grad_data = grad.as_f32();

        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let h_out = grad.shape[2];
        let w_out = grad.shape[3];

        let mut x_grad = vec![0.0f32; batch * channels * h_in * w_in];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    let ih_start = (oh * h_in) / h_out;
                    let ih_end = ((oh + 1) * h_in) / h_out;
                    for ow in 0..w_out {
                        let iw_start = (ow * w_in) / w_out;
                        let iw_end = ((ow + 1) * w_in) / w_out;

                        let count = (ih_end - ih_start) * (iw_end - iw_start);
                        let g_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                        let grad_val = if count > 0 {
                            grad_data[g_idx] / count as f32
                        } else {
                            0.0
                        };

                        for ih in ih_start..ih_end {
                            for iw in iw_start..iw_end {
                                let x_idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                                x_grad[x_idx] += grad_val;
                            }
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(x_grad, vec![batch, channels, h_in, w_in])
    }

    // ========================================================================
    // 11. max_pool2d
    // ========================================================================
    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let [kh, kw] = kernel_size;
        let [sh, sw] = stride;
        let [ph, pw] = padding;
        let [dh, dw] = dilation;

        let h_out = pool_output_size(h_in, kh, sh, ph, dh, ceil_mode);
        let w_out = pool_output_size(w_in, kw, sw, pw, dw, ceil_mode);

        let mut output = vec![f32::NEG_INFINITY; batch * channels * h_out * w_out];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;
                        for fh in 0..kh {
                            let ih = oh as isize * sh as isize + fh as isize * dh as isize - ph as isize;
                            if ih < 0 || ih >= h_in as isize {
                                continue;
                            }
                            let ih = ih as usize;
                            for fw in 0..kw {
                                let iw = ow as isize * sw as isize + fw as isize * dw as isize - pw as isize;
                                if iw < 0 || iw >= w_in as isize {
                                    continue;
                                }
                                let iw = iw as usize;
                                let idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                                let val = x_data[idx];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                        let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, channels, h_out, w_out])
    }

    // ========================================================================
    // 12. max_pool2d_with_indices
    // ========================================================================
    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let x_data = x.as_f32();
        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let [kh, kw] = kernel_size;
        let [sh, sw] = stride;
        let [ph, pw] = padding;
        let [dh, dw] = dilation;

        let h_out = pool_output_size(h_in, kh, sh, ph, dh, ceil_mode);
        let w_out = pool_output_size(w_in, kw, sw, pw, dw, ceil_mode);

        let out_len = batch * channels * h_out * w_out;
        let mut output = vec![f32::NEG_INFINITY; out_len];
        let mut indices = vec![0i64; out_len];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0i64;
                        for fh in 0..kh {
                            let ih = oh as isize * sh as isize + fh as isize * dh as isize - ph as isize;
                            if ih < 0 || ih >= h_in as isize {
                                continue;
                            }
                            let ih = ih as usize;
                            for fw in 0..kw {
                                let iw = ow as isize * sw as isize + fw as isize * dw as isize - pw as isize;
                                if iw < 0 || iw >= w_in as isize {
                                    continue;
                                }
                                let iw = iw as usize;
                                let idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                                let val = x_data[idx];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = (ih * w_in + iw) as i64;
                                }
                            }
                        }
                        let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                        output[out_idx] = max_val;
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }

        let out_shape = vec![batch, channels, h_out, w_out];
        MaxPool2dWithIndices::new(
            RustyNumTensor::from_f32(output, out_shape.clone()),
            RustyNumTensor::from_i64(indices, out_shape),
        )
    }

    // ========================================================================
    // 13. max_pool2d_with_indices_backward
    // ========================================================================
    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: <Self as burn_backend::Backend>::IntTensorPrimitive,
    ) -> MaxPool2dBackward<Self> {
        let grad_data = output_grad.as_f32();
        let idx_data = indices.as_i64();

        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];

        let h_out = output_grad.shape[2];
        let w_out = output_grad.shape[3];

        let mut x_grad = vec![0.0f32; batch * channels * h_in * w_in];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                        let grad_val = grad_data[g_idx];
                        let flat_idx = idx_data[g_idx] as usize; // ih * w_in + iw
                        let x_idx = (b * channels + c) * h_in * w_in + flat_idx;
                        if x_idx < x_grad.len() {
                            x_grad[x_idx] += grad_val;
                        }
                    }
                }
            }
        }

        MaxPool2dBackward::new(RustyNumTensor::from_f32(
            x_grad,
            vec![batch, channels, h_in, w_in],
        ))
    }

    // ========================================================================
    // 14. interpolate -- nearest, bilinear, bicubic
    // ========================================================================
    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let x_data = x.as_f32();
        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];
        let [h_out, w_out] = output_size;

        let mut output = vec![0.0f32; batch * channels * h_out * w_out];

        match options.mode {
            InterpolateMode::Nearest => {
                let scale_h = h_in as f32 / h_out as f32;
                let scale_w = w_in as f32 / w_out as f32;
                for b in 0..batch {
                    for c in 0..channels {
                        for oh in 0..h_out {
                            let ih = ((oh as f32 + 0.5) * scale_h).floor() as usize;
                            let ih = ih.min(h_in - 1);
                            for ow in 0..w_out {
                                let iw = ((ow as f32 + 0.5) * scale_w).floor() as usize;
                                let iw = iw.min(w_in - 1);
                                let src_idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                                let dst_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                                output[dst_idx] = x_data[src_idx];
                            }
                        }
                    }
                }
            }
            InterpolateMode::Bilinear => {
                for b in 0..batch {
                    for c in 0..channels {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let (src_h, src_w) = interpolate_coord(
                                    oh, ow, h_in, w_in, h_out, w_out, options.align_corners,
                                );

                                let h0 = src_h.floor();
                                let w0 = src_w.floor();
                                let h1 = h0 + 1.0;
                                let w1 = w0 + 1.0;
                                let dh = src_h - h0;
                                let dw = src_w - w0;

                                let get = |h: f32, w: f32| -> f32 {
                                    let hi = h as isize;
                                    let wi = w as isize;
                                    if hi >= 0
                                        && hi < h_in as isize
                                        && wi >= 0
                                        && wi < w_in as isize
                                    {
                                        x_data[((b * channels + c) * h_in + hi as usize) * w_in
                                            + wi as usize]
                                    } else {
                                        0.0
                                    }
                                };

                                let val = get(h0, w0) * (1.0 - dh) * (1.0 - dw)
                                    + get(h0, w1) * (1.0 - dh) * dw
                                    + get(h1, w0) * dh * (1.0 - dw)
                                    + get(h1, w1) * dh * dw;

                                let dst_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                                output[dst_idx] = val;
                            }
                        }
                    }
                }
            }
            InterpolateMode::Bicubic => {
                for b in 0..batch {
                    for c in 0..channels {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let (src_h, src_w) = interpolate_coord(
                                    oh, ow, h_in, w_in, h_out, w_out, options.align_corners,
                                );

                                let h_floor = src_h.floor() as isize;
                                let w_floor = src_w.floor() as isize;
                                let dh = src_h - h_floor as f32;
                                let dw = src_w - w_floor as f32;

                                let get = |h: isize, w: isize| -> f32 {
                                    let hc = h.clamp(0, h_in as isize - 1) as usize;
                                    let wc = w.clamp(0, w_in as isize - 1) as usize;
                                    x_data[((b * channels + c) * h_in + hc) * w_in + wc]
                                };

                                let mut val = 0.0f32;
                                for j in -1..=2isize {
                                    let wt_h = cubic_weight(j as f32 - dh);
                                    for i in -1..=2isize {
                                        let wt_w = cubic_weight(i as f32 - dw);
                                        val += get(h_floor + j, w_floor + i) * wt_h * wt_w;
                                    }
                                }

                                let dst_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                                output[dst_idx] = val;
                            }
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(output, vec![batch, channels, h_out, w_out])
    }

    // ========================================================================
    // 15. interpolate_backward
    // ========================================================================
    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let grad_data = grad.as_f32();

        let batch = x.shape[0];
        let channels = x.shape[1];
        let h_in = x.shape[2];
        let w_in = x.shape[3];
        let [h_out, w_out] = output_size;

        let mut x_grad = vec![0.0f32; batch * channels * h_in * w_in];

        match options.mode {
            InterpolateMode::Nearest => {
                let scale_h = h_in as f32 / h_out as f32;
                let scale_w = w_in as f32 / w_out as f32;
                for b in 0..batch {
                    for c in 0..channels {
                        for oh in 0..h_out {
                            let ih = ((oh as f32 + 0.5) * scale_h).floor() as usize;
                            let ih = ih.min(h_in - 1);
                            for ow in 0..w_out {
                                let iw = ((ow as f32 + 0.5) * scale_w).floor() as usize;
                                let iw = iw.min(w_in - 1);
                                let g_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                                let x_idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                                x_grad[x_idx] += grad_data[g_idx];
                            }
                        }
                    }
                }
            }
            InterpolateMode::Bilinear => {
                for b in 0..batch {
                    for c in 0..channels {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let (src_h, src_w) = interpolate_coord(
                                    oh, ow, h_in, w_in, h_out, w_out, options.align_corners,
                                );

                                let h0 = src_h.floor() as isize;
                                let w0 = src_w.floor() as isize;
                                let dh = src_h - h0 as f32;
                                let dw = src_w - w0 as f32;

                                let g_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                                let g_val = grad_data[g_idx];

                                // Distribute gradient to 4 neighbors
                                for (dhi, wh) in [(h0, 1.0 - dh), (h0 + 1, dh)] {
                                    if dhi >= 0 && dhi < h_in as isize {
                                        for (dwi, ww) in [(w0, 1.0 - dw), (w0 + 1, dw)] {
                                            if dwi >= 0 && dwi < w_in as isize {
                                                let x_idx = ((b * channels + c) * h_in
                                                    + dhi as usize)
                                                    * w_in
                                                    + dwi as usize;
                                                x_grad[x_idx] += g_val * wh * ww;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            InterpolateMode::Bicubic => {
                for b in 0..batch {
                    for c in 0..channels {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let (src_h, src_w) = interpolate_coord(
                                    oh, ow, h_in, w_in, h_out, w_out, options.align_corners,
                                );

                                let h_floor = src_h.floor() as isize;
                                let w_floor = src_w.floor() as isize;
                                let dh = src_h - h_floor as f32;
                                let dw = src_w - w_floor as f32;

                                let g_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                                let g_val = grad_data[g_idx];

                                for j in -1..=2isize {
                                    let wt_h = cubic_weight(j as f32 - dh);
                                    let hi = (h_floor + j).clamp(0, h_in as isize - 1) as usize;
                                    for i in -1..=2isize {
                                        let wt_w = cubic_weight(i as f32 - dw);
                                        let wi =
                                            (w_floor + i).clamp(0, w_in as isize - 1) as usize;
                                        let x_idx = ((b * channels + c) * h_in + hi) * w_in + wi;
                                        x_grad[x_idx] += g_val * wt_h * wt_w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        RustyNumTensor::from_f32(x_grad, vec![batch, channels, h_in, w_in])
    }

    // ========================================================================
    // 16. attention -- use fallback
    // ========================================================================
    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<burn_backend::tensor::BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        attention_fallback::<RustyNum>(query, key, value, mask, attn_bias, options)
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Bilinear sampling from a 4D tensor at fractional (h, w) coordinates.
/// Returns 0 for out-of-bounds positions.
fn bilinear_sample(
    data: &[f32],
    b: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    c_total: usize,
    h: f32,
    w: f32,
) -> f32 {
    let h0 = h.floor() as isize;
    let w0 = w.floor() as isize;
    let h1 = h0 + 1;
    let w1 = w0 + 1;
    let dh = h - h0 as f32;
    let dw = w - w0 as f32;

    let get = |hi: isize, wi: isize| -> f32 {
        if hi >= 0 && hi < h_in as isize && wi >= 0 && wi < w_in as isize {
            data[((b * c_total + c) * h_in + hi as usize) * w_in + wi as usize]
        } else {
            0.0
        }
    };

    get(h0, w0) * (1.0 - dh) * (1.0 - dw)
        + get(h0, w1) * (1.0 - dh) * dw
        + get(h1, w0) * dh * (1.0 - dw)
        + get(h1, w1) * dh * dw
}

/// Backward pass of bilinear sampling: scatter-add gradient to the 4 neighbors.
fn bilinear_sample_backward_add(
    grad: &mut [f32],
    b: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    c_total: usize,
    h: f32,
    w: f32,
    grad_val: f32,
) {
    let h0 = h.floor() as isize;
    let w0 = w.floor() as isize;
    let h1 = h0 + 1;
    let w1 = w0 + 1;
    let dh = h - h0 as f32;
    let dw = w - w0 as f32;

    let base = (b * c_total + c) * h_in;

    let mut add = |hi: isize, wi: isize, weight: f32| {
        if hi >= 0 && hi < h_in as isize && wi >= 0 && wi < w_in as isize {
            let idx = (base + hi as usize) * w_in + wi as usize;
            grad[idx] += grad_val * weight;
        }
    };

    add(h0, w0, (1.0 - dh) * (1.0 - dw));
    add(h0, w1, (1.0 - dh) * dw);
    add(h1, w0, dh * (1.0 - dw));
    add(h1, w1, dh * dw);
}

/// Compute source coordinates for interpolation.
fn interpolate_coord(
    oh: usize,
    ow: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    align_corners: bool,
) -> (f32, f32) {
    if align_corners {
        let src_h = if h_out > 1 {
            oh as f32 * (h_in as f32 - 1.0) / (h_out as f32 - 1.0)
        } else {
            0.0
        };
        let src_w = if w_out > 1 {
            ow as f32 * (w_in as f32 - 1.0) / (w_out as f32 - 1.0)
        } else {
            0.0
        };
        (src_h, src_w)
    } else {
        let src_h = (oh as f32 + 0.5) * (h_in as f32 / h_out as f32) - 0.5;
        let src_w = (ow as f32 + 0.5) * (w_in as f32 / w_out as f32) - 0.5;
        (src_h, src_w)
    }
}

/// Cubic interpolation weight (Keys / Catmull-Rom with a = -0.75).
fn cubic_weight(x: f32) -> f32 {
    let a = -0.75f32;
    let ax = x.abs();
    if ax <= 1.0 {
        (a + 2.0) * ax * ax * ax - (a + 3.0) * ax * ax + 1.0
    } else if ax < 2.0 {
        a * ax * ax * ax - 5.0 * a * ax * ax + 8.0 * a * ax - 4.0 * a
    } else {
        0.0
    }
}
