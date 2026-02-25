use alloc::string::String;

use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{Backend, DType, DeviceId, DeviceOps};
use burn_ir::{BackendIr, HandleKind, TensorHandle};
use burn_std::stub::Mutex;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::tensor::{RustyNumQTensor, RustyNumTensor};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// The device type for the RustyNum backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RustyNumDevice {
    /// The CPU device (only option for now).
    #[default]
    Cpu,
}

impl DeviceOps for RustyNumDevice {}

impl burn_backend::Device for RustyNumDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self::Cpu
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

/// Tensor backend that uses RustyNum for AVX-512 SIMD-accelerated tensor operations.
///
/// CPU-only. Uses explicit `portable_simd` kernels for dot products, reductions,
/// and element-wise operations, with cache-blocked Goto GEMM for matmul.
#[derive(Clone, Copy, Default, Debug)]
pub struct RustyNum;

impl Backend for RustyNum {
    type Device = RustyNumDevice;

    type FloatTensorPrimitive = RustyNumTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = RustyNumTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = RustyNumTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = RustyNumQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        String::from("rustynum")
    }

    fn seed(_device: &Self::Device, seed: u64) {
        let rng = StdRng::seed_from_u64(seed);
        let mut s = SEED.lock().unwrap();
        *s = Some(rng);
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> burn_backend::DTypeUsageSet {
        match dtype {
            DType::F64
            | DType::F32
            | DType::Flex32
            | DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8
            | DType::Bool => burn_backend::DTypeUsage::general(),
            DType::F16 | DType::BF16 => burn_backend::DTypeUsageSet::empty(),
            DType::QFloat(_) => burn_backend::DTypeUsageSet::empty(),
        }
    }
}

impl BackendIr for RustyNum {
    type Handle = HandleKind<Self>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        match handle.handle {
            HandleKind::Float(handle) => handle,
            _ => panic!("Expected float handle, got {}", handle.handle.name()),
        }
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        match handle.handle {
            HandleKind::Int(handle) => handle,
            _ => panic!("Expected int handle, got {}", handle.handle.name()),
        }
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        match handle.handle {
            HandleKind::Bool(handle) => handle,
            _ => panic!("Expected bool handle, got {}", handle.handle.name()),
        }
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        match handle.handle {
            HandleKind::Quantized(handle) => handle,
            _ => panic!("Expected quantized handle, got {}", handle.handle.name()),
        }
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        HandleKind::Float(tensor)
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        HandleKind::Int(tensor)
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        HandleKind::Bool(tensor)
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        HandleKind::Quantized(tensor)
    }
}
