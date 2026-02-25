use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Shape, TensorData, TensorMetadata};
use burn_backend::ElementConversion;
use burn_backend::quantization::{QuantScheme, QParams};
use burn_backend::QTensorPrimitive;

/// Typed inner storage for a RustyNum tensor.
///
/// Uses `Arc<Vec<T>>` for COW semantics: mutations check `Arc::get_mut()`
/// and clone-on-write if shared.
#[derive(Debug, Clone)]
pub(crate) enum TensorStorage {
    F64(Arc<Vec<f64>>),
    F32(Arc<Vec<f32>>),
    I64(Arc<Vec<i64>>),
    I32(Arc<Vec<i32>>),
    I16(Arc<Vec<i16>>),
    I8(Arc<Vec<i8>>),
    U64(Arc<Vec<u64>>),
    U32(Arc<Vec<u32>>),
    U16(Arc<Vec<u16>>),
    U8(Arc<Vec<u8>>),
    Bool(Arc<Vec<bool>>),
}

/// Tensor primitive for the RustyNum backend.
///
/// Flat contiguous storage + shape metadata. All SIMD kernels
/// require contiguous memory; strides are metadata-only.
#[derive(Debug, Clone)]
pub struct RustyNumTensor {
    pub(crate) storage: TensorStorage,
    pub(crate) shape: Vec<usize>,
}

impl RustyNumTensor {
    /// Create from TensorData (the burn-standard interchange type).
    pub fn from_data(data: TensorData) -> Self {
        let shape = data.shape.clone();

        macro_rules! from_dtype {
            ($data:expr, [$($dtype:ident => $ty:ty => $variant:ident),*]) => {
                match $data.dtype {
                    $(DType::$dtype => {
                        let vec = $data.into_vec::<$ty>().expect(concat!("Expected ", stringify!($ty)));
                        RustyNumTensor {
                            storage: TensorStorage::$variant(Arc::new(vec)),
                            shape,
                        }
                    },)*
                    DType::Flex32 => {
                        let vec = $data.into_vec::<f32>().expect("Expected f32 for Flex32");
                        RustyNumTensor {
                            storage: TensorStorage::F32(Arc::new(vec)),
                            shape,
                        }
                    },
                    other => panic!("Unsupported dtype: {:?}", other),
                }
            };
        }

        from_dtype!(data, [
            F64 => f64 => F64,
            F32 => f32 => F32,
            I64 => i64 => I64,
            I32 => i32 => I32,
            I16 => i16 => I16,
            I8 => i8 => I8,
            U64 => u64 => U64,
            U32 => u32 => U32,
            U16 => u16 => U16,
            U8 => u8 => U8,
            Bool => bool => Bool
        ])
    }

    /// Convert to TensorData for interchange.
    pub fn into_data(self) -> TensorData {
        let shape = Shape::from(self.shape);
        match self.storage {
            TensorStorage::F64(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::F32(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::I64(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::I32(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::I16(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::I8(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::U64(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::U32(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::U16(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::U8(data) => TensorData::new(unwrap_arc(data), shape),
            TensorStorage::Bool(data) => TensorData::new(unwrap_arc(data), shape),
        }
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Create a zeros tensor of the given shape and dtype.
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let n: usize = shape.iter().product();
        let storage = match dtype {
            DType::F64 => TensorStorage::F64(Arc::new(vec![0.0f64; n])),
            DType::F32 | DType::Flex32 => TensorStorage::F32(Arc::new(vec![0.0f32; n])),
            DType::I64 => TensorStorage::I64(Arc::new(vec![0i64; n])),
            DType::I32 => TensorStorage::I32(Arc::new(vec![0i32; n])),
            DType::I16 => TensorStorage::I16(Arc::new(vec![0i16; n])),
            DType::I8 => TensorStorage::I8(Arc::new(vec![0i8; n])),
            DType::U64 => TensorStorage::U64(Arc::new(vec![0u64; n])),
            DType::U32 => TensorStorage::U32(Arc::new(vec![0u32; n])),
            DType::U16 => TensorStorage::U16(Arc::new(vec![0u16; n])),
            DType::U8 => TensorStorage::U8(Arc::new(vec![0u8; n])),
            DType::Bool => TensorStorage::Bool(Arc::new(vec![false; n])),
            other => panic!("Unsupported dtype for zeros: {:?}", other),
        };
        Self { storage, shape }
    }

    /// Create an f32 tensor from a flat vec and shape.
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            storage: TensorStorage::F32(Arc::new(data)),
            shape,
        }
    }

    /// Create an f64 tensor from a flat vec and shape.
    pub fn from_f64(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self {
            storage: TensorStorage::F64(Arc::new(data)),
            shape,
        }
    }

    /// Create an i64 tensor.
    pub fn from_i64(data: Vec<i64>, shape: Vec<usize>) -> Self {
        Self {
            storage: TensorStorage::I64(Arc::new(data)),
            shape,
        }
    }

    /// Create a bool tensor.
    pub fn from_bool(data: Vec<bool>, shape: Vec<usize>) -> Self {
        Self {
            storage: TensorStorage::Bool(Arc::new(data)),
            shape,
        }
    }

    /// Get f32 data, panics if wrong dtype.
    pub fn as_f32(&self) -> &[f32] {
        match &self.storage {
            TensorStorage::F32(v) => v.as_slice(),
            _ => panic!("Expected F32 tensor, got {:?}", self.dtype()),
        }
    }

    /// Get f64 data, panics if wrong dtype.
    pub fn as_f64(&self) -> &[f64] {
        match &self.storage {
            TensorStorage::F64(v) => v.as_slice(),
            _ => panic!("Expected F64 tensor, got {:?}", self.dtype()),
        }
    }

    /// Get mutable f32 data (COW: clones if shared).
    pub fn as_f32_mut(&mut self) -> &mut Vec<f32> {
        match &mut self.storage {
            TensorStorage::F32(v) => Arc::make_mut(v),
            other => panic!("Expected F32 tensor, got {:?}", storage_dtype_name(other)),
        }
    }

    /// Get mutable f64 data (COW: clones if shared).
    pub fn as_f64_mut(&mut self) -> &mut Vec<f64> {
        match &mut self.storage {
            TensorStorage::F64(v) => Arc::make_mut(v),
            other => panic!("Expected F64 tensor, got {:?}", storage_dtype_name(other)),
        }
    }

    /// Get i64 data.
    pub fn as_i64(&self) -> &[i64] {
        match &self.storage {
            TensorStorage::I64(v) => v.as_slice(),
            _ => panic!("Expected I64 tensor, got {:?}", self.dtype()),
        }
    }

    /// Get bool data.
    pub fn as_bool(&self) -> &[bool] {
        match &self.storage {
            TensorStorage::Bool(v) => v.as_slice(),
            _ => panic!("Expected Bool tensor, got {:?}", self.dtype()),
        }
    }

    /// Reshape (metadata-only, no copy if contiguous).
    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            self.num_elements(),
            shape.iter().product::<usize>(),
            "Reshape size mismatch"
        );
        self.shape = shape;
        self
    }

    /// Cast the storage to a target dtype.
    pub fn cast_to(self, target: DType) -> Self {
        if self.dtype() == target {
            return self;
        }

        let shape = self.shape.clone();

        macro_rules! cast_from {
            ($src:expr, $target_dtype:expr, [$($dt:ident => $ty:ty => $var:ident),*]) => {
                match $target_dtype {
                    $(DType::$dt => {
                        let vec: Vec<$ty> = $src.iter().map(|&x| x.elem()).collect();
                        TensorStorage::$var(Arc::new(vec))
                    },)*
                    other => panic!("Unsupported cast target: {:?}", other),
                }
            };
        }

        // Extract source data and cast element-by-element
        let new_storage = match &self.storage {
            TensorStorage::F32(data) => cast_from!(data, target, [
                F64 => f64 => F64, F32 => f32 => F32,
                I64 => i64 => I64, I32 => i32 => I32,
                Bool => bool => Bool
            ]),
            TensorStorage::F64(data) => cast_from!(data, target, [
                F64 => f64 => F64, F32 => f32 => F32,
                I64 => i64 => I64, I32 => i32 => I32,
                Bool => bool => Bool
            ]),
            TensorStorage::I64(data) => cast_from!(data, target, [
                F64 => f64 => F64, F32 => f32 => F32,
                I64 => i64 => I64, I32 => i32 => I32,
                Bool => bool => Bool
            ]),
            TensorStorage::I32(data) => cast_from!(data, target, [
                F64 => f64 => F64, F32 => f32 => F32,
                I64 => i64 => I64, I32 => i32 => I32,
                Bool => bool => Bool
            ]),
            TensorStorage::Bool(data) => {
                let bools = data.as_slice();
                match target {
                    DType::F32 => TensorStorage::F32(Arc::new(bools.iter().map(|&b| if b { 1.0f32 } else { 0.0 }).collect())),
                    DType::F64 => TensorStorage::F64(Arc::new(bools.iter().map(|&b| if b { 1.0f64 } else { 0.0 }).collect())),
                    DType::I64 => TensorStorage::I64(Arc::new(bools.iter().map(|&b| if b { 1i64 } else { 0 }).collect())),
                    DType::I32 => TensorStorage::I32(Arc::new(bools.iter().map(|&b| if b { 1i32 } else { 0 }).collect())),
                    _ => panic!("Unsupported bool cast to {:?}", target),
                }
            }
            _ => panic!("Unsupported source dtype for cast: {:?}", self.dtype()),
        };

        Self { storage: new_storage, shape }
    }
}

impl TensorMetadata for RustyNumTensor {
    fn dtype(&self) -> DType {
        match &self.storage {
            TensorStorage::F64(_) => DType::F64,
            TensorStorage::F32(_) => DType::F32,
            TensorStorage::I64(_) => DType::I64,
            TensorStorage::I32(_) => DType::I32,
            TensorStorage::I16(_) => DType::I16,
            TensorStorage::I8(_) => DType::I8,
            TensorStorage::U64(_) => DType::U64,
            TensorStorage::U32(_) => DType::U32,
            TensorStorage::U16(_) => DType::U16,
            TensorStorage::U8(_) => DType::U8,
            TensorStorage::Bool(_) => DType::Bool,
        }
    }

    fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// A quantized tensor for the RustyNum backend.
#[derive(Clone, Debug)]
pub struct RustyNumQTensor {
    /// The quantized tensor.
    pub qtensor: RustyNumTensor,
    /// The quantization scheme.
    pub scheme: QuantScheme,
    /// The quantization parameters.
    pub qparams: Vec<QParams<f32>>,
}

impl QTensorPrimitive for RustyNumQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }

    fn default_scheme() -> QuantScheme {
        QuantScheme::default()
    }
}

impl TensorMetadata for RustyNumQTensor {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.qtensor.shape()
    }

    fn rank(&self) -> usize {
        self.qtensor.rank()
    }
}

/// Get a display name for a TensorStorage variant (avoids borrow issues).
fn storage_dtype_name(storage: &TensorStorage) -> &'static str {
    match storage {
        TensorStorage::F64(_) => "F64",
        TensorStorage::F32(_) => "F32",
        TensorStorage::I64(_) => "I64",
        TensorStorage::I32(_) => "I32",
        TensorStorage::I16(_) => "I16",
        TensorStorage::I8(_) => "I8",
        TensorStorage::U64(_) => "U64",
        TensorStorage::U32(_) => "U32",
        TensorStorage::U16(_) => "U16",
        TensorStorage::U8(_) => "U8",
        TensorStorage::Bool(_) => "Bool",
    }
}

/// Unwrap Arc, cloning inner data if shared.
fn unwrap_arc<T: Clone>(arc: Arc<Vec<T>>) -> Vec<T> {
    Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone())
}

/// Compute C-order strides from a shape.
pub(crate) fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
