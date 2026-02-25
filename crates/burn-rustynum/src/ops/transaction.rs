use burn_backend::ops::TransactionOps;

use crate::backend::RustyNum;

/// TransactionOps has a default implementation that iterates tensors and calls
/// the per-type into_data methods. No override needed for Phase 1.
impl TransactionOps<Self> for RustyNum {}
