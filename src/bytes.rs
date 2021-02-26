//! Trait for converting float to bytes
//! there are some modules that do this, but their goal is serialization, so it made more sense to
//! just pseudo-extend num_traits
use core::borrow::{Borrow, BorrowMut};
use core::cmp::{Eq, Ord, PartialEq, PartialOrd};
use core::fmt::Debug;
use core::hash::Hash;

// TODO num_traits might include them at some point:
// https://github.com/rust-num/num-traits/pull/103
pub trait ToBytes {
    type Bytes: Debug
        + AsRef<[u8]>
        + AsMut<[u8]>
        + PartialEq
        + Eq
        + PartialOrd
        + Ord
        + Hash
        + Borrow<[u8]>
        + BorrowMut<[u8]>
        + Default;

    fn to_bytes(self: &Self) -> Self::Bytes;
}

impl ToBytes for f32 {
    type Bytes = [u8; 4];

    fn to_bytes(self: &f32) -> Self::Bytes {
        self.to_ne_bytes()
    }
}

impl ToBytes for f64 {
    type Bytes = [u8; 8];

    fn to_bytes(self: &f64) -> Self::Bytes {
        self.to_ne_bytes()
    }
}
