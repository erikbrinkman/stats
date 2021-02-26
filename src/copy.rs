//! Trait for copying for ownership
//!
//! This ultimately allows methods to take either references or owned values and still do type
//! inference

pub trait DerefCopy: Copy {
    type Output;
    fn deref_copy(self) -> Self::Output;
}

impl<T: Copy> DerefCopy for &T {
    type Output = T;
    fn deref_copy(self) -> T {
        *self
    }
}

impl DerefCopy for f64 {
    type Output = Self;
    fn deref_copy(self) -> Self {
        self
    }
}

impl DerefCopy for f32 {
    type Output = Self;
    fn deref_copy(self) -> Self {
        self
    }
}
