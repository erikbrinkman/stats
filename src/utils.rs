//! Various utilities
use num_traits::{Float, FromPrimitive};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Any error from this library
#[derive(Debug)]
pub struct StatsError(String);

impl StatsError {
    pub fn new(msg: String) -> StatsError {
        StatsError(msg)
    }
}

impl<'a> From<&'a str> for StatsError {
    fn from(msg: &'a str) -> StatsError {
        StatsError(String::from(msg))
    }
}

impl Display for StatsError {
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        write!(formatter, "statistical error: {}", self.0)
    }
}

impl Error for StatsError {}

/// An iterator that does BFS assuming slice is a full binary tree
pub struct InsideOut<'a, T> {
    elems: &'a [T],
    current: usize,
    step: usize,
}

impl<'a, T> Iterator for InsideOut<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.current += self.step;
        match self.elems.get(self.current) {
            Some(elem) => Some(elem),
            _ if self.step <= 2 => None,
            _ => {
                self.step /= 2;
                self.current = (self.step / 2) - 1;
                Some(&self.elems[self.current])
            }
        }
    }
}

/// Compute the step involved
fn checked_step(len: usize) -> Option<usize> {
    len.checked_add(1)?
        .checked_next_power_of_two()?
        .checked_mul(2)
}

/// Create an iterator that does BFS treating the slice as a maximal tree
///
/// Errors if the input is too long
pub fn inside_out<'a, T>(elems: &'a [T]) -> Result<InsideOut<'a, T>, StatsError> {
    let step = checked_step(elems.len())
        .ok_or(StatsError::from("elems to long to efficiently alternate"))?;
    Ok(InsideOut {
        elems,
        current: 0,
        step,
    })
}

/// weighted average between two values, weight given to high
pub fn weighted_average<T: Float + FromPrimitive>(low: T, high: T, weight: f64) -> Option<T> {
    Some(low * T::from_f64(1.0 - weight)? + high * T::from_f64(weight)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_out_tests() {
        let best_case: Vec<_> = (0..7).collect();
        let best_actual: Vec<_> = inside_out(&best_case).unwrap().copied().collect();
        assert_eq!(best_actual, vec!(3, 1, 5, 0, 2, 4, 6));

        let worst_case: Vec<_> = (0..8).collect();
        let worst_actual: Vec<_> = inside_out(&worst_case).unwrap().copied().collect();
        assert_eq!(worst_actual, vec!(7, 3, 1, 5, 0, 2, 4, 6));

        let middle_case: Vec<_> = (0..5).collect();
        let middle_actual: Vec<_> = inside_out(&middle_case).unwrap().copied().collect();
        assert_eq!(middle_actual, vec!(3, 1, 0, 2, 4));
    }
}
