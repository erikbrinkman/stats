//! Module with incremental statistics functions
//!
//! This contains helper functions for computing statistics on iterators, as well as structs that
//! support incremental addition of data.
mod bytes;
mod copy;
mod utils;

use bytes::ToBytes;
pub use copy::DerefCopy;
use num_traits::{Float, FromPrimitive};
use std::cell::RefCell;
use std::cmp::{self, Eq};
use std::collections::{BTreeSet, HashMap};
use std::f64;
use std::hash::{Hash, Hasher};
use std::iter::{self, FromIterator};
use std::ops::AddAssign;
pub use utils::StatsError;

/// Summary statistics struct
///
/// This struct aggregates data to compute summary statistics using constant space overhead. It
/// implements the FromIterator trait so it can be collected from an iterator of floats.
///
/// # Examples
///
/// ```
/// let mut stats = inc_stats::SummStats::new();
/// for &num in &[2.0, 4.0, 8.0] {
///     stats.add(num);
/// }
/// assert_eq!(3, stats.count());
/// ```
///
/// ```
/// let stats: inc_stats::SummStats<f64> = [2.0, 4.0, 8.0].iter().collect();
/// assert_eq!(3, stats.count());
/// ```
#[derive(Debug)]
pub struct SummStats<T: Float + FromPrimitive + AddAssign> {
    non_nan: bool,
    count: u64,
    mean: T,
    ssd: T,
    min: T,
    max: T,
}

impl<T: Float + FromPrimitive + AddAssign> SummStats<T> {
    /// Create a new SummStats struct with no data
    pub fn new() -> Self {
        SummStats {
            non_nan: false, // any value is not nan
            count: 0,
            mean: T::zero(),
            ssd: T::zero(),
            min: T::infinity(),
            max: T::neg_infinity(),
        }
    }

    /// Add a number
    ///
    /// # Examples
    ///
    /// ```
    /// let mut stats = inc_stats::SummStats::new();
    /// stats.add(0.0);
    /// stats.add(&1.2);
    /// assert_eq!(2, stats.count());
    /// ```
    ///
    /// # Panics
    ///
    /// when the internal count can't be converted into the float data type.
    pub fn add(&mut self, bval: impl DerefCopy<Output = T>) {
        self.checked_add(bval).unwrap();
    }

    /// Add a number
    ///
    /// Check for conversion errors, will only happen when the internal count can't be converted
    /// into the float data type.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut stats = inc_stats::SummStats::new();
    /// stats.checked_add(0.0).unwrap();
    /// assert_eq!(1, stats.count());
    /// ```
    pub fn checked_add(&mut self, rval: impl DerefCopy<Output = T>) -> Result<(), StatsError> {
        // NOTE need to exit early before mutating state
        let count = T::from_u64(self.count + 1).ok_or("can't convert from count to float type")?;
        let val = rval.deref_copy();
        self.non_nan |= !val.is_nan();
        self.count += 1;
        let delta = val - self.mean;
        self.mean += delta / count;
        self.ssd += (val - self.mean) * delta;
        if val < self.min {
            self.min = val;
        }
        if self.max < val {
            self.max = val;
        }
        Ok(())
    }

    /// Get the number of values added
    pub fn count(&self) -> u64 {
        self.count
    }

    fn tcount(&self) -> T {
        // if we could add the last value, then we must have been able to convert this
        T::from_u64(self.count).unwrap()
    }

    /// Get the minimum non nan value
    ///
    /// Constant time. If no non nan values have been added, this is None.
    ///
    /// # Examples
    ///
    /// ```
    /// let stats: inc_stats::SummStats<_> = [2.0, 4.0, std::f64::NAN].iter().collect();
    /// assert_eq!(2.0, stats.min().unwrap());
    /// ```
    ///
    /// ```
    /// let mut stats = inc_stats::SummStats::new();
    /// stats.add(std::f64::NAN);
    /// assert!(stats.min().is_none());
    /// ```
    pub fn min(&self) -> Option<T> {
        if self.non_nan {
            Some(self.min)
        } else {
            None
        }
    }

    /// Get the maximum non nan value
    ///
    /// Constant time. If no non nan values have been added, this is None.
    ///
    /// # Examples
    ///
    /// ```
    /// let stats: inc_stats::SummStats<_> = [2.0, 4.0, std::f64::NAN].iter().collect();
    /// assert_eq!(4.0, stats.max().unwrap());
    /// ```
    pub fn max(&self) -> Option<T> {
        if self.non_nan {
            Some(self.max)
        } else {
            None
        }
    }

    /// Get the mean
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let stats: inc_stats::SummStats<f64> = [2.0, 4.0].iter().collect();
    /// assert!((3.0 - stats.mean().unwrap()).abs() < 1.0e-6);
    /// ```
    ///
    /// ```
    /// let stats = inc_stats::SummStats::<f64>::new();
    /// assert!(stats.mean().is_none());
    /// ```
    pub fn mean(&self) -> Option<T> {
        match self.count {
            0 => None,
            _ => Some(self.mean),
        }
    }

    /// Get the sum
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let stats: inc_stats::SummStats<f64> = [2.0, 4.0].iter().collect();
    /// assert!((6.0 - stats.sum()).abs() < 1.0e-6);
    /// ```
    pub fn sum(&self) -> T {
        self.tcount() * self.mean
    }

    /// Get the sample standard deviation
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let stats: inc_stats::SummStats<f64> = [2.0, 4.0].iter().collect();
    /// assert!((1.4142136 - stats.standard_deviation().unwrap()).abs() < 1.0e-6);
    /// ```
    pub fn standard_deviation(&self) -> Option<T> {
        self.variance().map(T::sqrt)
    }

    /// Get the sample variance
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let stats: inc_stats::SummStats<f64> = [2.0, 4.0].iter().collect();
    /// assert!((2.0 - stats.variance().unwrap()).abs() < 1.0e-6);
    /// ```
    ///
    /// ```
    /// let mut stats = inc_stats::SummStats::new();
    /// stats.add(0.0);
    /// assert!(stats.variance().is_none());
    /// ```
    pub fn variance(&self) -> Option<T> {
        match self.count {
            0 | 1 => None,
            // if we could add to this, it must be possible
            _ => Some(self.ssd / T::from_u64(self.count - 1).unwrap()),
        }
    }

    /// Get the standard error
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let stats: inc_stats::SummStats<f64> = [2.0, 4.0].iter().collect();
    /// assert!((1.0 - stats.standard_error().unwrap()).abs() < 1.0e-6);
    /// ```
    pub fn standard_error(&self) -> Option<T> {
        self.standard_deviation().map(|d| d / self.tcount().sqrt())
    }
}

impl<T: Float + FromPrimitive + AddAssign> Default for SummStats<T> {
    fn default() -> Self {
        SummStats::new()
    }
}

impl<T: Float + FromPrimitive + AddAssign, V: DerefCopy<Output = T>> FromIterator<V>
    for SummStats<T>
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = V>,
    {
        let mut stats = SummStats::new();
        for val in iter {
            stats.add(val);
        }
        stats
    }
}

/// Get the mean of a set of data
///
/// This method takes constant space and linear time.
///
/// # Examples:
///
/// ```
/// let mean: f64 = inc_stats::mean(&[2.0, 4.0]).unwrap();
/// assert!((3.0 - mean).abs() < 1.0e-6);
/// ```
pub fn mean<T, V, I>(data: I) -> Option<T>
where
    T: Float + FromPrimitive + AddAssign,
    V: DerefCopy<Output = T>,
    I: IntoIterator<Item = V>,
{
    data.into_iter().collect::<SummStats<_>>().mean()
}

/// The mutable data structure that caches ordered percentiles
#[derive(Debug)]
struct CachedOrdering<T: Float + FromPrimitive> {
    data: Vec<T>,
    in_order: BTreeSet<usize>,
}

impl<T: Float + FromPrimitive> CachedOrdering<T> {
    /// Create a new Percentiles object with no data
    fn new() -> Self {
        CachedOrdering {
            // all of the points aded so far
            data: Vec::new(),
            // indices in data that are known to be in sorted order
            in_order: BTreeSet::new(),
        }
    }

    /// Add a data point
    fn add(&mut self, val: T) {
        self.data.push(val);
        self.in_order.clear();
    }

    /// assert index is in sorted order, and get value at that order
    fn order_index(&mut self, index: usize) -> T {
        if self.in_order.insert(index) {
            let start = match self.in_order.range(..index).next_back() {
                Some(ind) => ind + 1,
                None => 0,
            };
            let end = match self.in_order.range(index + 1..).next() {
                Some(&ind) => ind,
                None => self.data.len(),
            };
            self.data[start..end].select_nth_unstable_by(index - start, |a, b| {
                // we filter out nans
                a.partial_cmp(b).unwrap()
            });
        }
        self.data[index]
    }

    /// Get the amount of data
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Data percentile struct
///
/// This struct stores data to allow efficient computation of percentiles. This struct takes linear
/// space. It implements FromIterator to allow collection. This collection ignores NaNs.
///
/// The structure is designed for efficient computation of percentiles when data is added and then
/// percentiles are computed. Adding data is constant time, querying percentiles is linear time,
/// with some caching to make it faster for computing several percentiles. If you were going to
/// query percentiles while adding data, then you probably want to use a different data structure.
///
/// # Examples
///
/// ```
/// let mut percs = inc_stats::Percentiles::new();
/// for &num in &[2.0, 4.0, 8.0] {
///     percs.add(num);
/// }
/// assert_eq!(3, percs.count());
/// ```
///
/// ```
/// let percs: inc_stats::Percentiles<f64> = [2.0, 4.0, 8.0].iter().collect();
/// assert_eq!(3, percs.count());
/// ```
#[derive(Debug)]
pub struct Percentiles<T: Float + FromPrimitive> {
    data: RefCell<CachedOrdering<T>>,
    nan_count: usize,
}

impl<T: Float + FromPrimitive> Percentiles<T> {
    /// Create a new Percentiles object with no data
    pub fn new() -> Self {
        Percentiles {
            data: RefCell::new(CachedOrdering::new()),
            nan_count: 0,
        }
    }

    /// Add a data point
    pub fn add(&mut self, rval: impl DerefCopy<Output = T>) {
        let val = rval.deref_copy();
        if val.is_nan() {
            self.nan_count += 1;
        } else {
            self.data.borrow_mut().add(val);
        }
    }

    /// Get the number of data points
    pub fn count(&self) -> usize {
        self.data.borrow().len() + self.nan_count
    }

    /// Get a number of percentiles
    ///
    /// This takes linear time in the number of added data points, and log linear in the number of
    /// percentiles. This will be marginally more efficient than calling percentile repeatedly in a
    /// bad order.
    ///
    /// # Examples:
    ///
    /// ```
    /// let percs: inc_stats::Percentiles<f64> = [1.0, 3.0, 7.0].iter().collect();
    /// let quarts = percs.percentiles(&[0.75, 0.25, 0.5]).unwrap().unwrap();
    /// assert!((5.0 - quarts[0]).abs() < 1.0e-6);
    /// assert!((2.0 - quarts[1]).abs() < 1.0e-6);
    /// assert!((3.0 - quarts[2]).abs() < 1.0e-6);
    /// ```
    // NOTE inside out does not guarantee worst case linear complexity. Asking for percentiles that
    // correspond to the 1st, 2nd, 3rd, index etc will still have `log p * n` complexity (versus `p
    // * n` for the native way). If we instead picked the percentiles closest to the midpoint of
    // the remaining space, the complexity would drop to `log p + n`, which is just n.
    pub fn percentiles<P, I>(&self, percentiles: I) -> Result<Option<Vec<T>>, StatsError>
    where
        P: DerefCopy<Output = f64>,
        I: IntoIterator<Item = P>,
    {
        let len = self.data.borrow().len();
        match len {
            0 => Ok(None),
            _ => {
                // need to output result in same order, but need this sorted for efficiency
                let mut indexed: Vec<(usize, f64)> = percentiles
                    .into_iter()
                    .map(DerefCopy::deref_copy)
                    .enumerate()
                    .collect();
                if indexed.iter().any(|(_, e)| e.is_nan()) {
                    Err(StatsError::from("percentiles can't be nan"))?
                }
                // we checked there were no nans
                indexed.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
                // allocate result
                let mut result: Vec<Option<T>> = iter::repeat(None).take(indexed.len()).collect();
                for &(ind, perc) in utils::inside_out(&indexed)? {
                    // we checked that we had data
                    result[ind] = Some(self.percentile(perc)?.unwrap());
                }
                let checked_result: Option<Vec<_>> = result.iter().copied().collect();
                // fails if there is a logic error in inside_out
                Ok(Some(checked_result.unwrap()))
            }
        }
    }

    /// Get a percentile
    ///
    /// Linear time.
    ///
    /// # Examples:
    ///
    /// ```
    /// let percs: inc_stats::Percentiles<f64> = [1.0, 5.0].iter().collect();
    /// let quart = percs.percentile(0.25).unwrap().unwrap();
    /// assert!((2.0 - quart).abs() < 1.0e-6);
    /// ```
    pub fn percentile(
        &self,
        percentile: impl DerefCopy<Output = f64>,
    ) -> Result<Option<T>, StatsError> {
        let perc = percentile.deref_copy();
        if perc < 0.0 || 1.0 < perc {
            Err(StatsError::new(format!(
                "all percentiles must be between 0 and 1, but got: {}",
                perc
            )))
        } else {
            let mut ordering = self.data.borrow_mut();
            match ordering.len() {
                0 => Ok(None),
                _ => {
                    let p_index = (ordering.len() - 1) as f64 * perc;
                    let low_index = p_index.floor() as usize;
                    let high_index = p_index.ceil() as usize;
                    let low = ordering.order_index(low_index);
                    let high = ordering.order_index(high_index);
                    let weight = p_index - low_index as f64;
                    let perc = utils::weighted_average(low, high, weight)
                        .ok_or("can't convert from weight to float")?;
                    Ok(Some(perc))
                }
            }
        }
    }

    /// Get the median
    ///
    /// Linear time.
    ///
    /// # Examples:
    ///
    /// ```
    /// let percs: inc_stats::Percentiles<f64> = [1.0, 5.0, 100.0].iter().collect();
    /// let med = percs.median().unwrap();
    /// assert_eq!(5.0, med);
    /// ```
    pub fn median(&self) -> Option<T> {
        self.percentile(0.5).expect("0.5 is a valid percentile")
    }
}

impl<T: Float + FromPrimitive> Default for Percentiles<T> {
    fn default() -> Self {
        Percentiles::new()
    }
}

impl<T: Float + FromPrimitive, V: DerefCopy<Output = T>> FromIterator<V> for Percentiles<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = V>,
    {
        let mut percs = Percentiles::new();
        for val in iter {
            percs.add(val);
        }
        percs
    }
}

/// Get the median of a set of data
///
/// This takes linear time and linear space.
///
/// # Examples
///
/// ```
/// let med = inc_stats::median(&[3.0, 1.0, 2.0]).unwrap();
/// assert_eq!(2.0, med);
/// ```
///
/// ```
/// let med = inc_stats::median(std::iter::empty::<f64>());
/// assert!(med.is_none());
/// ```
pub fn median<T, V, I>(data: I) -> Option<T>
where
    T: Float + FromPrimitive,
    V: DerefCopy<Output = T>,
    I: IntoIterator<Item = V>,
{
    data.into_iter().collect::<Percentiles<T>>().median()
}

#[derive(Debug, PartialEq)]
struct HashFloat<T: Float + ToBytes>(T);

impl<T: Float + ToBytes> Eq for HashFloat<T> {}

impl<T: Float + ToBytes> Hash for HashFloat<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bytes().hash(state);
    }
}

/// Mode computation struct
///
/// This struct stores data to allow efficient computation of the mode. This struct takes linear
/// space. It implements FromIterator to allow collection.
///
/// # Examples
///
/// ```
/// let mut mode = inc_stats::Mode::new();
/// for &num in &[2.0, 4.0, 8.0] {
///     mode.add(num);
/// }
/// assert_eq!(3, mode.count());
/// ```
///
/// ```
/// let mode: inc_stats::Mode<f64> = [2.0, 4.0, 8.0].iter().collect();
/// assert_eq!(3, mode.count());
/// ```
#[derive(Debug)]
pub struct Mode<T: Float + ToBytes> {
    counts: HashMap<HashFloat<T>, usize>,
    count: usize,
    nan_count: usize,
    mode: Vec<T>,
    mode_count: usize,
}

impl<T: Float + ToBytes> Mode<T> {
    /// Create a new Mode object with no data
    pub fn new() -> Self {
        Mode {
            counts: HashMap::new(),
            count: 0,
            nan_count: 0,
            mode: Vec::new(),
            mode_count: 0,
        }
    }

    /// Add a data point
    pub fn add(&mut self, rval: impl DerefCopy<Output = T>) {
        let val = rval.deref_copy();
        self.count += 1;
        if val.is_nan() {
            self.nan_count += 1;
        } else {
            let val_count = self.counts.entry(HashFloat(val)).or_insert(0);
            *val_count += 1;
            if *val_count > self.mode_count {
                self.mode.clear();
                self.mode.push(val);
                self.mode_count += 1;
            } else if *val_count == self.mode_count {
                self.mode.push(val);
            }
        }
    }

    /// Get the number of data points
    ///
    /// # Examples
    ///
    /// ```
    /// let num: inc_stats::Mode<_> = [1.0, 2.0, std::f64::NAN].iter().collect();
    /// assert_eq!(3, num.count());
    /// ```
    pub fn count(&self) -> usize {
        self.count
    }

    /// Count the number of distinct values
    ///
    /// Distinctness for floating points is very finicy. Values that may print the same may not be
    /// same underlying value. Computations that yield the same value in "real" math may not yield
    /// the same value in floating point math.
    ///
    /// This ignores nans
    ///
    /// # Examples
    ///
    /// ```
    /// let num: inc_stats::Mode<_> = [1.0, 2.0, 2.0, std::f64::NAN].iter().collect();
    /// assert_eq!(2, num.count_distinct());
    /// ```
    pub fn count_distinct(&self) -> usize {
        self.counts.len()
    }

    /// Count the number of distinct values
    ///
    /// This treats all NaNs as different
    ///
    /// # Examples
    ///
    /// ```
    /// let num: inc_stats::Mode<_> = [1.0, std::f64::NAN, std::f64::NAN].iter().collect();
    /// assert_eq!(3, num.count_distinct_nan());
    /// ```
    ///
    /// Treat all nans the same
    /// ```
    /// let num: inc_stats::Mode<_> = [1.0, std::f64::NAN, std::f64::NAN].iter().collect();
    /// assert_eq!(2, std::cmp::min(num.count_distinct() + 1, num.count_distinct_nan()));
    /// ```
    pub fn count_distinct_nan(&self) -> usize {
        self.counts.len() + self.nan_count
    }

    /// Return an iterator of all of the modes
    ///
    /// Multiple modes are retruned in the order they became a mode. NaNs are ignored.
    ///
    /// This iterator has read only reference to the mode data structure that must be dropped to
    /// continue modifying the mode.
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mode = inc_stats::Mode::new();
    /// {
    ///     let mut it = mode.modes();
    ///     assert!(it.next().is_none());
    /// }
    ///
    /// mode.add(5.0);
    /// {
    ///     let mut it = mode.modes();
    ///     assert_eq!(Some(5.0), it.next());
    ///     assert!(it.next().is_none());
    /// }
    ///
    /// mode.add(3.0);
    /// {
    ///     let mut it = mode.modes();
    ///     assert_eq!(Some(5.0), it.next());
    ///     assert_eq!(Some(3.0), it.next());
    ///     assert!(it.next().is_none());
    /// }
    ///
    /// mode.add(3.0);
    /// {
    ///     let mut it = mode.modes();
    ///     assert_eq!(Some(3.0), it.next());
    ///     assert!(it.next().is_none());
    /// }
    /// ```
    pub fn modes(&self) -> impl Iterator<Item = T> + '_ {
        self.mode.iter().copied()
    }

    /// gets an option for if nan would be in the mode
    fn nan_mode(&self) -> Option<T> {
        if self.nan_count > 0 && self.nan_count >= self.mode_count {
            Some(T::nan())
        } else {
            None
        }
    }

    /// Return an iterator of all of the modes
    ///
    /// This iterator will include NaN if present as a mode. NaN will always be returned last
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mode: inc_stats::Mode<_> = [std::f64::NAN, 5.0].iter().collect();
    /// let mut it = mode.modes_nan();
    /// assert_eq!(Some(5.0), it.next());
    /// assert!(it.next().unwrap().is_nan());
    /// assert!(it.next().is_none());
    /// ```
    pub fn modes_nan(&self) -> impl Iterator<Item = T> + '_ {
        self.modes().chain(self.nan_mode())
    }

    /// Return the current mode
    ///
    /// If multiple modes exist, this returns the first element that reached the largest count.
    /// NaNs are ignored when computing the mode.
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mode: inc_stats::Mode<_> = [2.0, 4.0, std::f64::NAN, 4.0].iter().collect();
    /// assert_eq!(4.0, mode.mode().unwrap());
    /// ```
    ///
    /// ```
    /// let mode = inc_stats::Mode::<f64>::new();
    /// assert!(mode.mode().is_none());
    /// ```
    pub fn mode(&self) -> Option<T> {
        self.modes().next()
    }

    /// Return the current mode
    ///
    /// If multiple modes exist, this returns the first element that reached the largest count that
    /// wasn't NaN. NaN will be returned only if it is the unique mode.
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mode: inc_stats::Mode<_> = [2.0, 4.0, std::f64::NAN, std::f64::NAN].iter().collect();
    /// assert!(mode.mode_nan().unwrap().is_nan());
    /// ```
    pub fn mode_nan(&self) -> Option<T> {
        if self.nan_count > self.mode_count {
            Some(T::nan())
        } else {
            self.mode()
        }
    }

    /// Return the number of times the mode occurred
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mode: inc_stats::Mode<_> = [2.0, 4.0, std::f64::NAN, 4.0].iter().collect();
    /// assert_eq!(2, mode.mode_count());
    /// ```
    pub fn mode_count(&self) -> usize {
        self.mode_count
    }

    /// Return the number of times the mode occurred
    ///
    /// Counts NaNs as a possible mode.
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let mode: inc_stats::Mode<_> = [2.0, 4.0, std::f64::NAN, std::f64::NAN].iter().collect();
    /// assert_eq!(2, mode.mode_count_nan());
    /// ```
    pub fn mode_count_nan(&self) -> usize {
        cmp::max(self.mode_count, self.nan_count)
    }
}

impl<T: Float + ToBytes, V: DerefCopy<Output = T>> FromIterator<V> for Mode<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = V>,
    {
        let mut mode = Mode::new();
        for val in iter {
            mode.add(val);
        }
        mode
    }
}

/// Get the mode of a set of data
///
/// If multiple modes exist, this returns the first element that reached the largest count.
/// NaNs are ignored when computing the mode.
///
/// # Examples:
///
/// ```
/// let mode = inc_stats::mode(&[2.0, 4.0, 2.0]);
/// assert_eq!(Some(2.0), mode);
/// ```
///
/// ```
/// let mode: Option<f64> = inc_stats::mode(&[]);
/// assert!(mode.is_none());
/// ```
pub fn mode<T, V, I>(data: I) -> Option<T>
where
    T: Float + ToBytes,
    V: DerefCopy<Output = T>,
    I: IntoIterator<Item = V>,
{
    data.into_iter().collect::<Mode<T>>().mode()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_mean_test() {
        let avg: f32 = mean(&[0.0, 1.0, 2.0]).unwrap();
        assert!((avg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f32_median_test() {
        let avg: f32 = median(&[0.0, 1.0, 2.0, 3.0]).unwrap();
        assert!((avg - 1.5).abs() < 1e-6);
    }

    #[test]
    fn nan_percentile_test() {
        let percs: Percentiles<_> = [f64::NAN].iter().collect();
        // we know we put something in
        assert_eq!(1, percs.count());
        // but don't have enough data to get median
        assert_eq!(None, percs.median());
    }

    #[test]
    fn nan_mode_test() {
        let avg: Mode<_> = [f64::NAN].iter().collect();
        assert!(avg.mode_nan().unwrap().is_nan());
    }

    #[test]
    fn cached_ordering_test() {
        let mut ord = CachedOrdering::new();
        ord.add(0.0);
        ord.add(1.0);
        ord.add(2.0);
        // gets correct index
        assert_eq!(1.0, ord.order_index(1));
        // cached for later
        assert!(ord.in_order.contains(&1));
    }
}
