//! Module with incremental statistics functions
//!
//! This contains helper functions for computing statistics on iterators, as well as structs that
//! support incremental addition of data.
extern crate order_stat;

use std::cmp::{Eq, Ordering};
use std::collections::HashMap;
use std::f64;
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::mem;


/// Summary statistics struct
///
/// This struct aggregates data to compute summary statistics using constant space overhead. It
/// implements the FromIterator trait so it can be collected from an iterator of floats.
///
/// # Examples
///
/// ```
/// let nums = [2.0, 4.0, 8.0];
/// let mut stats = inc_stats::SummStats::new();
/// for num in nums.iter() {
///     stats.add(num.clone());
/// }
/// assert_eq!(3, stats.count());
/// ```
///
/// ```
/// let nums = [2.0, 4.0, 8.0];
/// let stats: inc_stats::SummStats = nums.iter().cloned().collect();
/// assert_eq!(3, stats.count());
/// ```
#[derive(Debug)]
pub struct SummStats {
    non_nan: bool,
    count: i64,
    mean: f64,
    ssd: f64,
    min: f64,
    max: f64,
}

impl SummStats {
    /// Create a new SummStats struct with no data
    pub fn new() -> Self {
        SummStats{non_nan: false, count: 0, mean: 0.0, ssd: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY}
    }

    /// Add a number
    pub fn add(&mut self, val: f64) {
        self.non_nan |= !val.is_nan();
        self.count += 1;
        let delta = val - self.mean;
        self.mean += delta / self.count as f64;
        self.ssd += (val - self.mean) * delta;
        if val < self.min {
            self.min = val;
        }
        if self.max < val {
            self.max = val;
        }
    }

    /// Get the number of values added
    pub fn count(&self) -> i64 {
        self.count
    }

    /// Get the minimum non nan value
    ///
    /// Constant time. If no non nan values have been added, this is None.
    ///
    /// # Examples
    ///
    /// ```
    /// let nums = [2.0, 4.0, std::f64::NAN];
    /// let stats: inc_stats::SummStats = nums.iter().cloned().collect();
    /// assert!((2.0 - stats.min().unwrap()).abs() < 1.0e-6);
    /// ```
    ///
    /// ```
    /// let mut stats = inc_stats::SummStats::new();
    /// stats.add(std::f64::NAN);
    /// assert!(stats.min().is_none());
    /// ```
    pub fn min(&self) -> Option<f64> {
        match self.non_nan {
            false => None,
            true => Some(self.min),
        }
    }

    /// Get the maximum non nan value
    ///
    /// Constant time. If no non nan values have been added, this is None.
    ///
    /// # Examples
    ///
    /// ```
    /// let nums = [2.0, 4.0, std::f64::NAN];
    /// let stats: inc_stats::SummStats = nums.iter().cloned().collect();
    /// assert!((4.0 - stats.max().unwrap()).abs() < 1.0e-6);
    /// ```
    pub fn max(&self) -> Option<f64> {
        match self.non_nan {
            false => None,
            true => Some(self.max),
        }
    }

    /// Get the mean
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let nums = [2.0, 4.0];
    /// let stats: inc_stats::SummStats = nums.iter().cloned().collect();
    /// assert!((3.0 - stats.mean().unwrap()).abs() < 1.0e-6);
    /// ```
    ///
    /// ```
    /// let stats = inc_stats::SummStats::new();
    /// assert!(stats.mean().is_none());
    /// ```
    pub fn mean(&self) -> Option<f64> {
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
    /// let nums = [2.0, 4.0];
    /// let stats: inc_stats::SummStats = nums.iter().cloned().collect();
    /// assert!((6.0 - stats.sum()).abs() < 1.0e-6);
    /// ```
    pub fn sum(&self) -> f64 {
        self.count as f64 * self.mean
    }

    /// Get the sample standard deviation
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let nums = [2.0, 4.0];
    /// let stats: inc_stats::SummStats = nums.iter().cloned().collect();
    /// assert!((1.4142136 - stats.standard_deviation().unwrap()).abs() < 1.0e-6);
    /// ```
    pub fn standard_deviation(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Get the sample variance
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let nums = [2.0, 4.0];
    /// let stats: inc_stats::SummStats = nums.iter().cloned().collect();
    /// assert!((2.0 - stats.variance().unwrap()).abs() < 1.0e-6);
    /// ```
    ///
    /// ```
    /// let mut stats = inc_stats::SummStats::new();
    /// stats.add(0.0);
    /// assert!(stats.variance().is_none());
    /// ```
    pub fn variance(&self) -> Option<f64> {
        match self.count {
            0|1 => None,
            _ => Some(self.ssd / (self.count - 1) as f64),
        }
    }

    /// Get the standard error
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let nums = [2.0, 4.0];
    /// let stats: inc_stats::SummStats = nums.iter().cloned().collect();
    /// assert!((1.0 - stats.standard_error().unwrap()).abs() < 1.0e-6);
    /// ```
    pub fn standard_error(&self) -> Option<f64> {
        self.standard_deviation().map(|d| d / (self.count as f64).sqrt())
    }
}


impl FromIterator<f64> for SummStats {
    fn from_iter<I>(iter: I) -> Self where I: IntoIterator<Item=f64> {
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
/// let nums = [2.0, 4.0];
/// let mean = inc_stats::mean(nums.iter().cloned()).unwrap();
/// assert!((3.0 - mean).abs() < 1.0e-6);
/// ```
pub fn mean<I>(data: I) -> Option<f64> where I: Iterator<Item=f64> {
    data.collect::<SummStats>().mean()
}

/// Data percentile struct
///
/// This struct stores data to allow efficient computation of percentiles. This struct takes linear
/// space. It implements FromIterator to allow collection.
///
/// # Examples
///
/// ```
/// let nums = [2.0, 4.0, 8.0];
/// let mut percs = inc_stats::Percentiles::new();
/// for num in nums.iter() {
///     percs.add(num.clone());
/// }
/// assert_eq!(3, percs.count());
/// ```
///
/// ```
/// let nums = [2.0, 4.0, 8.0];
/// let percs: inc_stats::Percentiles = nums.iter().cloned().collect();
/// assert_eq!(3, percs.count());
/// ```
#[derive(Debug)]
pub struct Percentiles {
    data: Vec<f64>,
    // TODO This should be a BTreeSet, but that doesn't support finding insertion index
    in_order: Vec<usize>,
}

impl Percentiles {
    /// Create a new Percentiles object with no data
    pub fn new() -> Self {
        Percentiles{data: Vec::new(), in_order: Vec::new()}
    }

    /// Add a data point
    pub fn add(&mut self, val: f64) {
        self.data.push(val);
        self.in_order.clear();
    }

    /// Get the number of data points
    pub fn count(&self) -> i64 {
        self.data.len() as i64
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
    /// let nums = [1.0, 3.0, 7.0];
    /// let mut percs: inc_stats::Percentiles = nums.iter().cloned().collect();
    /// let quarts = percs.percentiles([0.75, 0.25, 0.5].iter().cloned()).unwrap();
    /// assert!((5.0 - quarts[0]).abs() < 1.0e-6);
    /// assert!((2.0 - quarts[1]).abs() < 1.0e-6);
    /// assert!((3.0 - quarts[2]).abs() < 1.0e-6);
    /// ```
    pub fn percentiles<I>(&mut self, percentiles: I) -> Option<Vec<f64>> where I: Iterator<Item=f64> {
        match self.data.len() {
            0 => None,
            _ => {
                let mut indexed: Vec<(usize, f64)> = percentiles.enumerate().collect();
                indexed.sort_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap());
                let mut result = Vec::with_capacity(indexed.len());
                unsafe {  // We will allocate everything in the following procedure
                    result.set_len(indexed.len());
                }
                self.percentile_recurse(&mut result, &indexed);
                Some(result)
            },
        }
    }

    fn percentile_recurse(&mut self, result: &mut [f64], percs: &[(usize, f64)]) {
        if !percs.is_empty() {
            let index = percs.len() / 2;
            let (i, perc) = percs[index];
            result[i] = self.percentile(&perc).unwrap();
            self.percentile_recurse(result, &percs[..index]);
            self.percentile_recurse(result, &percs[index+1..]);
        }
    }

    fn order_index(&mut self, index: usize) {
        match self.in_order.binary_search(&index) {
            Err(insert) => {
                let start = if insert == 0 {
                    0
                } else {
                    self.in_order[insert - 1] + 1
                };
                let end = if insert == self.in_order.len() {
                    self.data.len()
                } else {
                    self.in_order[insert]
                };
                self.in_order.insert(insert, index);
                // TODO Short circuit to min/max if index = start + 1 or end - 1
                order_stat::kth_by(&mut self.data[start..end], index - start, |a, b| a.partial_cmp(b).unwrap());
            },
            _ => (),
        }
    }

    /// Get a percentile
    ///
    /// Linear time.
    ///
    /// # Examples:
    ///
    /// ```
    /// let nums = [1.0, 5.0];
    /// let mut percs: inc_stats::Percentiles = nums.iter().cloned().collect();
    /// let quart = percs.percentile(&0.25).unwrap();
    /// assert!((2.0 - quart).abs() < 1.0e-6);
    /// ```
    pub fn percentile(&mut self, percentile: &f64) -> Option<f64> {
        assert!(&0.0 <= percentile && percentile <= &1.0, "all percentiles must be between 0 and 1");
        match self.data.len() {
            0 => None,
            _ => {
                let p_index = (self.data.len() - 1) as f64 * percentile;
                let low_index = p_index.floor() as usize;
                let high_index = p_index.ceil() as usize;
                self.order_index(low_index);
                let low = self.data[low_index];
                self.order_index(high_index);
                let high = self.data[high_index];
                let weight = p_index - low_index as f64;
                Some(low * (1.0 - weight) + high * weight)
            },
        }
    }

    /// Get the median
    ///
    /// Linear time.
    ///
    /// # Examples:
    ///
    /// ```
    /// let nums = [1.0, 5.0, 100.0];
    /// let mut percs: inc_stats::Percentiles = nums.iter().cloned().collect();
    /// let med = percs.median().unwrap();
    /// assert!((5.0 - med).abs() < 1.0e-6);
    /// ```
    pub fn median(&mut self) -> Option<f64> {
        self.percentile(&0.5)
    }
}

impl FromIterator<f64> for Percentiles {
    fn from_iter<I>(iter: I) -> Self where I: IntoIterator<Item=f64> {
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
/// let data = [3.0, 1.0, 2.0];
/// let med = inc_stats::median(data.iter().cloned()).unwrap();
/// assert!((2.0 - med).abs() < 1.0e-6);
/// ```
///
/// ```
/// let med = inc_stats::median(std::iter::empty());
/// assert!(med.is_none());
/// ```
pub fn median<I>(data: I) -> Option<f64> where I: Iterator<Item=f64> {
    data.collect::<Percentiles>().median()
}

#[derive(Debug)]
struct Hashf64(f64);

impl PartialEq for Hashf64 {
    fn eq(&self, other: &Self) -> bool {
        match self.0.partial_cmp(&other.0) {
            Some(res) => res == Ordering::Equal,
            None => unreachable!(),
        }
    }
}

impl Eq for Hashf64 {}

impl Hash for Hashf64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let int: u64 = unsafe { mem::transmute(self) };
        int.hash(state);
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
/// let nums = [2.0, 4.0, 8.0];
/// let mut mode = inc_stats::Mode::new();
/// for num in nums.iter() {
///     mode.add(num.clone());
/// }
/// assert_eq!(3, mode.count());
/// ```
///
/// ```
/// let nums = [2.0, 4.0, 8.0];
/// let mode: inc_stats::Mode = nums.iter().cloned().collect();
/// assert_eq!(3, mode.count());
/// ```
#[derive(Debug)]
pub struct Mode {
    counts: HashMap<Hashf64, i64>,
    count: i64,
    mode: Vec<f64>,
    mode_count: i64,
}

impl Mode {
    /// Create a new Mode object with no data
    pub fn new() -> Self {
        Mode{counts: HashMap::new(), count: 0, mode: Vec::new(), mode_count: 0}
    }

    /// Add a data point
    pub fn add(&mut self, val: f64) {
        self.count += 1;
        if !val.is_nan() {
            let val_count = self.counts.entry(Hashf64(val)).or_insert(0);
            *val_count += 1;
            if *val_count > self.mode_count {
                self.mode.clear();
                self.mode.push(val);
                self.mode_count += 1;
            } else if val_count == &self.mode_count {
                self.mode.push(val);
            }
        }
    }

    /// Get the number of data points
    pub fn count(&self) -> i64 {
        self.count
    }

    /// Return an iterator of all of the modes
    ///
    /// This iterator has read only reference to the mode data structure that must be dropped to
    /// continue modifying the mode. Constant time.
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
    pub fn modes(&self) -> std::iter::Cloned<std::slice::Iter<f64>> {
        self.mode.iter().cloned()
    }

    /// Return the current mode
    ///
    /// If multiple modes exist, this returns the first element that reached the largest count.
    /// NaNs are ignored when computing the mode. Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let nums = [2.0, 4.0, std::f64::NAN, 4.0];
    /// let mode: inc_stats::Mode = nums.iter().cloned().collect();
    /// assert!((4.0 - mode.mode().unwrap()).abs() < 1.0e-6);
    /// ```
    ///
    /// ```
    /// let mode = inc_stats::Mode::new();
    /// assert!(mode.mode().is_none());
    /// ```
    pub fn mode(&self) -> Option<f64> {
        self.modes().next()
    }

    /// Return the number of times the mode occurred
    ///
    /// Constant time.
    ///
    /// # Examples
    ///
    /// ```
    /// let nums = [2.0, 4.0, std::f64::NAN, 4.0];
    /// let mode: inc_stats::Mode = nums.iter().cloned().collect();
    /// assert_eq!(2, mode.mode_count());
    /// ```
    pub fn mode_count(&self) -> i64 {
        self.mode_count
    }
}

impl FromIterator<f64> for Mode {
    fn from_iter<I>(iter: I) -> Self where I: IntoIterator<Item=f64> {
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
/// let nums = [2.0, 4.0, 2.0];
/// let mode = inc_stats::mode(nums.iter().cloned());
/// assert_eq!(Some(2.0), mode);
/// ```
///
/// ```
/// let mode = inc_stats::mode(std::iter::empty());
/// assert!(mode.is_none());
/// ```
pub fn mode<I>(data: I) -> Option<f64> where I: Iterator<Item=f64> {
    data.collect::<Mode>().mode()
}
