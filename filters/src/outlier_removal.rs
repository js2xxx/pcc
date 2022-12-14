use std::fmt::Debug;

use nalgebra::{RealField, Scalar};
use num::ToPrimitive;
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point::Point,
    point_cloud::PointCloud,
    search::SearchType,
};
use pcc_search::searcher;

/// Calculate the mean distance between each point and its `mean_k` nearest
/// neighbors. If its mean distance is larger (or smaller if `negative`) than
/// the overall mean distance plus their standard deviation by `stddev_mul`,
/// then it'll be removed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StatOutlierRemoval<T: Scalar> {
    pub mean_k: usize,
    pub stddev_mul: T,
    pub negative: bool,
}

impl<T: Scalar> StatOutlierRemoval<T> {
    pub fn new(mean_k: usize, stddev_mul: T, negative: bool) -> Self {
        StatOutlierRemoval {
            mean_k,
            stddev_mul,
            negative,
        }
    }
}

impl<T: RealField + ToPrimitive> StatOutlierRemoval<T> {
    fn filter_data<P: Point<Data = T>>(&self, input: &PointCloud<P>) -> (Vec<T>, T) {
        searcher!(searcher in input, T::default_epsilon());

        let distance = {
            let mut result = Vec::with_capacity(self.mean_k);
            let mut dmean_of_point = |point: &P| {
                result.clear();
                searcher.search(point.coords(), SearchType::Knn(self.mean_k), &mut result);
                let sum = result
                    .iter()
                    .map(|(_, distance)| distance.clone())
                    .fold(T::zero(), |acc, distance| acc + distance);
                sum / T::from_usize(result.len()).unwrap()
            };

            if input.is_bounded() {
                input.iter().map(dmean_of_point).collect::<Vec<_>>()
            } else {
                { input.iter() }
                    .filter_map(|point| point.is_finite().then(|| dmean_of_point(point)))
                    .collect::<Vec<_>>()
            }
        };

        let (num, dsum, dsum2) = {
            distance
                .iter()
                .cloned()
                .fold((0, T::zero(), T::zero()), |(num, dsum, dsum2), dmean| {
                    (num + 1, dsum + dmean.clone(), dsum2 + dmean.clone() * dmean)
                })
        };

        let dnum = T::from_usize(num).unwrap();
        let dmean = dsum / dnum.clone();
        let dmean2 = dsum2 / dnum;
        let dvar = dmean2 - dmean.clone() * dmean.clone();
        let dstddev = dvar.sqrt();

        let threshold = dmean + dstddev * self.stddev_mul.clone();

        (distance, threshold)
    }
}

impl<T: RealField + ToPrimitive, P: Point<Data = T>> Filter<PointCloud<P>>
    for StatOutlierRemoval<T>
{
    fn filter_indices(&mut self, input: &PointCloud<P>) -> Vec<usize> {
        let (distance, threshold) = self.filter_data(input);

        let mut indices = (0..input.len()).collect::<Vec<_>>();
        indices.retain(|&index| (distance[index] <= threshold) ^ self.negative);
        indices
    }

    fn filter_all_indices(&mut self, input: &PointCloud<P>) -> (Vec<usize>, Vec<usize>) {
        let (distance, threshold) = self.filter_data(input);

        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        indices.retain(|&index| {
            let ret = (distance[index] <= threshold) ^ self.negative;
            if !ret {
                removed.push(index)
            }
            ret
        });
        (indices, removed)
    }
}

impl<T: RealField + ToPrimitive, P: Point<Data = T>> ApproxFilter<PointCloud<P>>
    for StatOutlierRemoval<T>
{
    #[inline]
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let mut new = input.clone();
        self.filter_mut(&mut new);
        new
    }

    fn filter_mut(&mut self, obj: &mut PointCloud<P>) {
        let (distance, threshold) = self.filter_data(obj);

        let storage = unsafe { obj.storage() };
        let mut index = 0;
        storage.retain(|_| {
            let ret = (distance[index] <= threshold) ^ self.negative;
            index += 1;
            ret
        });

        obj.reinterpret(1)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RadiusOutlierRemoval<T: Scalar> {
    pub radius: T,
    pub min_neighbors: usize,
    pub negative: bool,
}

impl<T: Scalar> RadiusOutlierRemoval<T> {
    pub fn new(radius: T, min_neighbors: usize, negative: bool) -> Self {
        RadiusOutlierRemoval {
            radius,
            min_neighbors,
            negative,
        }
    }
}

impl<T: RealField + ToPrimitive> RadiusOutlierRemoval<T> {
    fn filter_inner<P: Point<Data = T>, U>(
        &self,
        input: &PointCloud<P>,
        retainer: &mut Vec<U>,
        removed: Option<&mut Vec<usize>>,
    ) {
        macro_rules! retain {
            ($condition:expr) => {
                match removed {
                    Some(removed) => retainer.retain(|_| {
                        let (index, ret) = $condition();
                        if !ret {
                            removed.push(index)
                        }
                        ret
                    }),
                    None => retainer.retain(|_| $condition().1),
                }
            };
        }

        searcher!(searcher in input, T::default_epsilon());

        let mut index = 0;
        if input.is_bounded() {
            let mut result = Vec::with_capacity(self.min_neighbors);
            let mut condition = || {
                result.clear();
                searcher.search(
                    input[index].coords(),
                    SearchType::Knn(self.min_neighbors),
                    &mut result,
                );

                let enough_neighbors = result.len() >= self.min_neighbors;
                let enough_distance = result.pop().unwrap().1 <= self.radius;

                let ret = (enough_neighbors && enough_distance) ^ self.negative;
                index += 1;
                (index - 1, ret)
            };
            retain!(condition)
        } else {
            let mut result = Vec::with_capacity(self.min_neighbors);
            let mut condition = || {
                if !input[index].is_finite() {
                    return (index, false);
                }
                result.clear();
                searcher.search(
                    input[index].coords(),
                    SearchType::Radius(self.radius.clone()),
                    &mut result,
                );

                let enough_neighbors = result.len() >= self.min_neighbors;

                let ret = enough_neighbors ^ self.negative;
                index += 1;
                (index - 1, ret)
            };
            retain!(condition)
        }
    }
}

impl<T: RealField + ToPrimitive, P: Point<Data = T>> Filter<PointCloud<P>>
    for RadiusOutlierRemoval<T>
{
    fn filter_indices(&mut self, input: &PointCloud<P>) -> Vec<usize> {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        self.filter_inner(input, &mut indices, None);
        indices
    }

    fn filter_all_indices(&mut self, input: &PointCloud<P>) -> (Vec<usize>, Vec<usize>) {
        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        self.filter_inner(input, &mut indices, Some(&mut removed));
        (indices, removed)
    }
}

impl<T: RealField + ToPrimitive, P: Point<Data = T>> ApproxFilter<PointCloud<P>>
    for RadiusOutlierRemoval<T>
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let mut storage = Vec::from(&**input);
        self.filter_inner(input, &mut storage, None);
        PointCloud::from_vec(storage, 1)
    }
}
