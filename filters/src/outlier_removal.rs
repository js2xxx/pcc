use std::fmt::Debug;

use nalgebra::{RealField, Scalar};
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point_cloud::PointCloud,
    points::Point3Infoed,
};
use pcc_kdtree::{KdTree, KnnResultSet};

/// Calculate the mean distance between each point and its `mean_k` nearest
/// neighbors. If its mean distance is larger (or smaller if `negative`) than
/// the overall mean distance plus their standard deviation by `stddev_mul`,
/// then it'll be removed.
pub struct StatOutlierRemoval<T: Scalar> {
    pub mean_k: usize,
    pub stddev_mul: T,
    pub negtive: bool,
}

impl<T: Scalar> StatOutlierRemoval<T> {
    pub fn new(mean_k: usize, stddev_mul: T, negtive: bool) -> Self {
        StatOutlierRemoval {
            mean_k,
            stddev_mul,
            negtive,
        }
    }
}

impl<T: RealField> StatOutlierRemoval<T> {
    fn filter_data<I: Clone + Debug>(&self, input: &PointCloud<Point3Infoed<T, I>>) -> (Vec<T>, T) {
        let searcher = KdTree::new(input);

        let distance = {
            let mut result = KnnResultSet::new(self.mean_k);
            let mut dmean_of_point = |point: &Point3Infoed<T, I>| {
                result.clear();
                searcher.search_typed(&point.coords, &mut result);
                let sum = result
                    .iter()
                    .map(|(distance, _)| distance.clone())
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

impl<T: RealField, I: Clone + Debug> Filter<PointCloud<Point3Infoed<T, I>>>
    for StatOutlierRemoval<T>
{
    fn filter_indices(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> Vec<usize> {
        let (distance, threshold) = self.filter_data(input);

        let mut indices = (0..input.len()).collect::<Vec<_>>();
        indices.retain(|&index| (distance[index] <= threshold) ^ self.negtive);
        indices
    }

    fn filter_all_indices(
        &mut self,
        input: &PointCloud<Point3Infoed<T, I>>,
    ) -> (Vec<usize>, Vec<usize>) {
        let (distance, threshold) = self.filter_data(input);

        let mut indices = (0..input.len()).collect::<Vec<_>>();
        let mut removed = Vec::with_capacity(indices.len());
        indices.retain(|&index| {
            let ret = (distance[index] <= threshold) ^ self.negtive;
            if !ret {
                removed.push(index)
            }
            ret
        });
        (indices, removed)
    }
}

impl<T: RealField, I: Clone + Debug> ApproxFilter<PointCloud<Point3Infoed<T, I>>>
    for StatOutlierRemoval<T>
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        let (distance, threshold) = self.filter_data(input);

        let mut output = Vec::from(&**input);
        let mut index = 0;
        output.retain(|_| {
            let ret = (distance[index] <= threshold) ^ self.negtive;
            index += 1;
            ret
        });

        PointCloud::from_vec(output, 1)
    }
}
