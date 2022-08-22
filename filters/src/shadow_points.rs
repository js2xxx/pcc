use std::fmt::Debug;

use nalgebra::{RealField, Scalar};
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point::{PointNormal, Point},
    point_cloud::PointCloud,
};

pub struct ShadowPoints<T: Scalar> {
    pub threshold: T,
    pub negative: bool,
}

impl<T: Scalar> ShadowPoints<T> {
    pub fn new(threshold: T, negative: bool) -> Self {
        ShadowPoints {
            threshold,
            negative,
        }
    }
}

impl<T: RealField> ShadowPoints<T> {
    fn filter_one<P: PointNormal + Point<Data = T>>(&self, point: &P) -> bool {
        let normal = point.normal();
        let value = (point.coords().x.clone() * normal.x.clone()
            + point.coords().y.clone() * normal.y.clone()
            + point.coords().z.clone() * normal.z.clone())
        .abs();

        (value >= self.threshold) ^ self.negative
    }

    fn inner<P: PointNormal + Point<Data = T>>(&self) -> impl FnMut(&P) -> bool + '_ {
        |point| self.filter_one(point)
    }
}

impl<T: RealField, P: PointNormal + Point<Data = T>> Filter<PointCloud<P>> for ShadowPoints<T> {
    fn filter_indices(&mut self, input: &PointCloud<P>) -> Vec<usize> {
        self.inner().filter_indices(input)
    }

    fn filter_all_indices(&mut self, input: &PointCloud<P>) -> (Vec<usize>, Vec<usize>) {
        self.inner().filter_all_indices(input)
    }
}

impl<T: RealField, P: PointNormal + Point<Data = T> + Clone + Debug> ApproxFilter<PointCloud<P>>
    for ShadowPoints<T>
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        self.inner().filter(input)
    }
}
