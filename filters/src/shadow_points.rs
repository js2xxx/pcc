use std::fmt::Debug;

use nalgebra::{RealField, Scalar};
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point_cloud::PointCloud,
    points::{Point3Infoed, PointNormal},
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
    fn filter_one<I: PointNormal<T>>(&self, point: &Point3Infoed<T, I>) -> bool {
        let normal = point.extra.normal().normal.clone();
        let value = (point.coords.x.clone() * normal.x.clone()
            + point.coords.y.clone() * normal.y.clone()
            + point.coords.z.clone() * normal.z.clone())
        .abs();

        (value >= self.threshold) ^ self.negative
    }

    fn inner<I: PointNormal<T>>(&self) -> impl FnMut(&Point3Infoed<T, I>) -> bool + '_ {
        |point| self.filter_one(point)
    }
}

impl<T: RealField, I: PointNormal<T>> Filter<PointCloud<Point3Infoed<T, I>>> for ShadowPoints<T> {
    fn filter_indices(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> Vec<usize> {
        self.inner().filter_indices(input)
    }

    fn filter_all_indices(
        &mut self,
        input: &PointCloud<Point3Infoed<T, I>>,
    ) -> (Vec<usize>, Vec<usize>) {
        self.inner().filter_all_indices(input)
    }
}

impl<T: RealField, I: PointNormal<T> + Clone + Debug> ApproxFilter<PointCloud<Point3Infoed<T, I>>>
    for ShadowPoints<T>
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        self.inner().filter(input)
    }
}
