// use nalgebra::Scalar;

use nalgebra::{Scalar, Vector4};

use crate::{point_cloud::PointCloud, points::Point3Infoed};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SearchType<T> {
    Knn(usize),
    Radius(T),
}

pub trait Searcher<'a, T: Scalar> {
    type FromExtra = ();

    fn from_point_cloud<I>(
        point_cloud: &'a PointCloud<Point3Infoed<T, I>>,
        extra: Self::FromExtra,
    ) -> Self;

    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<usize>);

    fn search_exact(
        &self,
        pivot: &Vector4<T>,
        ty: SearchType<T>,
        result: &mut Vec<usize>,
    ) {
        self.search(pivot, ty, result)
    }
}
