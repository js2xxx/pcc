// use nalgebra::Scalar;

use nalgebra::{Vector4, Scalar};

use crate::{point_cloud::PointCloud, points::Point3Infoed};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SearchType<T> {
    Knn(usize),
    Radius(T),
}

pub trait Searcher<'a, T: Scalar> {
    fn from_point_cloud<I>(point_cloud: &'a PointCloud<Point3Infoed<T, I>>) -> Self;

    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<&'a Vector4<T>>);

    fn search_exact(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<&'a Vector4<T>>) {
        self.search(pivot, ty, result)
    }
}
