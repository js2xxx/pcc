use nalgebra::{Scalar, Vector4};

use crate::{point_cloud::PointCloud, points::Point3Infoed};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SearchType<T> {
    Knn(usize),
    Radius(T),
}

pub trait Searcher<'a, T: Scalar, I> {
    fn point_cloud(&self) -> &'a PointCloud<Point3Infoed<T, I>>;

    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<(usize, T)>);

    fn search_exact(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<(usize, T)>) {
        self.search(pivot, ty, result)
    }
}
