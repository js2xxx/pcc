use nalgebra::Vector4;

use crate::{point::Point, point_cloud::PointCloud};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SearchType<T> {
    Knn(usize),
    Radius(T),
}

pub trait Searcher<'a, P: Point> {
    fn point_cloud(&self) -> &'a PointCloud<P>;

    fn search(
        &self,
        pivot: &Vector4<P::Data>,
        ty: SearchType<P::Data>,
        result: &mut Vec<(usize, P::Data)>,
    );

    fn search_exact(
        &self,
        pivot: &Vector4<P::Data>,
        ty: SearchType<P::Data>,
        result: &mut Vec<(usize, P::Data)>,
    ) {
        self.search(pivot, ty, result)
    }
}
