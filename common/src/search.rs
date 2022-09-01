use nalgebra::Vector4;
use static_assertions::assert_obj_safe;

use crate::{point::Point, point_cloud::PointCloud};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SearchType<T> {
    Knn(usize),
    Radius(T),
}

pub trait Search<'a, P: Point> {
    fn input(&self) -> &'a PointCloud<P>;

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

assert_obj_safe!(Search<'_, crate::point::Point3>);
