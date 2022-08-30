use nalgebra::{RealField, Scalar, Vector4};
use pcc_common::{
    feature::Feature,
    point::{Normal, Point},
    point_cloud::PointCloud,
    search::{Search, SearchType},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NormalEstimation<T: Scalar, S> {
    pub viewpoint: Vector4<T>,
    pub search: S,
    pub search_type: SearchType<T>,
}

impl<T: Scalar, S> NormalEstimation<T, S> {
    pub fn new(viewpoint: Vector4<T>, search: S, search_type: SearchType<T>) -> Self {
        NormalEstimation {
            viewpoint,
            search,
            search_type,
        }
    }
}

impl<'a, T, I, S, O> Feature<PointCloud<I>, PointCloud<O>> for NormalEstimation<T, S>
where
    T: RealField,
    I: Point<Data = T>,
    S: Search<'a, I>,
    O: Normal<Data = T>,
{
    fn compute(&self, input: &PointCloud<I>) -> PointCloud<O> {
        let mut result = Vec::new();
        let storage = input
            .iter()
            .filter_map(|point| {
                self.search
                    .search(point.coords(), self.search_type.clone(), &mut result);
                let (normal, curvature) = pcc_common::normal(
                    result.iter().map(|&(index, _)| input[index].coords()),
                    &self.viewpoint,
                )?;
                Some(O::default().with_normal(normal).with_curvature(curvature))
            })
            .collect::<Vec<_>>();
        PointCloud::from_vec(storage, input.width())
    }
}
