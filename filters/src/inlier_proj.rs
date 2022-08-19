use std::{fmt::Debug, marker::PhantomData};

use nalgebra::{ComplexField, Scalar, Vector4};
use pcc_common::{filter::ApproxFilter, point::Point, point_cloud::PointCloud};
use pcc_sac::SacModel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlierProjection<T: Scalar, M: SacModel<Vector4<T>>> {
    pub model: M,
    pub inliers: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T: Scalar, M: SacModel<Vector4<T>>> InlierProjection<T, M> {
    pub fn new(model: M, inliers: Vec<usize>) -> Self {
        InlierProjection {
            model,
            inliers,
            _marker: PhantomData,
        }
    }
}

impl<T: ComplexField, M: SacModel<Vector4<T>>, P: Point<Data = T>> ApproxFilter<PointCloud<P>>
    for InlierProjection<T, M>
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let storage = { self.inliers.iter() }
            .map(|&index| {
                { input[index].clone() }.with_coords(self.model.project(input[index].coords()))
            })
            .collect::<Vec<_>>();
        PointCloud::from_vec(storage, 1)
    }
}
