use std::{fmt::Debug, marker::PhantomData};

use nalgebra::{ComplexField, Scalar, Vector4};
use pcc_common::{filter::ApproxFilter, point_cloud::PointCloud, points::Point3Infoed};
use pcc_sac::SacModel;

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

impl<T: ComplexField<RealField = T>, M: SacModel<Vector4<T>>, I: Clone + Debug>
    ApproxFilter<PointCloud<Point3Infoed<T, I>>> for InlierProjection<T, M>
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        let storage = { self.inliers.iter() }
            .map(|&index| Point3Infoed {
                coords: self.model.project(&input[index].coords),
                extra: input[index].extra.clone(),
            })
            .collect::<Vec<_>>();
        PointCloud::from_vec(storage, 1)
    }
}
