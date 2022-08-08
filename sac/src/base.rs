mod arrsac;

pub use arrsac::Arrsac;
use nalgebra::{Scalar, Vector4};
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed};
use sample_consensus::{Consensus, Estimator};

pub struct PcSac<'a, P, C> {
    point_cloud: &'a PointCloud<P>,
    inner: C,
}

impl<'a, P, C> PcSac<'a, P, C> {
    pub fn new(point_cloud: &'a PointCloud<P>, inner: C) -> Self {
        PcSac { point_cloud, inner }
    }
}

impl<'a, T: Scalar, I, C> PcSac<'a, Point3Infoed<T, I>, C> {
    pub fn compute<E: Estimator<Vector4<T>>>(
        &mut self,
        estimator: &E,
    ) -> Option<(E::Model, C::Inliers)>
    where
        C: Consensus<E, Vector4<T>>,
    {
        self.inner.model_inliers(
            estimator,
            self.point_cloud.iter().map(|point| point.coords.clone()),
        )
    }
}
