mod arrsac;

pub use arrsac::Arrsac;
use nalgebra::{ComplexField, Scalar, Vector4};
use num::FromPrimitive;
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed};
use rand::rngs::ThreadRng;
use sample_consensus::{Consensus, Estimator};

pub struct Sac<'a, P, T: Scalar> {
    point_cloud: &'a PointCloud<P>,
    inner: Arrsac<ThreadRng, T>,
}

impl<'a, P, T: Scalar + FromPrimitive> Sac<'a, P, T> {
    pub fn new(point_cloud: &'a PointCloud<P>, threshold: T) -> Self {
        Sac {
            point_cloud,
            inner: Arrsac::new(threshold, rand::thread_rng()),
        }
    }
}

impl<'a, T: Scalar + num::Float + ComplexField<RealField = T> + PartialOrd, I>
    Sac<'a, Point3Infoed<T, I>, T>
{
    pub fn compute<E: Estimator<Vector4<T>>>(
        &mut self,
        estimator: &E,
    ) -> Option<(E::Model, Vec<usize>)> {
        self.inner
            .model_inliers(estimator, self.point_cloud.iter().map(|point| point.coords))
    }
}
