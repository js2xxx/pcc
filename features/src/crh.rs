use std::sync::Arc;

use nalgebra::{convert, Complex, DVector, RealField, Rotation3, Scalar, Unit, Vector3};
use num::{ToPrimitive, Zero};
use pcc_common::{
    feature::Feature,
    point::{Normal, Point},
    point_cloud::PointCloud,
};
use rustfft::{Fft, FftPlanner};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Crh<T: Scalar> {
    pub centroid: Vector3<T>,
    pub viewpoint: Vector3<T>,
}

impl<T: Scalar> Crh<T> {
    #[inline]
    pub fn new(centroid: Vector3<T>, viewpoint: Vector3<T>) -> Self {
        Crh {
            centroid,
            viewpoint,
        }
    }

    #[inline]
    pub fn fft() -> Arc<dyn Fft<T>>
    where
        T: RealField + Copy,
    {
        FftPlanner::new().plan_fft_forward(Self::NUM_BINS)
    }

    pub const NUM_BINS: usize = 90;
}

impl<'a, T, P, N>
    Feature<(&'a PointCloud<P>, &'a PointCloud<N>), DVector<T>, &'a mut Option<Arc<dyn Fft<T>>>, ()>
    for Crh<T>
where
    T: RealField + ToPrimitive + Copy,
    P: Point<Data = T>,
    N: Normal<Data = T>,
{
    /// NOTE: `fft` must be produced by self if some.
    fn compute(
        &self,
        (input, normals): (&'a PointCloud<P>, &'a PointCloud<N>),
        fft: &'a mut Option<Arc<dyn Fft<T>>>,
        _: (),
    ) -> DVector<T> {
        let plane_normal = -&self.centroid;
        let axis = plane_normal.normalize().cross(&Vector3::z());
        let (axis, an) = Unit::new_and_get(axis);
        let rotation = an.asin();

        let bin_angle = T::two_pi() / convert(Self::NUM_BINS as f64);
        let transform = Rotation3::new(axis.scale(rotation));

        let (mut buffer, weight) = {
            let grid = input.iter().zip(normals.iter()).map(|(point, normal)| {
                (
                    transform * point.na_point(),
                    transform * normal.normal().xyz(),
                )
            });
            let bin_weight = grid.map(|(point, normal)| {
                let bin = (normal.y.atan2(normal.x) + T::pi()) / bin_angle;
                (
                    bin.to_usize().unwrap() % Self::NUM_BINS,
                    point.xy().coords.norm(),
                )
            });
            bin_weight.fold(
                (vec![Complex::<T>::zero(); Self::NUM_BINS], T::zero()),
                |(mut spatial, weight), (bin, w)| {
                    spatial[bin] += w;
                    (spatial, weight + w)
                },
            )
        };
        buffer.iter_mut().for_each(|data| *data /= weight);

        fft.get_or_insert_with(Self::fft).process(&mut buffer);

        let hist = buffer[..(Self::NUM_BINS / 2)]
            .iter()
            .flat_map(|num| [num.re, num.im])
            .collect::<Vec<_>>();

        hist.into()
    }
}
