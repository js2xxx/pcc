use nalgebra::{RealField, Scalar, Vector4};
use pcc_common::points::Point3Infoed;

use super::DynamicKernel;

pub struct Gauss<T: Scalar> {
    pub stddev: T,
    pub stddev_mul: T,
}

impl<T: Scalar> Gauss<T> {
    pub fn new(stddev: T, stddev_mul: T) -> Self {
        Gauss { stddev, stddev_mul }
    }
}

impl<T: RealField> DynamicKernel<T> for Gauss<T> {
    fn convolve<'a, I: 'a + Default, Iter>(&self, data: Iter) -> Point3Infoed<T, I>
    where
        Iter: IntoIterator<Item = (&'a Point3Infoed<T, I>, T)>,
    {
        let threshold = self.stddev.clone() * self.stddev_mul.clone();
        let var = self.stddev.clone() * self.stddev.clone();

        let (sum, weight) = data.into_iter().fold(
            (Vector4::zeros(), T::zero()),
            |(sum, weight), (point, distance)| {
                if distance <= threshold {
                    let w = (-distance / var.clone() / (T::one() + T::one())).exp();
                    (sum + &point.coords * w.clone(), weight + w)
                } else {
                    (sum, weight)
                }
            },
        );

        Point3Infoed {
            coords: if weight != T::zero() {
                sum / weight
            } else {
                Vector4::zeros()
            },
            extra: Default::default(),
        }
    }
}
