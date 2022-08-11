use nalgebra::{RealField, Scalar, Vector4};
use pcc_common::points::{Point3Infoed, PointRgba};

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

impl<'a, T: RealField, I: 'a + Default> DynamicKernel<'a, T, I> for Gauss<T> {
    fn convolve<Iter>(&self, data: Iter) -> Point3Infoed<T, I>
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

pub struct GaussRgba<T: Scalar> {
    pub inner: Gauss<T>,
}

impl<T: Scalar> GaussRgba<T> {
    pub fn new(stddev: T, stddev_mul: T) -> Self {
        GaussRgba {
            inner: Gauss::new(stddev, stddev_mul),
        }
    }
}

impl<'a, T: RealField, I: 'a + Default + PointRgba> DynamicKernel<'a, T, I> for GaussRgba<T> {
    fn convolve<Iter>(&self, data: Iter) -> Point3Infoed<T, I>
    where
        Iter: IntoIterator<Item = (&'a Point3Infoed<T, I>, T)>,
    {
        let threshold = self.inner.stddev.clone() * self.inner.stddev_mul.clone();
        let var = self.inner.stddev.clone() * self.inner.stddev.clone();

        let (sum, [r, g, b, a], weight) = data.into_iter().fold(
            (Vector4::zeros(), [0.; 4], T::zero()),
            |(sum, [r, g, b, a], weight), (point, distance)| {
                if distance <= threshold {
                    let w = (-distance / var.clone() / (T::one() + T::one())).exp();
                    let rgba: [f32; 4] = (*point.extra.point_rgba()).into();
                    (
                        sum + &point.coords * w.clone(),
                        [r + rgba[0], g + rgba[1], b + rgba[2], a + rgba[3]],
                        weight + w,
                    )
                } else {
                    (sum, [r, g, b, a], weight)
                }
            },
        );

        Point3Infoed {
            coords: if weight != T::zero() {
                sum / weight
            } else {
                Vector4::zeros()
            },
            extra: I::from([r, g, b, a].into()),
        }
    }
}
