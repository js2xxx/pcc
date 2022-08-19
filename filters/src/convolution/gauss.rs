use nalgebra::{RealField, Scalar, Vector4};
use pcc_common::point::{Point, PointRgba};

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

impl<'a, T: RealField, P: Point<Data = T> + 'a> DynamicKernel<'a, P> for Gauss<T> {
    fn convolve<Iter>(&self, data: Iter) -> P
    where
        Iter: IntoIterator<Item = (&'a P, T)>,
    {
        let threshold = self.stddev.clone() * self.stddev_mul.clone();
        let var = self.stddev.clone() * self.stddev.clone();

        let (sum, weight) = data.into_iter().fold(
            (Vector4::zeros(), T::zero()),
            |(sum, weight), (point, distance)| {
                if distance <= threshold {
                    let w = (-distance / var.clone() / (T::one() + T::one())).exp();
                    (sum + point.coords() * w.clone(), weight + w)
                } else {
                    (sum, weight)
                }
            },
        );

        P::default().with_coords(if weight != T::zero() {
            sum / weight
        } else {
            Vector4::zeros()
        })
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

impl<'a, T: RealField, P: PointRgba<Data = T> + 'a> DynamicKernel<'a, P> for GaussRgba<T> {
    fn convolve<Iter>(&self, data: Iter) -> P
    where
        Iter: IntoIterator<Item = (&'a P, T)>,
    {
        let threshold = self.inner.stddev.clone() * self.inner.stddev_mul.clone();
        let var = self.inner.stddev.clone() * self.inner.stddev.clone();

        let (sum, rgba, weight) = data.into_iter().fold(
            (Vector4::zeros(), [0.; 4], T::zero()),
            |(sum, [b, g, r, a], weight), (point, distance)| {
                if distance <= threshold {
                    let w = (-distance / var.clone() / (T::one() + T::one())).exp();
                    let rgba = point.rgba_array();
                    (
                        sum + point.coords() * w.clone(),
                        [b + rgba[0], g + rgba[1], r + rgba[2], a + rgba[3]],
                        weight + w,
                    )
                } else {
                    (sum, [b, g, r, a], weight)
                }
            },
        );

        P::default()
            .with_coords(if weight != T::zero() {
                sum / weight
            } else {
                Vector4::zeros()
            })
            .with_rgba_array(&rgba)
    }
}
