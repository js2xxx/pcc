use nalgebra::{convert, matrix, RealField, Scalar, Vector3, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

use crate::base::SacModel;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Sphere<T: Scalar> {
    pub coords: Vector4<T>,
    pub radius: T,
}

impl<T: RealField> Sphere<T> {
    pub fn distance(&self, point: &Vector4<T>) -> T {
        self.distance_directed(point).abs()
    }

    pub fn distance_directed(&self, point: &Vector4<T>) -> T {
        let radius = (point - &self.coords).xyz().norm();
        radius - self.radius.clone()
    }
}

impl<T: RealField + ToPrimitive> Model<Vector4<T>> for Sphere<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance(data).to_f64().unwrap()
    }
}

impl<T: RealField + ToPrimitive> SacModel<Vector4<T>> for Sphere<T> {
    fn project(&self, coords: &Vector4<T>) -> Vector4<T> {
        let distance = self.distance_directed(coords);
        let direction = (coords - &self.coords).normalize();
        coords - direction * distance
    }
}

pub struct SphereEstimator;

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for SphereEstimator {
    type Model = Sphere<T>;

    type ModelIter = Option<Sphere<T>>;

    const MIN_SAMPLES: usize = 4;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        match (data.next(), data.next(), data.next(), data.next()) {
            (Some(a), Some(b), Some(c), Some(d)) => {
                let xa_2 = (&b - &a).xyz() * convert::<_, T>(2.);
                let xb_2 = (&c - &a).xyz() * convert::<_, T>(2.);
                let xc_2 = (&d - &a).xyz() * convert::<_, T>(2.);

                let a_norm2 = a.xyz().norm_squared();
                let b_norm2 = b.xyz().norm_squared();
                let c_norm2 = c.xyz().norm_squared();
                let d_norm2 = d.xyz().norm_squared();

                let matrix_a = matrix![
                    xa_2.x.clone(), xa_2.y.clone(), xa_2.z.clone();
                    xb_2.x.clone(), xb_2.y.clone(), xb_2.z.clone();
                    xc_2.x.clone(), xc_2.y.clone(), xc_2.z.clone()
                ];
                let mut coords = Vector3::new(
                    a_norm2.clone() - b_norm2,
                    a_norm2.clone() - c_norm2,
                    a_norm2 - d_norm2,
                );

                matrix_a.qr().solve_mut(&mut coords);
                let radius = (&coords - a.xyz()).norm();

                Some(Sphere {
                    coords: Vector4::from([
                        coords.x.clone(),
                        coords.y.clone(),
                        coords.z.clone(),
                        T::one(),
                    ]),
                    radius,
                })
            }
            _ => None,
        }
    }
}
