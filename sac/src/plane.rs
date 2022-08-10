use nalgebra::{ComplexField, Scalar, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

use crate::base::SacModel;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Plane<T: Scalar> {
    pub coords: Vector4<T>,
    pub normal: Vector4<T>,
}

impl<T: ComplexField<RealField = T>> Plane<T> {
    pub fn distance_directed(&self, point: &Vector4<T>) -> T {
        let side = (point - self.coords.clone()).xyz();
        let dot = side.dot(&self.normal.xyz());
        dot / self.normal.xyz().norm()
    }

    pub fn distance(&self, point: &Vector4<T>) -> T {
        self.distance_directed(point).abs()
    }

    pub fn distance_squared(&self, point: &Vector4<T>) -> T {
        let side = (point - self.coords.clone()).xyz();
        let dot = side.dot(&self.normal.xyz());
        dot.clone() * dot / self.normal.xyz().norm_squared()
    }
}

impl<T: ComplexField<RealField = T> + ToPrimitive> Model<Vector4<T>> for Plane<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance_directed(data).to_f64().unwrap()
    }
}

impl<T: ComplexField<RealField = T> + ToPrimitive> SacModel<Vector4<T>> for Plane<T> {
    fn project(&self, coords: &Vector4<T>) -> Vector4<T> {
        let distance = self.distance_directed(coords);
        let direction = self.normal.normalize();
        coords - direction * distance
    }
}

pub struct PlaneEstimator;

impl<T: ComplexField<RealField = T> + ToPrimitive> Estimator<Vector4<T>> for PlaneEstimator {
    type Model = Plane<T>;

    type ModelIter = Option<Plane<T>>;

    const MIN_SAMPLES: usize = 3;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        match (data.next(), data.next(), data.next()) {
            (Some(a), Some(b), Some(c)) => {
                let xa = (b - &a).xyz();
                let xb = (c - &a).xyz();
                let normal = xa.cross(&xb);

                Some(Plane {
                    coords: a,
                    normal: Vector4::from([
                        normal.x.clone(),
                        normal.y.clone(),
                        normal.z.clone(),
                        T::zero(),
                    ]),
                })
            }
            _ => None,
        }
    }
}

pub struct PerpendicularPlaneEstimator<T: Scalar> {
    pub normal: Vector4<T>,
}

impl<T: ComplexField<RealField = T> + ToPrimitive> Estimator<Vector4<T>>
    for PerpendicularPlaneEstimator<T>
{
    type Model = Plane<T>;

    type ModelIter = Option<Plane<T>>;

    const MIN_SAMPLES: usize = 1;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        data.next().map(|coords| Plane {
            coords,
            normal: self.normal.clone(),
        })
    }
}

pub struct ParallelPlaneEstimator<T: Scalar> {
    pub direction: Vector4<T>,
}

impl<T: ComplexField<RealField = T> + ToPrimitive> Estimator<Vector4<T>>
    for ParallelPlaneEstimator<T>
{
    type Model = Plane<T>;

    type ModelIter = Option<Plane<T>>;

    const MIN_SAMPLES: usize = 2;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        match (data.next(), data.next()) {
            (Some(a), Some(b)) => {
                let xa = (b - &a).xyz();
                let xb = self.direction.xyz();
                let normal = xa.cross(&xb);

                Some(Plane {
                    coords: a,
                    normal: Vector4::from([
                        normal.x.clone(),
                        normal.y.clone(),
                        normal.z.clone(),
                        T::zero(),
                    ]),
                })
            }
            _ => None,
        }
    }
}
