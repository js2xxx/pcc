use nalgebra::{RealField, Scalar, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

use crate::base::SacModel;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Plane<T: Scalar> {
    pub coords: Vector4<T>,
    pub normal: Vector4<T>,
}

impl<T: RealField> Plane<T> {
    pub fn same_side_with_normal(&self, coords: &Vector4<T>) -> bool {
        let side = (coords - self.coords.clone()).xyz();
        let dot = side.dot(&self.normal.xyz());
        dot >= T::zero()
    }
}

impl<T: RealField> Plane<T> {
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

impl<T: RealField + ToPrimitive> Model<Vector4<T>> for Plane<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance_directed(data).to_f64().unwrap()
    }
}

impl<T: RealField + ToPrimitive> SacModel<Vector4<T>> for Plane<T> {
    fn project(&self, coords: &Vector4<T>) -> Vector4<T> {
        let distance = self.distance_directed(coords);
        let direction = self.normal.normalize();
        coords - direction * distance
    }
}

pub struct PlaneEstimator;

impl PlaneEstimator {
    pub fn make<T: RealField>(a: &Vector4<T>, b: &Vector4<T>, c: &Vector4<T>) -> Plane<T> {
        let xa = (a - b).xyz();
        let xb = (a - c).xyz();
        let normal = xa.cross(&xb);

        Plane {
            coords: a.clone(),
            normal: Vector4::from([
                normal.x.clone(),
                normal.y.clone(),
                normal.z.clone(),
                T::zero(),
            ]),
        }
    }
}

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for PlaneEstimator {
    type Model = Plane<T>;

    type ModelIter = Option<Plane<T>>;

    const MIN_SAMPLES: usize = 3;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        match (data.next(), data.next(), data.next()) {
            (Some(a), Some(b), Some(c)) => Some(Self::make(&a, &b, &c)),
            _ => None,
        }
    }
}

pub struct PerpendicularPlaneEstimator<T: Scalar> {
    pub normal: Vector4<T>,
}

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for PerpendicularPlaneEstimator<T> {
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

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for ParallelPlaneEstimator<T> {
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
