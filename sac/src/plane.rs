use nalgebra::{ComplexField, Scalar, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Plane<T: Scalar> {
    pub coords: Vector4<T>,
    pub normal: Vector4<T>,
}

impl<T: Scalar + ComplexField<RealField = T>> Plane<T> {
    pub fn distance(&self, point: &Vector4<T>) -> T {
        let side = (point - self.coords.clone()).xyz();
        let dot = side.dot(&self.normal.xyz());
        dot / self.normal.xyz().norm()
    }

    pub fn distance_squared(&self, point: &Vector4<T>) -> T {
        let side = (point - self.coords.clone()).xyz();
        let dot = side.dot(&self.normal.xyz());
        dot.clone() * dot / self.normal.xyz().norm_squared()
    }
}

impl<T: Scalar + ComplexField<RealField = T> + ToPrimitive> Model<Vector4<T>> for Plane<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance(data).to_f64().unwrap()
    }
}

pub struct PlaneEstimator;

impl<T: Scalar + ComplexField<RealField = T> + ToPrimitive> Estimator<Vector4<T>>
    for PlaneEstimator
{
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

impl<T: Scalar + ComplexField<RealField = T> + ToPrimitive> Estimator<Vector4<T>>
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
