use nalgebra::{ComplexField, Scalar, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Line<T: Scalar> {
    pub coords: Vector4<T>,
    pub direction: Vector4<T>,
}

impl<T: Scalar + ComplexField<RealField = T>> Line<T> {
    pub fn distance_squared(&self, point: &Vector4<T>) -> T {
        let side = (point - self.coords.clone()).xyz();
        let dot = side.dot(&self.direction.xyz());
        let height2 = dot.clone() * dot / self.direction.xyz().norm_squared();

        side.norm_squared() - height2
    }

    pub fn distance(&self, point: &Vector4<T>) -> T {
        self.distance_squared(point).sqrt()
    }
}

impl<T: Scalar + ComplexField<RealField = T> + ToPrimitive> Model<Vector4<T>> for Line<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance(data).to_f64().unwrap()
    }
}

pub struct LineEstimator;

impl<T: Scalar + ComplexField<RealField = T> + ToPrimitive> Estimator<Vector4<T>>
    for LineEstimator
{
    type Model = Line<T>;

    type ModelIter = Option<Line<T>>;

    const MIN_SAMPLES: usize = 2;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        match (data.next(), data.next()) {
            (Some(a), Some(b)) => Some(Line {
                coords: a.clone(),
                direction: b - a,
            }),
            _ => None,
        }
    }
}

pub struct ParallelLineEstimator<T: Scalar> {
    pub direction: Vector4<T>,
}

impl<T: Scalar + ComplexField<RealField = T> + ToPrimitive> Estimator<Vector4<T>>
    for ParallelLineEstimator<T>
{
    type Model = Line<T>;

    type ModelIter = Option<Line<T>>;

    const MIN_SAMPLES: usize = 1;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        data.next().map(|coords| Line {
            coords,
            direction: self.direction.clone(),
        })
    }
}
