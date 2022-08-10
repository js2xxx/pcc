use nalgebra::{RealField, Scalar, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

use crate::base::SacModel;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Line<T: Scalar> {
    pub coords: Vector4<T>,
    pub direction: Vector4<T>,
}

impl<T: RealField> Line<T> {
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

impl<T: RealField> Line<T> {
    pub fn stick_distance_squared(&self, point: &Vector4<T>) -> T {
        let v1 = (point - &self.coords).xyz();
        let v2 = (point - &self.coords - &self.direction).xyz();

        if v1.dot(&self.direction.xyz()) < T::zero() {
            v1.norm_squared()
        } else if v2.dot(&self.direction.xyz()) > T::zero() {
            v2.norm_squared()
        } else {
            self.distance_squared(point)
        }
    }

    pub fn stick_distance(&self, point: &Vector4<T>) -> T {
        let v1 = (point - &self.coords).xyz();
        let v2 = (point - &self.coords - &self.direction).xyz();

        if v1.dot(&self.direction.xyz()) < T::zero() {
            v1.norm()
        } else if v2.dot(&self.direction.xyz()) > T::zero() {
            v2.norm()
        } else {
            self.distance(point)
        }
    }
}

impl<T: RealField + ToPrimitive> Model<Vector4<T>> for Line<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance(data).to_f64().unwrap()
    }
}

impl<T: RealField + ToPrimitive> SacModel<Vector4<T>> for Line<T> {
    fn project(&self, coords: &Vector4<T>) -> Vector4<T> {
        let distance = self.distance(coords);
        let direction = { (coords - &self.coords).xyz() }
            .cross(&self.direction.xyz())
            .cross(&self.direction.xyz())
            .normalize();
        let direction = direction.insert_row(3, T::zero());
        coords - direction * distance
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Stick<T: Scalar>(pub Line<T>);

impl<T: RealField> Stick<T> {
    pub fn distance_squared(&self, point: &Vector4<T>) -> T {
        self.0.stick_distance_squared(point)
    }

    pub fn distance(&self, point: &Vector4<T>) -> T {
        self.0.stick_distance(point)
    }
}

impl<T: RealField + ToPrimitive> Model<Vector4<T>> for Stick<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance(data).to_f64().unwrap()
    }
}

impl<T: RealField + ToPrimitive> SacModel<Vector4<T>> for Stick<T> {
    fn project(&self, coords: &Vector4<T>) -> Vector4<T> {
        let v1 = (coords - &self.0.coords).xyz();
        let v2 = (coords - &self.0.coords - &self.0.direction).xyz();

        if v1.dot(&self.0.direction.xyz()) < T::zero() {
            v1.insert_row(3, T::one())
        } else if v2.dot(&self.0.direction.xyz()) > T::zero() {
            v2.insert_row(3, T::one())
        } else {
            self.0.project(coords)
        }
    }
}

pub struct LineEstimator;

impl LineEstimator {
    pub fn make<T: RealField>(a: &Vector4<T>, b: &Vector4<T>) -> Line<T> {
        Line {
            coords: a.clone(),
            direction: b - a,
        }
    }
}

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for LineEstimator {
    type Model = Line<T>;

    type ModelIter = Option<Line<T>>;

    const MIN_SAMPLES: usize = 2;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        match (data.next(), data.next()) {
            (Some(a), Some(b)) => Some(Self::make(&a, &b)),
            _ => None,
        }
    }
}

pub struct StickEstimator;

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for StickEstimator {
    type Model = Stick<T>;

    type ModelIter = Option<Stick<T>>;

    const MIN_SAMPLES: usize = 2;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        match (data.next(), data.next()) {
            (Some(a), Some(b)) => Some(Stick(LineEstimator::make(&a, &b))),
            _ => None,
        }
    }
}

pub struct ParallelLineEstimator<T: Scalar> {
    pub direction: Vector4<T>,
}

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for ParallelLineEstimator<T> {
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
