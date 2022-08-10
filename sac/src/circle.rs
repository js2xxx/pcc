use nalgebra::{matrix, ComplexField, RealField, Scalar, Vector3, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

use crate::{base::SacModel, line::Line, plane::Plane};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Circle<T: Scalar> {
    pub center: Vector4<T>,
    pub normal: Vector4<T>,
    pub radius: T,
}

impl<T: Scalar> Circle<T> {
    pub fn plane(&self) -> Plane<T> {
        Plane {
            coords: self.center.clone(),
            normal: self.normal.clone(),
        }
    }

    pub fn axis(&self) -> Line<T> {
        Line {
            coords: self.center.clone(),
            direction: self.normal.clone(),
        }
    }
}

impl<T: ComplexField<RealField = T>> Circle<T> {
    pub(crate) fn target_radius(&self, point: &Vector4<T>) -> Vector4<T> {
        let delta = (point - &self.center).xyz();
        let normal = self.normal.xyz();

        // TODO: Check if `normal` is colinear with `delta`.
        let plane = delta.cross(&normal);
        let direction = normal.cross(&plane).normalize();
        let target = direction.scale(self.radius.clone());
        target.insert_row(3, T::zero())
    }

    pub fn circumference_distance(&self, point: &Vector4<T>) -> T {
        let delta = (point - &self.center).xyz();
        let target = self.target_radius(point);
        (target.xyz() - delta).norm()
    }
}

impl<T: RealField> Circle<T> {
    pub fn distance(&self, point: &Vector4<T>) -> T {
        if self.axis().distance(point) <= self.radius {
            self.plane().distance(point)
        } else {
            self.circumference_distance(point)
        }
    }
}

impl<T: RealField + ToPrimitive> Model<Vector4<T>> for Circle<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance(data).to_f64().unwrap()
    }
}

impl<T: RealField + ToPrimitive> SacModel<Vector4<T>> for Circle<T> {
    fn project(&self, coords: &Vector4<T>) -> Vector4<T> {
        if self.axis().distance(coords) <= self.radius {
            self.plane().project(coords)
        } else {
            let target = self.target_radius(coords);
            target + &self.center
        }
    }
}

pub struct CircleEstimator;

impl CircleEstimator {
    pub fn make<T: ComplexField<RealField = T>>(
        a: &Vector4<T>,
        b: &Vector4<T>,
        c: &Vector4<T>,
    ) -> Circle<T> {
        let xa = (b - a).xyz();
        let xb = (c - a).xyz();
        let normal = xa.cross(&xb);
        let d0 = -normal.dot(&a.xyz());

        let a_norm2 = a.xyz().norm_squared();
        let b_norm2 = b.xyz().norm_squared();
        let c_norm2 = c.xyz().norm_squared();

        let xa_2 = xa.scale(T::one() + T::one());
        let xb_2 = xb.scale(T::one() + T::one());
        let matrix_a = matrix![
            normal.x.clone(), normal.y.clone(), normal.z.clone();
            xa_2.x.clone(), xa_2.y.clone(), xa_2.z.clone();
            xb_2.x.clone(), xb_2.y.clone(), xb_2.z.clone()
        ];

        let mut center = Vector3::new(d0, a_norm2.clone() - b_norm2, a_norm2 - c_norm2);
        matrix_a.qr().solve_mut(&mut center);

        let radius = (&center - a.xyz()).norm();

        Circle {
            center: Vector4::from([
                center.x.clone(),
                center.y.clone(),
                center.z.clone(),
                T::one(),
            ]),
            normal: Vector4::from([
                normal.x.clone(),
                normal.y.clone(),
                normal.z.clone(),
                T::zero(),
            ]),
            radius,
        }
    }
}

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for CircleEstimator {
    type Model = Circle<T>;

    type ModelIter = Option<Circle<T>>;

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
