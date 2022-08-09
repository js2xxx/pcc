use nalgebra::{ComplexField, RealField, Scalar, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

use crate::{
    circle::{Circle, CircleEstimator},
    line::{Line, LineEstimator},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Cone<T: Scalar> {
    circle: Circle<T>,
    height: T,
}

impl<T: ComplexField<RealField = T>> Cone<T> {
    pub fn top_point(&self) -> Vector4<T> {
        let diff = self
            .circle
            .normal
            .scale(self.height.clone() / self.circle.normal.norm());

        &self.circle.center + diff
    }

    pub fn generatrix(&self, point: &Vector4<T>) -> Line<T> {
        let a = self.circle.target_radius(point) + &self.circle.center;
        let b = self.top_point();
        LineEstimator::make(&a, &b)
    }
}

impl<T: RealField> Cone<T> {
    pub fn distance(&self, point: &Vector4<T>) -> T {
        let generatrix = self.generatrix(point);
        let top_point = self.top_point();

        let mut ret = self.circle.distance(point);

        let d1 = (point - top_point).xyz().norm();
        if ret > d1 {
            ret = d1
        }

        let d2 = generatrix.stick_distance(point);
        if ret > d2 {
            ret = d2
        }

        ret
    }
}

impl<T: RealField + ToPrimitive> Model<Vector4<T>> for Cone<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance(data).to_f64().unwrap()
    }
}

pub struct ConeEstimator;

impl ConeEstimator {
    fn try_make<T: RealField>(
        ca: &Vector4<T>,
        cb: &Vector4<T>,
        cc: &Vector4<T>,
        another: &Vector4<T>,
    ) -> Option<Cone<T>> {
        let circle = CircleEstimator::make(ca, cb, cc);
        let target = circle.target_radius(another);
        let dir_gx = another - &target - &circle.center;

        let dot = dir_gx.xyz().dot(&target.xyz());
        (dot > T::zero()).then(|| {
            let cos2 =
                dot.clone() * dot / dir_gx.xyz().norm_squared() / target.xyz().norm_squared();
            let tan = (T::one() / cos2 - T::one()).sqrt();
            let height = tan * circle.radius.clone();
            Cone { circle, height }
        })
    }
}

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for ConeEstimator {
    type Model = Cone<T>;

    type ModelIter = Vec<Cone<T>>;

    const MIN_SAMPLES: usize = 4;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector4<T>> + Clone,
    {
        match (data.next(), data.next(), data.next(), data.next()) {
            (Some(a), Some(b), Some(c), Some(d)) => { Self::try_make(&a, &b, &c, &d).into_iter() }
                .chain(Self::try_make(&b, &c, &d, &a).into_iter())
                .chain(Self::try_make(&c, &d, &a, &b).into_iter())
                .chain(Self::try_make(&d, &a, &b, &c).into_iter())
                .collect(),
            _ => vec![],
        }
    }
}
