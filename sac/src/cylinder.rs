use nalgebra::{RealField, Scalar, Vector4};
use num::ToPrimitive;
use sample_consensus::{Estimator, Model};

use crate::{
    base::SacModel,
    circle::{Circle, CircleEstimator},
    line::{Line, Stick},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Cylinder<T: Scalar> {
    pub circle: Circle<T>,
    pub height: T,
}

impl<T: RealField> Cylinder<T> {
    pub fn top_circle(&self) -> Circle<T> {
        let diff = self
            .circle
            .normal
            .scale(self.height.clone() / self.circle.normal.norm());
        Circle {
            center: &self.circle.center + diff,
            normal: self.circle.normal.clone(),
            radius: self.circle.radius.clone(),
        }
    }

    pub fn generatrix(&self, point: &Vector4<T>) -> Stick<T> {
        Stick(Line {
            coords: self.circle.target_radius(point),
            direction: self
                .circle
                .normal
                .scale(self.height.clone() / self.circle.normal.norm()),
        })
    }
}

impl<T: RealField> Cylinder<T> {
    pub fn distance(&self, point: &Vector4<T>) -> T {
        let top_circle = self.top_circle();

        let mut ret = self.circle.distance(point);

        let d1 = top_circle.distance(point);
        if ret > d1 {
            ret = d1
        }

        let d2 = self.generatrix(point).distance(point);
        if ret > d2 {
            ret = d2
        }

        ret
    }
}

impl<T: RealField + ToPrimitive> Model<Vector4<T>> for Cylinder<T> {
    fn residual(&self, data: &Vector4<T>) -> f64 {
        self.distance(data).to_f64().unwrap()
    }
}

impl<T: RealField + ToPrimitive> SacModel<Vector4<T>> for Cylinder<T> {
    fn project(&self, coords: &Vector4<T>) -> Vector4<T> {
        let top_circle = self.top_circle();
        let (circle, circle_distance) = {
            let d1 = self.circle.distance(coords);
            let d2 = top_circle.distance(coords);
            if d1 < d2 {
                (&self.circle, d1)
            } else {
                (&top_circle, d2)
            }
        };

        let generatrix = self.generatrix(coords);
        if circle_distance >= generatrix.distance(coords) {
            circle.project(coords)
        } else {
            generatrix.project(coords)
        }
    }
}

pub struct CylinderEstimator;

impl CylinderEstimator {
    pub fn try_make<T: RealField>(
        ca: &Vector4<T>,
        cb: &Vector4<T>,
        cc: &Vector4<T>,
        top: &Vector4<T>,
    ) -> Option<Cylinder<T>> {
        let circle = CircleEstimator::make(ca, cb, cc);

        let plane = circle.plane();
        let axis = circle.axis();

        (axis.distance(top) <= circle.radius).then(|| Cylinder {
            circle,
            height: plane.distance(top),
        })
    }
}

impl<T: RealField + ToPrimitive> Estimator<Vector4<T>> for CylinderEstimator {
    type Model = Cylinder<T>;

    type ModelIter = Vec<Cylinder<T>>;

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
