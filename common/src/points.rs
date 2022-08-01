pub use nalgebra::Point3;
use nalgebra::{Scalar, Vector4};
use num::Float;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Point3Infoed<T: Scalar, I> {
    pub coords: Vector4<T>,
    pub extra: I,
}

impl<T: Scalar + Default, I: Default> Default for Point3Infoed<T, I> {
    fn default() -> Self {
        Self {
            coords: Default::default(),
            extra: Default::default(),
        }
    }
}

impl<T: Scalar + Float, I> Point3Infoed<T, I> {
    pub fn is_finite(&self) -> bool {
        self.coords.x.is_finite() && self.coords.y.is_finite() && self.coords.z.is_finite()
    }
}

pub struct PointInfoHsv<T: Scalar> {
    pub h: T,
    pub s: T,
    pub v: T,
}

pub struct PointInfoIntensity<T: Scalar> {
    pub intensity: T,
}

pub struct PointInfoLabel {
    pub label: u32,
}

pub struct PointInfoRgba {
    pub rgba: u32,
}

pub struct PointInfoRgbaLabel {
    pub rgba: u32,
    pub label: u32,
}
