pub use nalgebra::Point3;
use nalgebra::{Scalar, Vector4, ComplexField};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(align(16))]
pub struct Point3Infoed<T: Scalar, I> {
    pub coords: Vector4<T>,
    pub extra: I,
}

impl<T: Scalar, I> AsRef<Vector4<T>> for Point3Infoed<T, I> {
    fn as_ref(&self) -> &Vector4<T> {
        &self.coords
    }
}

impl<T: Scalar + Default, I: Default> Default for Point3Infoed<T, I> {
    fn default() -> Self {
        Self {
            coords: Default::default(),
            extra: Default::default(),
        }
    }
}

impl<T: Scalar + ComplexField<RealField = T>, I> Point3Infoed<T, I> {
    pub fn is_finite(&self) -> bool {
        self.coords.x.is_finite() && self.coords.y.is_finite() && self.coords.z.is_finite()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(align(16))]
pub struct PointInfoHsv<T: Scalar> {
    pub h: T,
    pub s: T,
    pub v: T,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(align(16))]
pub struct PointInfoIntensity<T: Scalar> {
    pub intensity: T,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(align(16))]
pub struct PointInfoLabel {
    pub label: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(align(16))]
pub struct PointInfoRgba {
    pub rgba: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(align(16))]
pub struct PointInfoNormal<T: Scalar> {
    pub normal: Vector4<T>,
    pub curvature: T,
}

pub type Point3H<T> = Point3Infoed<T, PointInfoHsv<T>>;
pub type Point3I<T> = Point3Infoed<T, PointInfoIntensity<T>>;
pub type Point3L<T> = Point3Infoed<T, PointInfoLabel>;
pub type Point3R<T> = Point3Infoed<T, PointInfoRgba>;
pub type Point3N<T> = Point3Infoed<T, PointInfoNormal<T>>;
pub type Point3RN<T> = Point3Infoed<T, (PointInfoRgba, PointInfoNormal<T>)>;
pub type Point3IN<T> = Point3Infoed<T, (PointInfoIntensity<T>, PointInfoNormal<T>)>;
pub type Point3LN<T> = Point3Infoed<T, (PointInfoLabel, PointInfoNormal<T>)>;
