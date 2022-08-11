#[macro_use]
mod macros;
mod centroid;

use std::collections::HashMap;

pub use nalgebra::Point3;
use nalgebra::{ComplexField, Scalar, Vector4};

pub use self::centroid::{Centroid, CentroidBuilder};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(align(16))]
pub struct Point3Infoed<T: Scalar, I> {
    pub coords: Vector4<T>,
    pub extra: I,
}

impl<T: ComplexField, I: Centroid> Centroid for Point3Infoed<T, I> {
    type Accumulator = (<Vector4<T> as Centroid>::Accumulator, I::Accumulator);

    fn accumulate(&self, accum: &mut Self::Accumulator) {
        self.coords.accumulate(&mut accum.0);
        self.extra.accumulate(&mut accum.1);
    }

    fn compute(accum: Self::Accumulator, num: usize) -> Self {
        Point3Infoed {
            coords: Centroid::compute(accum.0, num),
            extra: Centroid::compute(accum.1, num),
        }
    }
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

impl<T: ComplexField, I> Point3Infoed<T, I> {
    pub fn is_finite(&self) -> bool {
        self.coords.x.is_finite() && self.coords.y.is_finite() && self.coords.z.is_finite()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(align(16))]
pub struct PointInfoHsv<T: Scalar> {
    pub h: T,
    pub s: T,
    pub v: T,
}

impl<T: ComplexField> Centroid for PointInfoHsv<T> {
    type Accumulator = Self;

    fn accumulate(&self, other: &mut Self) {
        other.h += self.h.clone();
        other.s += self.s.clone();
        other.v += self.v.clone();
    }

    fn compute(accum: Self, num: usize) -> Self {
        let num = T::from_usize(num).unwrap();
        PointInfoHsv {
            h: accum.h / num.clone(),
            s: accum.s / num.clone(),
            v: accum.v / num,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(align(16))]
pub struct PointInfoIntensity<T: Scalar> {
    pub intensity: T,
}

impl<T: ComplexField> Centroid for PointInfoIntensity<T> {
    type Accumulator = Self;

    fn accumulate(&self, other: &mut Self) {
        other.intensity += self.intensity.clone();
    }

    fn compute(accum: Self, num: usize) -> Self {
        let num = T::from_usize(num).unwrap();
        PointInfoIntensity {
            intensity: accum.intensity / num,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(align(16))]
pub struct PointInfoLabel {
    pub label: u32,
}

impl Centroid for PointInfoLabel {
    type Accumulator = HashMap<u32, usize>;

    fn accumulate(&self, accum: &mut Self::Accumulator) {
        if let Err(mut e) = accum.try_insert(self.label, 1) {
            *e.entry.get_mut() += 1;
        }
    }

    fn compute(accum: Self::Accumulator, _: usize) -> Self {
        let (label, _) = { accum.into_iter() }
            .fold(None, |acc, (label, times)| match acc {
                Some((_, t)) if t >= times => acc,
                _ => Some((label, times)),
            })
            .unwrap();
        PointInfoLabel { label }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(align(16))]
pub struct PointInfoRgba {
    pub rgba: u32,
}

impl From<[f32; 4]> for PointInfoRgba {
    fn from(rgba: [f32; 4]) -> Self {
        PointInfoRgba {
            rgba: (rgba[0]) as u32
                | (((rgba[1]) as u32) << 8)
                | (((rgba[2]) as u32) << 16)
                | (((rgba[3]) as u32) << 24),
        }
    }
}

impl From<PointInfoRgba> for [f32; 4] {
    fn from(info: PointInfoRgba) -> Self {
        [
            (info.rgba & 0xff) as f32,
            ((info.rgba >> 8) & 0xff) as f32,
            ((info.rgba >> 16) & 0xff) as f32,
            (info.rgba >> 24) as f32,
        ]
    }
}

impl Centroid for PointInfoRgba {
    type Accumulator = [f32; 4];

    fn accumulate(&self, accum: &mut [f32; 4]) {
        accum[0] += (self.rgba & 0xff) as f32;
        accum[1] += ((self.rgba >> 8) & 0xff) as f32;
        accum[2] += ((self.rgba >> 16) & 0xff) as f32;
        accum[3] += (self.rgba >> 24) as f32;
    }

    fn compute(accum: [f32; 4], num: usize) -> Self {
        let num = num as f32;
        PointInfoRgba {
            rgba: (accum[0] / num) as u32
                | (((accum[1] / num) as u32) << 8)
                | (((accum[2] / num) as u32) << 16)
                | (((accum[3] / num) as u32) << 24),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(align(16))]
pub struct PointInfoNormal<T: Scalar> {
    pub normal: Vector4<T>,
    pub curvature: T,
}

impl<T: ComplexField> Centroid for PointInfoNormal<T> {
    type Accumulator = Self;
    fn accumulate(&self, other: &mut Self) {
        other.normal += &self.normal;
        other.curvature += self.curvature.clone();
    }

    fn compute(accum: Self, num: usize) -> Self {
        let num = T::from_usize(num).unwrap();
        PointInfoNormal {
            normal: accum.normal / num.clone(),
            curvature: accum.curvature / num,
        }
    }
}

impl_pi!(PointHsv<T>:       point_hsv       => PointInfoHsv; ;      A, B, C, D, E, F, G, H, I, J, K);
impl_pi!(PointIntensity<T>: point_intensity => PointInfoIntensity;  A; B, C, D, E, F, G, H, I, J, K);
impl_pi!(PointLabel:        point_label     => PointInfoLabel;      A, B; C, D, E, F, G, H, I, J, K);
impl_pi!(PointRgba:         point_rgba      => PointInfoRgba;       A, B, C; D, E, F, G, H, I, J, K);
impl_pi!(PointNormal<T>:    point_normal    => PointInfoNormal;     A, B, C, D; E, F, G, H, I, J, K);

pub type Point3H<T> = Point3Infoed<T, PointInfoHsv<T>>;
pub type Point3I<T> = Point3Infoed<T, PointInfoIntensity<T>>;
pub type Point3L<T> = Point3Infoed<T, PointInfoLabel>;
pub type Point3R<T> = Point3Infoed<T, PointInfoRgba>;
pub type Point3N<T> = Point3Infoed<T, PointInfoNormal<T>>;
pub type Point3RN<T> = Point3Infoed<T, (PointInfoRgba, PointInfoNormal<T>)>;
pub type Point3IN<T> = Point3Infoed<T, (PointInfoIntensity<T>, PointInfoNormal<T>)>;
pub type Point3LN<T> = Point3Infoed<T, (PointInfoLabel, PointInfoNormal<T>)>;
