mod info;
#[macro_use]
mod macros;
mod centroid;

use core::fmt::Debug;
use std::{array, collections::HashMap};

use nalgebra::{ComplexField, RawStorage, RawStorageMut, SVector, Scalar, ToConst, Vector4};
use num::FromPrimitive;
use static_assertions::const_assert;
use typenum::{Unsigned, U10, U4, U5, U8, U9};

pub use self::{
    centroid::{Centroid, CentroidBuilder},
    info::{DataFields, FieldInfo},
};

pub trait Data: Debug + Clone + PartialEq + PartialOrd + Default {
    type Data: Scalar;

    fn as_slice(&self) -> &[Self::Data];

    fn as_mut_slice(&mut self) -> &mut [Self::Data];

    fn is_finite(&self) -> bool;
}

pub trait Point: Data {
    type Dim: Unsigned + ToConst;

    fn coords(&self) -> &Vector4<Self::Data>;

    fn coords_mut(&mut self) -> &mut Vector4<Self::Data>;
    #[inline]
    fn with_coords(mut self, coords: Vector4<Self::Data>) -> Self {
        *self.coords_mut() = coords;
        self
    }

    #[inline]
    fn na_point(&self) -> nalgebra::Point3<Self::Data>
    where
        Self::Data: ComplexField,
    {
        nalgebra::Point3::from_homogeneous(self.coords().clone()).unwrap()
    }

    #[inline]
    fn fields() -> array::IntoIter<FieldInfo, 3> {
        [
            FieldInfo::single::<Self::Data>("x", 0),
            FieldInfo::single::<Self::Data>("y", 1),
            FieldInfo::single::<Self::Data>("z", 2),
        ]
        .into_iter()
    }
}

/// Lower bits <---[b; g; r; a]: [u8; 4]---- Higher bits
pub trait PointRgba: Point {
    fn rgb_value(&self) -> Self::Data;

    fn set_rgb_value(&mut self, value: Self::Data);
    #[inline]
    fn with_rgb_value(mut self, value: Self::Data) -> Self {
        self.set_rgb_value(value);
        self
    }

    fn rgba(&self) -> u32;

    fn set_rgba(&mut self, rgba: u32);
    #[inline]
    fn with_rgba(mut self, rgba: u32) -> Self {
        self.set_rgba(rgba);
        self
    }

    #[inline]
    fn rgba_array(&self) -> [f32; 4] {
        let rgba = self.rgba();
        [
            (rgba & 0xff) as f32,
            ((rgba >> 8) & 0xff) as f32,
            ((rgba >> 16) & 0xff) as f32,
            (rgba >> 24) as f32,
        ]
    }

    #[inline]
    fn set_rgba_array(&mut self, rgba: &[f32; 4]) {
        self.set_rgba(
            (rgba[0]) as u32
                | (((rgba[1]) as u32) << 8)
                | (((rgba[2]) as u32) << 16)
                | (((rgba[3]) as u32) << 24),
        )
    }
    #[inline]
    fn with_rgba_array(mut self, rgba: &[f32; 4]) -> Self {
        self.set_rgba_array(rgba);
        self
    }

    fn fields() -> array::IntoIter<FieldInfo, 1>;

    type CentroidAccumulator = [f32; 4];

    #[inline]
    fn centroid_accumulate(&self, accum: &mut [f32; 4]) {
        accum[0] += (self.rgba() & 0xff) as f32;
        accum[1] += ((self.rgba() >> 8) & 0xff) as f32;
        accum[2] += ((self.rgba() >> 16) & 0xff) as f32;
        accum[3] += (self.rgba() >> 24) as f32;
    }

    #[inline]
    fn centroid_compute(&mut self, accum: [f32; 4], num: usize) {
        let num = num as f32;
        self.set_rgba(
            (accum[0] / num) as u32
                | (((accum[1] / num) as u32) << 8)
                | (((accum[2] / num) as u32) << 16)
                | (((accum[3] / num) as u32) << 24),
        )
    }
}

pub trait Normal: Debug + Clone + PartialEq + PartialOrd + Default {
    type Data: Scalar;

    fn normal(&self) -> &Vector4<Self::Data>;

    fn normal_mut(&mut self) -> &mut Vector4<Self::Data>;
    #[inline]
    fn with_normal(mut self, normal: Vector4<Self::Data>) -> Self {
        *self.normal_mut() = normal;
        self
    }

    fn curvature(&self) -> Self::Data;

    fn curvature_mut(&mut self) -> &mut Self::Data;
    #[inline]
    fn set_curvature(&mut self, curvature: Self::Data) {
        *self.curvature_mut() = curvature;
    }
    #[inline]
    fn with_curvature(mut self, curvature: Self::Data) -> Self {
        *self.curvature_mut() = curvature;
        self
    }

    fn fields() -> array::IntoIter<FieldInfo, 2>;

    type CentroidAccumulator = (Vector4<Self::Data>, Self::Data);

    #[inline]
    fn centroid_accumulate(&self, accum: &mut (Vector4<Self::Data>, Self::Data))
    where
        Self::Data: ComplexField,
    {
        accum.0 += self.normal();
        accum.1 += self.curvature();
    }

    #[inline]
    fn centroid_compute(&mut self, accum: (Vector4<Self::Data>, Self::Data), num: usize)
    where
        Self::Data: ComplexField,
    {
        let num = <Self::Data>::from_usize(num).unwrap();
        self.normal_mut().set_column(0, &(accum.0 / num.clone()));
        self.set_curvature(accum.1 / num)
    }
}

pub trait PointNormal: Point + Normal<Data = <Self as Data>::Data> {
    type Data;
}
impl<T: Point + Normal<Data = <Self as Data>::Data>> PointNormal for T {
    type Data = <Self as Data>::Data;
}

pub trait PointIntensity: Point {
    fn intensity(&self) -> Self::Data;

    fn set_intensity(&mut self, intensity: Self::Data);
    #[inline]
    fn with_intensity(mut self, intensity: Self::Data) -> Self {
        self.set_intensity(intensity);
        self
    }

    fn fields() -> array::IntoIter<FieldInfo, 1>;

    type CentroidAccumulator = Self::Data;

    #[inline]
    fn centroid_accumulate(&self, accum: &mut Self::Data)
    where
        Self::Data: ComplexField,
    {
        *accum += self.intensity();
    }

    #[inline]
    fn centroid_compute(&mut self, accum: Self::Data, num: usize)
    where
        Self::Data: ComplexField,
    {
        let num = Self::Data::from_usize(num).unwrap();
        self.set_intensity(accum / num);
    }
}

pub trait PointRange: Point {
    fn range(&self) -> Self::Data;

    fn range_mut(&mut self) -> &mut Self::Data;
    #[inline]
    fn set_range(&mut self, range: Self::Data) {
        *self.range_mut() = range;
    }
    #[inline]
    fn with_range(mut self, range: Self::Data) -> Self {
        self.set_range(range);
        self
    }

    fn fields() -> array::IntoIter<FieldInfo, 1>;
}

pub trait PointLabel: Point {
    fn label(&self) -> u32;

    fn set_label(&mut self, label: u32);
    #[inline]
    fn with_label(mut self, label: u32) -> Self {
        self.set_label(label);
        self
    }

    fn fields() -> array::IntoIter<FieldInfo, 1>;

    type CentroidAccumulator = HashMap<u32, usize>;

    #[inline]
    fn centroid_accumulate(&self, accum: &mut HashMap<u32, usize>) {
        if let Err(mut e) = accum.try_insert(self.label(), 1) {
            *e.entry.get_mut() += 1;
        }
    }

    #[inline]
    fn centroid_compute(&mut self, accum: HashMap<u32, usize>, _: usize) {
        let (label, _) = { accum.into_iter() }
            .fold(None, |acc, (label, times)| match acc {
                Some((_, t)) if t >= times => acc,
                _ => Some((label, times)),
            })
            .unwrap();
        self.set_label(label);
    }
}

pub trait PointViewpoint: Point {
    fn viewpoint(&self) -> &Vector4<Self::Data>;

    fn viewpoint_mut(&mut self) -> &mut Vector4<Self::Data>;
    #[inline]
    fn with_viewpoint(mut self, viewpoint: Vector4<Self::Data>) -> Self {
        *self.viewpoint_mut() = viewpoint;
        self
    }

    fn fields() -> array::IntoIter<FieldInfo, 1>;
}

define_points! {
    #[auto_centroid]
    pub struct Point3<f32, U4>;

    #[auto_centroid]
    pub struct Point3Rgba<f32, U5> {
        rgba: PointRgba [4],
    }

    #[auto_centroid]
    pub struct Point3N<f32, U9> {
        normal: Normal [4, 8],
    }

    #[auto_centroid]
    pub struct Point3RgbaN<f32, U10> {
        normal: Normal [4, 8],
        rgba: PointRgba [9],
    }

    #[auto_centroid]
    pub struct Point3IN<f32, U10> {
        normal: Normal [4, 8],
        intensity: PointIntensity [9],
    }

    #[auto_centroid]
    pub struct Point3LN<f32, U10> {
        normal: Normal [4, 8],
        label: PointLabel [9],
    }

    pub struct Point3Range<f32, U5> {
        range: PointRange [4],
    }

    pub struct Point3V<f32, U8> {
        viewpoint: PointViewpoint [4],
    }

    #[non_point]
    pub struct Normal3<f32, U4> {
        normal: Normal [0, 3],
    }
}

impl Centroid for Point3Range {
    type Accumulator = Vector4<f32>;

    type Result = Point3;

    fn accumulate(&self, accum: &mut Self::Accumulator) {
        *accum += self.coords();
    }

    fn compute(accum: Self::Accumulator, num: usize) -> Self::Result {
        Point3(accum / (num as f32))
    }
}

impl Centroid for Point3V {
    type Accumulator = Vector4<f32>;

    type Result = Point3;

    fn accumulate(&self, accum: &mut Self::Accumulator) {
        *accum += self.coords();
    }

    fn compute(accum: Self::Accumulator, num: usize) -> Self::Result {
        Point3(accum / (num as f32))
    }
}

impl Data for Normal3 {
    type Data = f32;

    #[inline]
    fn as_slice(&self) -> &[Self::Data] {
        self.0.as_slice()
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Data] {
        self.0.as_mut_slice()
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.normal().iter().all(|x| x.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::vector;

    use super::*;
    #[test]
    fn test_points() {
        let point = Point3RgbaN::default()
            .with_coords(Vector4::new(4., 3., 2., 1.))
            .with_normal(Vector4::new(-1., -2., -3., 0.))
            .with_curvature(-0.5)
            .with_rgba(0xFF000000);

        assert_eq!(
            SVector::from(point),
            vector![
                4.,
                3.,
                2.,
                1.,
                -1.,
                -2.,
                -3.,
                0.,
                -0.5,
                f32::from_bits(0xFF000000)
            ]
        );
    }
}
