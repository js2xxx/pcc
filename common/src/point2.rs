mod info;
#[macro_use]
mod macros;
mod centroid;

use core::fmt::Debug;
use std::{array, collections::HashMap};

use nalgebra::{ComplexField, Const, MatrixSlice, MatrixSliceMut, SVector, Scalar, Vector4};
use num::FromPrimitive;
use static_assertions::const_assert;

pub use self::{
    centroid::{Centroid, CentroidBuilder},
    info::{FieldInfo, PointFields},
};

pub trait Point: Debug + Copy + Clone + PartialEq + PartialOrd + Default {
    type Data: Scalar;
    const DIM: usize;

    fn coords(&self)
        -> MatrixSlice<Self::Data, Const<4>, Const<1>, Const<1>, Const<{ Self::DIM }>>;

    fn coords_mut(
        &mut self,
    ) -> MatrixSliceMut<Self::Data, Const<4>, Const<1>, Const<1>, Const<{ Self::DIM }>>;

    fn as_slice(&self) -> &[Self::Data];

    fn as_mut_slice(&mut self) -> &mut [Self::Data];

    fn with_coords(coords: &Vector4<Self::Data>) -> Self;

    fn na_point(&self) -> nalgebra::Point3<Self::Data>
    where
        Self::Data: ComplexField,
        [(); Self::DIM]:,
    {
        nalgebra::Point3::from_homogeneous(self.coords().into_owned()).unwrap()
    }

    fn is_finite(&self) -> bool
    where
        Self::Data: ComplexField,
        [(); Self::DIM]:,
    {
        self.coords().iter().all(|x| x.is_finite())
    }

    fn fields() -> array::IntoIter<FieldInfo, 3> {
        [
            FieldInfo::single("x", 0),
            FieldInfo::single("y", 1),
            FieldInfo::single("z", 2),
        ]
        .into_iter()
    }
}

/// Lower bits <---[b; g; r; a]: [u8; 4]---- Higher bits
pub trait PointRgba: Point {
    fn rgb_value(&self) -> Self::Data;

    fn set_rgb_value(&mut self, value: Self::Data);

    fn rgba(&self) -> u32;

    fn set_rgba(&mut self, rgba: u32);

    fn fields() -> array::IntoIter<FieldInfo, 1>;

    type CentroidAccumulator = [f32; 4];

    fn centroid_accumulate(&self, accum: &mut [f32; 4]) {
        accum[0] += (self.rgba() & 0xff) as f32;
        accum[1] += ((self.rgba() >> 8) & 0xff) as f32;
        accum[2] += ((self.rgba() >> 16) & 0xff) as f32;
        accum[3] += (self.rgba() >> 24) as f32;
    }

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

pub trait PointNormal: Point {
    fn normal(&self)
        -> MatrixSlice<Self::Data, Const<4>, Const<1>, Const<1>, Const<{ Self::DIM }>>;

    fn normal_mut(
        &mut self,
    ) -> MatrixSliceMut<Self::Data, Const<4>, Const<1>, Const<1>, Const<{ Self::DIM }>>;

    fn curvature(&self) -> Self::Data;

    fn set_curvature(&mut self, curvature: Self::Data);

    fn fields() -> array::IntoIter<FieldInfo, 2>;

    type CentroidAccumulator = (Vector4<Self::Data>, Self::Data);

    fn centroid_accumulate(&self, accum: &mut (Vector4<Self::Data>, Self::Data))
    where
        Self::Data: ComplexField,
        [(); Self::DIM]:,
    {
        accum.0 += self.normal();
        accum.1 += self.curvature();
    }

    fn centroid_compute(&mut self, accum: (Vector4<Self::Data>, Self::Data), num: usize)
    where
        Self::Data: ComplexField,
        [(); Self::DIM]:,
    {
        let num = <Self::Data>::from_usize(num).unwrap();
        self.normal_mut().set_column(0, &(accum.0 / num.clone()));
        self.set_curvature(accum.1 / num)
    }
}

pub trait PointIntensity: Point {
    fn intensity(&self) -> Self::Data;

    fn set_intensity(&mut self, intensity: Self::Data);

    fn fields() -> array::IntoIter<FieldInfo, 1>;

    type CentroidAccumulator = Self::Data;

    fn centroid_accumulate(&self, accum: &mut Self::Data)
    where
        Self::Data: ComplexField,
    {
        *accum += self.intensity();
    }

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

    fn set_range(&mut self, intensity: Self::Data);

    fn fields() -> array::IntoIter<FieldInfo, 1>;
}

pub trait PointLabel: Point {
    fn label(&self) -> u32;

    fn set_label(&mut self, label: u32);

    fn fields() -> array::IntoIter<FieldInfo, 1>;

    type CentroidAccumulator = HashMap<u32, usize>;

    fn centroid_accumulate(&self, accum: &mut HashMap<u32, usize>) {
        if let Err(mut e) = accum.try_insert(self.label(), 1) {
            *e.entry.get_mut() += 1;
        }
    }

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
    fn viewpoint(
        &self,
    ) -> MatrixSlice<Self::Data, Const<4>, Const<1>, Const<1>, Const<{ Self::DIM }>>;

    fn viewpoint_mut(
        &mut self,
    ) -> MatrixSliceMut<Self::Data, Const<4>, Const<1>, Const<1>, Const<{ Self::DIM }>>;

    fn fields() -> array::IntoIter<FieldInfo, 1>;
}

define_points! {
    #[auto_centroid]
    pub struct Point3<f32, 4>;

    #[auto_centroid]
    pub struct Point3Rgba<f32, 5> {
        rgba: PointRgba [4],
    }

    #[auto_centroid]
    pub struct Point3N<f32, 9> {
        normal: PointNormal [4, 8],
    }

    #[auto_centroid]
    pub struct Point3RgbaN<f32, 10> {
        normal: PointNormal [4, 8],
        rgba: PointRgba [9],
    }

    #[auto_centroid]
    pub struct Point3IN<f32, 10> {
        normal: PointNormal [4, 8],
        intensity: PointIntensity [9],
    }

    #[auto_centroid]
    pub struct Point3LN<f32, 10> {
        normal: PointNormal [4, 8],
        label: PointLabel [9],
    }

    pub struct Point3Range<f32, 5> {
        range: PointRange [4],
    }

    pub struct Point3V<f32, 8> {
        viewpoint: PointViewpoint [4],
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
