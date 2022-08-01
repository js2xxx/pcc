mod transforms;

use std::fmt::Debug;
use std::ops::{Index, IndexMut};

use nalgebra::{ClosedAdd, ClosedMul, ClosedSub, Matrix3, Scalar, SimdComplexField, Vector4};
use num::Float;

use crate::points::Point3Infoed;

use self::transforms::Transform;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointCloud<T> {
    storage: Vec<T>,
    width: usize,
    bounded: bool,
}

impl<T> PointCloud<T> {
    pub fn width(&self) -> usize {
        assert_eq!(self.storage.len() % self.width, 0);
        self.width
    }

    pub fn height(&self) -> usize {
        assert_eq!(self.storage.len() % self.width, 0);
        self.storage.len() / self.width
    }

    pub fn is_bounded(&self) -> bool {
        self.bounded
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage
    }
}

impl<T> Index<(usize, usize)> for PointCloud<T> {
    type Output = T;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.storage[y * self.width + x]
    }
}

impl<T> IndexMut<(usize, usize)> for PointCloud<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.storage[y * self.width + x]
    }
}

impl<T> PointCloud<T> {
    pub fn new() -> Self {
        PointCloud {
            storage: Vec::new(),
            width: 0,
            bounded: true,
        }
    }
}

impl<T: Scalar + Float, I> PointCloud<Point3Infoed<T, I>> {
    pub fn try_from_vec(
        storage: Vec<Point3Infoed<T, I>>,
        width: usize,
    ) -> Result<Self, Vec<Point3Infoed<T, I>>> {
        if storage.len() % width == 0 {
            let bounded = storage.iter().all(|p| p.is_finite());
            Ok(PointCloud {
                storage,
                width,
                bounded,
            })
        } else {
            Err(storage)
        }
    }
}

impl<T: Scalar + Float, I: Debug> PointCloud<Point3Infoed<T, I>> {
    pub fn from_vec(storage: Vec<Point3Infoed<T, I>>, width: usize) -> Self {
        PointCloud::try_from_vec(storage, width)
            .expect("The length of the vector must be divisible by width")
    }
}

impl<T: Scalar, I> Default for PointCloud<Point3Infoed<T, I>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float + ClosedAdd + Default + SimdComplexField<SimdRealField = T>, I>
    PointCloud<Point3Infoed<T, I>>
{
    pub fn centroid(&self) -> (Option<Vector4<T>>, usize) {
        let (acc, num) = if self.bounded {
            self.storage
                .iter()
                .fold((Vector4::default(), 0), |(acc, num), v| {
                    (acc + v.coords, num + 1)
                })
        } else {
            self.storage
                .iter()
                .fold((Vector4::default(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (acc + v.coords, num + 1)
                    } else {
                        (acc, num)
                    }
                })
        };

        if num > 0 {
            let mut ret = acc.unscale(T::from(num).unwrap());
            ret.w = T::one();
            (Some(ret), num)
        } else {
            (None, num)
        }
    }
}

impl<
        T: Scalar + Float + ClosedAdd + ClosedSub + Default + SimdComplexField<SimdRealField = T>,
        I,
    > PointCloud<Point3Infoed<T, I>>
{
    pub fn cov_matrix(&self, centroid: &Vector4<T>) -> (Option<Matrix3<T>>, usize) {
        let accum = |mut acc: Matrix3<T>, v: &Vector4<T>| {
            let d = v - centroid;

            acc.m22 += d.y * d.y;
            acc.m23 += d.y * d.z;
            acc.m33 += d.z * d.z;
            let d = d.scale(d.x);
            acc.m11 += d.x;
            acc.m12 += d.y;
            acc.m13 += d.z;

            acc
        };

        let (acc, num) = if self.bounded {
            self.storage
                .iter()
                .fold((Matrix3::default(), 0), |(acc, num), v| {
                    (accum(acc, &v.coords), num + 1)
                })
        } else {
            self.storage
                .iter()
                .fold((Matrix3::default(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (accum(acc, &v.coords), num + 1)
                    } else {
                        (Matrix3::zeros(), num)
                    }
                })
        };

        if num > 0 {
            let mut ret = acc;
            ret.m21 = ret.m12;
            ret.m31 = ret.m13;
            ret.m32 = ret.m23;
            (Some(ret), num)
        } else {
            (None, num)
        }
    }

    pub fn cov_matrix_norm(&self, centroid: &Vector4<T>) -> (Option<Matrix3<T>>, usize) {
        match self.cov_matrix(centroid) {
            (Some(ret), num) => (Some(ret.unscale(T::from(num).unwrap())), num),
            (None, num) => (None, num),
        }
    }
}

impl<T: Scalar + Float + Copy + ClosedAdd + ClosedMul + Default, I: Clone + Default>
    PointCloud<Point3Infoed<T, I>>
{
    pub fn transform<Z: Transform<T>>(&self, z: &Z, out: &mut Self) {
        out.storage.resize(self.storage.len(), Default::default());

        out.width = self.width;
        out.bounded = self.bounded;

        if self.bounded {
            for (from, to) in self.storage.iter().zip(out.storage.iter_mut()) {
                z.se3(&from.coords, &mut to.coords)
            }
        } else {
            for (from, to) in self.storage.iter().zip(out.storage.iter_mut()) {
                if !from.is_finite() {
                    continue;
                }
                z.se3(&from.coords, &mut to.coords)
            }
        }
    }
}
