mod transforms;

use std::{
    fmt::Debug,
    ops::{Deref, Index, IndexMut},
};

use nalgebra::{
    ClosedAdd, ClosedMul, ClosedSub, Matrix3, SVector, Scalar, SimdComplexField, SimdValue, Vector4,
};
use num::Float;

use self::transforms::Transform;
use crate::points::Point3Infoed;

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

impl<T> Deref for PointCloud<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.storage
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
    /// Note: The result of this function is not normalized (descaled by the
    /// calculated point count); if wanted, use `cov_matrix_norm` instead.
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
                        (acc, num)
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

impl<
        T: Scalar + Float + ClosedAdd + ClosedSub + Default + SimdComplexField<SimdRealField = T>,
        I,
    > PointCloud<Point3Infoed<T, I>>
{
    #[allow(clippy::type_complexity)]
    pub fn centroid_and_cov_matrix(&self) -> (Option<(Vector4<T>, Matrix3<T>)>, usize) {
        let c = match self.storage.iter().find(|v| v.is_finite()) {
            Some(v) => v.coords,
            None => return (None, 0),
        };

        let accum = |mut acc: SVector<T, 9>, v: &Vector4<T>| {
            let d = v - c;

            acc[0] += d.x * d.x;
            acc[1] += d.x * d.y;
            acc[2] += d.x * d.z;
            acc[3] += d.y * d.y;
            acc[4] += d.y * d.z;
            acc[5] += d.z * d.z;
            acc[6] += d.x;
            acc[7] += d.y;
            acc[8] += d.z;

            acc
        };

        let (acc, num) = if self.bounded {
            self.storage
                .iter()
                .fold((SVector::default(), 0), |(acc, num), v| {
                    (accum(acc, &v.coords), num + 1)
                })
        } else {
            self.storage
                .iter()
                .fold((SVector::default(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (accum(acc, &v.coords), num + 1)
                    } else {
                        (acc, num)
                    }
                })
        };

        if num > 0 {
            let a = acc.unscale(T::from(num).unwrap());
            let centroid = Vector4::from([a[6] + c.x, a[7] + c.y, a[8] + c.z, T::one()]);

            let mut cov_matrix = Matrix3::from([
                [a[0] - a[6] * a[6], a[1] - a[6] * a[7], a[2] - a[6] * a[8]],
                [T::zero(), a[3] - a[7] * a[7], a[4] - a[7] * a[8]],
                [T::zero(), T::zero(), a[5] - a[8] * a[8]],
            ]);
            cov_matrix.m21 = cov_matrix.m12;
            cov_matrix.m31 = cov_matrix.m13;
            cov_matrix.m32 = cov_matrix.m23;

            (Some((centroid, cov_matrix)), num)
        } else {
            (None, num)
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

impl<T: Scalar + Float + ClosedSub, I: Clone> PointCloud<Point3Infoed<T, I>> {
    pub fn demean(&self, centroid: &Vector4<T>, out: &mut Self) {
        out.clone_from(self);

        for point in &mut out.storage {
            point.coords.x -= centroid.x;
            point.coords.y -= centroid.y;
            point.coords.z -= centroid.z;
        }
    }
}

impl<T: Scalar + Float + PartialOrd, I> PointCloud<Point3Infoed<T, I>> {
    pub fn box_select(&self, min: &Vector4<T>, max: &Vector4<T>) -> Vec<usize> {
        let mut ret = Vec::with_capacity(self.storage.len());

        if self.bounded {
            for (i, point) in self.storage.iter().enumerate() {
                let coords = point.coords.xyz();
                if min.xyz() <= coords && coords <= max.xyz() {
                    ret.push(i);
                }
            }
        } else {
            for (i, point) in self.storage.iter().enumerate() {
                if point.is_finite() {
                    let coords = point.coords.xyz();
                    if min.xyz() <= coords && coords <= max.xyz() {
                        ret.push(i);
                    }
                }
            }
        }

        ret
    }
}

impl<T: Scalar + Float + SimdValue<Element = T, SimdBool = bool>, I>
    PointCloud<Point3Infoed<T, I>>
{
    /// Note: Points that are not finite (infinite, NaN, etc) are not considered
    /// into calculations.
    pub fn finite_bound(&self) -> Option<(Vector4<T>, Vector4<T>)> {
        if self.bounded {
            self.storage.iter().fold(None, |acc, v| match acc {
                None => Some((v.coords, v.coords)),
                Some((min, max)) => Some((min.inf(&v.coords), max.sup(&v.coords))),
            })
        } else {
            self.storage.iter().fold(None, |acc, v| {
                if v.is_finite() {
                    match acc {
                        None => Some((v.coords, v.coords)),
                        Some((min, max)) => Some((min.inf(&v.coords), max.sup(&v.coords))),
                    }
                } else {
                    acc
                }
            })
        }
    }
}

impl<T: Scalar + Float + PartialOrd + ClosedSub + SimdComplexField<SimdRealField = T>, I>
    PointCloud<Point3Infoed<T, I>>
{
    pub fn max_distance(&self, pivot: &Vector4<T>) -> Option<(T, Vector4<T>)> {
        let pivot = pivot.xyz();

        if self.bounded {
            self.storage.iter().fold(None, |acc, v| {
                let distance = (v.coords.xyz() - pivot).norm();
                match acc {
                    Some((d, _)) if distance <= d => acc,
                    _ => Some((distance, v.coords)),
                }
            })
        } else {
            self.storage.iter().fold(None, |acc, v| {
                if v.is_finite() {
                    let distance = (v.coords.xyz() - pivot).norm();
                    match acc {
                        Some((d, _)) if distance <= d => acc,
                        _ => Some((distance, v.coords)),
                    }
                } else {
                    acc
                }
            })
        }
    }
}
