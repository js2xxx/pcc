mod transforms;

use std::{
    fmt::Debug,
    ops::{Deref, Index, IndexMut},
};

use nalgebra::{ClosedSub, ComplexField, Matrix3, SVector, Scalar, Vector4};

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

    pub fn reinterpret(&mut self, width: usize) {
        assert!(width > 0);
        assert_eq!(self.storage.len() % width, 0);
        self.width = width;
    }
}

impl<T: Clone> PointCloud<T> {
    pub fn try_create_sub(&self, indices: &[usize], width: usize) -> Option<Self> {
        (width > 0 && indices.len() % width == 0).then(|| PointCloud {
            storage: { indices.iter() }
                .map(|&index| self.storage[index].clone())
                .collect(),
            width,
            bounded: self.bounded,
        })
    }

    pub fn create_sub(&self, indices: &[usize], width: usize) -> Self {
        self.try_create_sub(indices, width)
            .expect("The length of the vector must be divisible by width")
    }
}

impl<T> Deref for PointCloud<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

impl<T> Index<usize> for PointCloud<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.storage[index]
    }
}

impl<T> IndexMut<usize> for PointCloud<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.storage[index]
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

impl<T: Scalar + ComplexField<RealField = T>, I> PointCloud<Point3Infoed<T, I>> {
    pub fn try_from_vec(
        storage: Vec<Point3Infoed<T, I>>,
        width: usize,
    ) -> Result<Self, Vec<Point3Infoed<T, I>>> {
        if width > 0 && storage.len() % width == 0 {
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

impl<T: Scalar + ComplexField<RealField = T>, I: Debug> PointCloud<Point3Infoed<T, I>> {
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

impl<T: Scalar + ComplexField<RealField = T> + Default, I> PointCloud<Point3Infoed<T, I>> {
    pub fn centroid(&self) -> (Option<Vector4<T>>, usize) {
        let (acc, num) = if self.bounded {
            self.storage
                .iter()
                .fold((Vector4::default(), 0), |(acc, num), v| {
                    (acc + &v.coords, num + 1)
                })
        } else {
            self.storage
                .iter()
                .fold((Vector4::default(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (acc + &v.coords, num + 1)
                    } else {
                        (acc, num)
                    }
                })
        };

        if num > 0 {
            let mut ret = acc.unscale(T::from_usize(num).unwrap());
            ret.w = T::one();
            (Some(ret), num)
        } else {
            (None, num)
        }
    }
}

impl<T: Scalar + ComplexField<RealField = T> + Default, I> PointCloud<Point3Infoed<T, I>> {
    /// Note: The result of this function is not normalized (descaled by the
    /// calculated point count); if wanted, use `cov_matrix_norm` instead.
    pub fn cov_matrix(&self, centroid: &Vector4<T>) -> (Option<Matrix3<T>>, usize) {
        let accum = |mut acc: Matrix3<T>, v: &Vector4<T>| {
            let d = v - centroid;

            acc.m22 += d.y.clone() * d.y.clone();
            acc.m23 += d.y.clone() * d.z.clone();
            acc.m33 += d.z.clone() * d.z.clone();
            let d = d.scale(d.x.clone());
            acc.m11 += d.x.clone();
            acc.m12 += d.y.clone();
            acc.m13 += d.z.clone();

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
            ret.m21 = ret.m12.clone();
            ret.m31 = ret.m13.clone();
            ret.m32 = ret.m23.clone();
            (Some(ret), num)
        } else {
            (None, num)
        }
    }

    pub fn cov_matrix_norm(&self, centroid: &Vector4<T>) -> (Option<Matrix3<T>>, usize) {
        match self.cov_matrix(centroid) {
            (Some(ret), num) => (Some(ret.unscale(T::from_usize(num).unwrap())), num),
            (None, num) => (None, num),
        }
    }
}

impl<T: Scalar + Default + ComplexField<RealField = T>, I>
    PointCloud<Point3Infoed<T, I>>
{
    #[allow(clippy::type_complexity)]
    pub fn centroid_and_cov_matrix(&self) -> (Option<(Vector4<T>, Matrix3<T>)>, usize) {
        let c = match self.storage.iter().find(|v| v.is_finite()) {
            Some(v) => v.coords.clone(),
            None => return (None, 0),
        };

        let accum = |mut acc: SVector<T, 9>, v: &Vector4<T>| {
            let d = v - &c;

            acc[0] += d.x.clone() * d.x.clone();
            acc[1] += d.x.clone() * d.y.clone();
            acc[2] += d.x.clone() * d.z.clone();
            acc[3] += d.y.clone() * d.y.clone();
            acc[4] += d.y.clone() * d.z.clone();
            acc[5] += d.z.clone() * d.z.clone();
            acc[6] += d.x.clone();
            acc[7] += d.y.clone();
            acc[8] += d.z.clone();

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
            let a = acc.unscale(T::from_usize(num).unwrap());
            let centroid = Vector4::from([
                a[6].clone() + c.x.clone(),
                a[7].clone() + c.y.clone(),
                a[8].clone() + c.z.clone(),
                T::one(),
            ]);

            let mut cov_matrix = Matrix3::from([
                [
                    a[0].clone() - a[6].clone() * a[6].clone(),
                    a[1].clone() - a[6].clone() * a[7].clone(),
                    a[2].clone() - a[6].clone() * a[8].clone(),
                ],
                [
                    T::zero(),
                    a[3].clone() - a[7].clone() * a[7].clone(),
                    a[4].clone() - a[7].clone() * a[8].clone(),
                ],
                [
                    T::zero(),
                    T::zero(),
                    a[5].clone() - a[8].clone() * a[8].clone(),
                ],
            ]);
            cov_matrix.m21 = cov_matrix.m12.clone();
            cov_matrix.m31 = cov_matrix.m13.clone();
            cov_matrix.m32 = cov_matrix.m23.clone();

            (Some((centroid, cov_matrix)), num)
        } else {
            (None, num)
        }
    }
}

impl<T: Scalar + ComplexField<RealField = T> + Default, I: Clone + Default>
    PointCloud<Point3Infoed<T, I>>
{
    pub fn transform<Z: Transform<T>>(&self, z: &Z, out: &mut Self) {
        out.storage
            .resize_with(self.storage.len(), Default::default);

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

impl<T: Scalar + ClosedSub, I: Clone> PointCloud<Point3Infoed<T, I>> {
    pub fn demean(&self, centroid: &Vector4<T>, out: &mut Self) {
        out.clone_from(self);

        for point in &mut out.storage {
            point.coords.x -= centroid.x.clone();
            point.coords.y -= centroid.y.clone();
            point.coords.z -= centroid.z.clone();
        }
    }
}

impl<T: Scalar + ComplexField<RealField = T> + PartialOrd, I> PointCloud<Point3Infoed<T, I>> {
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

impl<T: Scalar + ComplexField<RealField = T> + PartialOrd, I> PointCloud<Point3Infoed<T, I>> {
    /// Note: Points that are not finite (infinite, NaN, etc) are not considered
    /// into calculations.
    pub fn finite_bound(&self) -> Option<(Vector4<T>, Vector4<T>)> {
        if self.bounded {
            self.storage.iter().fold(None, |acc, v| match acc {
                None => Some((v.coords.clone(), v.coords.clone())),
                Some((min, max)) => Some((min.inf(&v.coords), max.sup(&v.coords))),
            })
        } else {
            self.storage.iter().fold(None, |acc, v| {
                if v.is_finite() {
                    match acc {
                        None => Some((v.coords.clone(), v.coords.clone())),
                        Some((min, max)) => Some((min.inf(&v.coords), max.sup(&v.coords))),
                    }
                } else {
                    acc
                }
            })
        }
    }
}

impl<T: Scalar + ComplexField<RealField = T> + PartialOrd, I> PointCloud<Point3Infoed<T, I>> {
    pub fn max_distance(&self, pivot: &Vector4<T>) -> Option<(T, Vector4<T>)> {
        let pivot = pivot.xyz();

        if self.bounded {
            self.storage.iter().fold(None, |acc, v| {
                let distance = (v.coords.xyz() - &pivot).norm();
                match acc {
                    Some((d, c)) if distance <= d => Some((d, c)),
                    _ => Some((distance, v.coords.clone())),
                }
            })
        } else {
            self.storage.iter().fold(None, |acc, v| {
                if v.is_finite() {
                    let distance = (v.coords.xyz() - &pivot).norm();
                    match acc {
                        Some((d, c)) if distance <= d => Some((d, c)),
                        _ => Some((distance, v.coords.clone())),
                    }
                } else {
                    acc
                }
            })
        }
    }
}
