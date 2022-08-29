mod transforms;

use std::{
    fmt::Debug,
    ops::{Deref, Index, IndexMut},
};

use nalgebra::{
    ComplexField, Matrix3, Matrix3x4, Matrix4, Matrix4x3, RealField, SVector, Vector3, Vector4,
};
use num::{one, zero, FromPrimitive, Zero};

use self::transforms::Transform;
use crate::point::{Centroid, Data, Point, PointViewpoint};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointCloud<P> {
    storage: Vec<P>,
    width: usize,
    bounded: bool,
}

impl<P> PointCloud<P> {
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

    pub fn into_vec(self) -> Vec<P> {
        self.storage
    }

    /// # Safety
    ///
    /// The width and boundedness of the point cloud must be valid.
    pub unsafe fn storage(&mut self) -> &mut Vec<P> {
        &mut self.storage
    }
}

impl<P: Clone> PointCloud<P> {
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

impl<P> Deref for PointCloud<P> {
    type Target = [P];

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

impl<P> Index<usize> for PointCloud<P> {
    type Output = P;

    fn index(&self, index: usize) -> &Self::Output {
        &self.storage[index]
    }
}

impl<P> IndexMut<usize> for PointCloud<P> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.storage[index]
    }
}

impl<P> Index<(usize, usize)> for PointCloud<P> {
    type Output = P;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.storage[y * self.width + x]
    }
}

impl<P> IndexMut<(usize, usize)> for PointCloud<P> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.storage[y * self.width + x]
    }
}

impl<P> PointCloud<P> {
    pub fn new() -> Self {
        PointCloud {
            storage: Vec::new(),
            width: 0,
            bounded: true,
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the points in the cloud are all `bounded`
    /// and the length of `storage` is divisible by `width`.
    pub unsafe fn from_raw_parts(storage: Vec<P>, width: usize, bounded: bool) -> Self {
        PointCloud {
            storage,
            width,
            bounded,
        }
    }
}

impl<P: Clone> PointCloud<P> {
    pub fn transpose(&self) -> Self {
        let mut other = Self::new();
        self.transpose_into(&mut other);
        other
    }

    pub fn transpose_into(&self, other: &mut Self) {
        other.storage.clear();
        other.storage.reserve(self.storage.len());

        let width = self.width;
        unsafe {
            let space = other.storage.spare_capacity_mut();

            for (index, obj) in self.storage.iter().enumerate() {
                let (x, y) = (index % width, index / width);
                space[x * width + y].write(obj.clone());
            }

            other.storage.set_len(other.storage.capacity());
        }

        other.width = self.height();
        other.bounded = self.bounded;
    }
}

impl<P: Data> PointCloud<P> {
    pub fn try_from_vec(storage: Vec<P>, width: usize) -> Result<Self, Vec<P>> {
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

    pub fn reinterpret(&mut self, width: usize) {
        assert!(width > 0);
        assert_eq!(self.storage.len() % width, 0);
        self.width = width;
        self.bounded = self.storage.iter().all(|p| p.is_finite());
    }

    pub fn from_vec(storage: Vec<P>, width: usize) -> Self {
        PointCloud::try_from_vec(storage, width)
            .expect("The length of the vector must be divisible by width")
    }
}

impl<P> Default for PointCloud<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: ComplexField,
{
    pub fn centroid_coords(&self) -> (Option<Vector4<P::Data>>, usize) {
        let (acc, num) = if self.bounded {
            self.storage
                .iter()
                .fold((Vector4::zeros(), 0), |(acc, num), v| {
                    (acc + v.coords(), num + 1)
                })
        } else {
            self.storage
                .iter()
                .fold((Vector4::zeros(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (acc + v.coords(), num + 1)
                    } else {
                        (acc, num)
                    }
                })
        };

        let ret = (num > 0).then(|| {
            let mut ret = acc / <P::Data>::from_usize(num).unwrap();
            ret.w = one();
            ret
        });
        (ret, num)
    }
}

impl<P: Point + Centroid> PointCloud<P>
where
    <P as Data>::Data: ComplexField,
    <P as Centroid>::Accumulator: Default,
{
    pub fn centroid(&self) -> (Option<P::Result>, usize) {
        let ret = if self.bounded {
            { self.storage.iter() }.fold(Centroid::default_builder(), |mut acc, v| {
                acc.accumulate(v);
                acc
            })
        } else {
            { self.storage.iter() }.fold(Centroid::default_builder(), |mut acc, v| {
                if v.is_finite() {
                    acc.accumulate(v);
                }
                acc
            })
        };

        let num = ret.num();
        (ret.compute(), num)
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: ComplexField,
{
    /// Note: The result of this function is not normalized (descaled by the
    /// calculated point count); if wanted, use `cov_matrix_norm` instead.
    pub fn cov_matrix(&self, centroid: &Vector4<P::Data>) -> (Option<Matrix3<P::Data>>, usize) {
        let accum = |mut acc: Matrix3<P::Data>, v: &P| {
            let d = v.coords() - centroid;

            acc.m22 += d.y.clone() * d.y.clone();
            acc.m23 += d.y.clone() * d.z.clone();
            acc.m33 += d.z.clone() * d.z.clone();
            let d = &d * d.x.clone();
            acc.m11 += d.x.clone();
            acc.m12 += d.y.clone();
            acc.m13 += d.z.clone();

            acc
        };

        let (acc, num) = if self.bounded {
            self.storage
                .iter()
                .fold((Matrix3::zeros(), 0), |(acc, num), v| {
                    (accum(acc, v), num + 1)
                })
        } else {
            self.storage
                .iter()
                .fold((Matrix3::zeros(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (accum(acc, v), num + 1)
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

    pub fn cov_matrix_norm(
        &self,
        centroid: &Vector4<P::Data>,
    ) -> (Option<Matrix3<P::Data>>, usize) {
        match self.cov_matrix(centroid) {
            (Some(ret), num) => (Some(ret / P::Data::from_usize(num).unwrap()), num),
            (None, num) => (None, num),
        }
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: ComplexField,
{
    #[allow(clippy::type_complexity)]
    pub fn centroid_and_cov_matrix(&self) -> (Option<(Vector4<P::Data>, Matrix3<P::Data>)>, usize) {
        let c = match self.storage.iter().find(|v| v.is_finite()) {
            Some(v) => v.coords().clone(),
            None => return (None, 0),
        };

        let accum = |mut acc: SVector<P::Data, 9>, v: &P| {
            let d = v.coords() - &c;

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
                .fold((SVector::zeros(), 0), |(acc, num), v| {
                    (accum(acc, v), num + 1)
                })
        } else {
            self.storage
                .iter()
                .fold((SVector::zeros(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (accum(acc, v), num + 1)
                    } else {
                        (acc, num)
                    }
                })
        };

        if num > 0 {
            let a = acc / P::Data::from_usize(num).unwrap();
            let centroid = Vector4::from([
                a[6].clone() + c.x.clone(),
                a[7].clone() + c.y.clone(),
                a[8].clone() + c.z.clone(),
                one(),
            ]);

            let mut cov_matrix = Matrix3::from([
                [
                    a[0].clone() - a[6].clone() * a[6].clone(),
                    a[1].clone() - a[6].clone() * a[7].clone(),
                    a[2].clone() - a[6].clone() * a[8].clone(),
                ],
                [
                    zero(),
                    a[3].clone() - a[7].clone() * a[7].clone(),
                    a[4].clone() - a[7].clone() * a[8].clone(),
                ],
                [zero(), zero(), a[5].clone() - a[8].clone() * a[8].clone()],
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

impl<P: Point> PointCloud<P>
where
    P::Data: RealField,
{
    #[inline]
    pub fn proj_matrix(&self) -> (Matrix3x4<P::Data>, P::Data) {
        let mut matrix = Matrix3x4::zeros();
        let residual = self.proj_matrix_into(&mut matrix);
        (matrix, residual)
    }

    pub fn proj_matrix_into(&self, matrix: &mut Matrix3x4<P::Data>) -> P::Data {
        matrix.set_zero();

        let mut xxt = Matrix4::zeros();
        let mut xut = Matrix4x3::zeros();
        let iter = self
            .storage
            .iter()
            .enumerate()
            .map(|(index, point)| (index % self.width, index / self.width, point))
            .map(|(x, y, point)| {
                (
                    Vector3::new(
                        P::Data::from_usize(x).unwrap(),
                        P::Data::from_usize(y).unwrap(),
                        one(),
                    ),
                    point.coords(),
                )
            });
        for (u, x) in iter.clone() {
            xxt.syger(one(), x, x, one());
            xut.ger(one(), x, &u, one());
        }
        let cholesky = xxt.cholesky().unwrap();
        for mut column in xut.column_iter_mut() {
            cholesky.solve_mut(&mut column);
        }
        xut.transpose_to(matrix);

        let mut j = Matrix3::zeros();
        for (u, x) in iter.clone() {
            let axmu = &*matrix * x - &u;
            j.syger(one(), &axmu, &axmu, one());
        }
        let [[residual, ..]] = j.symmetric_eigenvalues().data.0;
        residual
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: ComplexField,
{
    pub fn transform<Z: Transform<P::Data>>(&self, z: &Z, out: &mut Self) {
        out.storage
            .resize_with(self.storage.len(), Default::default);

        out.width = self.width;
        out.bounded = self.bounded;

        if self.bounded {
            for (from, to) in self.storage.iter().zip(out.storage.iter_mut()) {
                z.se3(from.coords(), to.coords_mut())
            }
        } else {
            for (from, to) in self.storage.iter().zip(out.storage.iter_mut()) {
                if !from.is_finite() {
                    continue;
                }
                z.se3(from.coords(), to.coords_mut())
            }
        }
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: ComplexField,
{
    pub fn demean(&self, centroid: &Vector4<P::Data>, out: &mut Self) {
        out.clone_from(self);

        for point in &mut out.storage {
            point.coords_mut().x -= centroid.x.clone();
            point.coords_mut().y -= centroid.y.clone();
            point.coords_mut().z -= centroid.z.clone();
        }
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: RealField,
{
    pub fn box_select(&self, min: &Vector4<P::Data>, max: &Vector4<P::Data>) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.storage.len());

        if self.bounded {
            for (i, point) in self.storage.iter().enumerate() {
                let coords = point.coords().xyz();
                if min.xyz() <= coords && coords <= max.xyz() {
                    indices.push(i);
                }
            }
        } else {
            for (i, point) in self.storage.iter().enumerate() {
                if point.is_finite() {
                    let coords = point.coords().xyz();
                    if min.xyz() <= coords && coords <= max.xyz() {
                        indices.push(i);
                    }
                }
            }
        }

        indices
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: RealField,
{
    /// Note: Points that are not finite (infinite, NaN, etc) are not considered
    /// into calculations.
    pub fn finite_bound(&self) -> Option<[Vector4<P::Data>; 2]> {
        if self.bounded {
            self.storage.iter().fold(None, |acc, v| match acc {
                None => Some([v.coords().clone(), v.coords().clone()]),
                Some([min, max]) => Some([min.inf(v.coords()), max.sup(v.coords())]),
            })
        } else {
            self.storage.iter().fold(None, |acc, v| {
                if v.is_finite() {
                    match acc {
                        None => Some([v.coords().clone(), v.coords().clone()]),
                        Some([min, max]) => Some([min.inf(v.coords()), max.sup(v.coords())]),
                    }
                } else {
                    acc
                }
            })
        }
    }
}

impl<P: Point> PointCloud<P>
where
    <P as Data>::Data: RealField,
{
    pub fn max_distance(&self, pivot: &Vector4<P::Data>) -> Option<(P::Data, Vector4<P::Data>)> {
        let pivot = pivot.xyz();

        if self.bounded {
            self.storage.iter().fold(None, |acc, v| {
                let distance = (v.coords().xyz() - &pivot).norm();
                match acc {
                    Some((d, c)) if distance <= d => Some((d, c)),
                    _ => Some((distance, v.coords().clone())),
                }
            })
        } else {
            self.storage.iter().fold(None, |acc, v| {
                if v.is_finite() {
                    let distance = (v.coords().xyz() - &pivot).norm();
                    match acc {
                        Some((d, c)) if distance <= d => Some((d, c)),
                        _ => Some((distance, v.coords().clone())),
                    }
                } else {
                    acc
                }
            })
        }
    }
}

impl<P: PointViewpoint> PointCloud<P>
where
    P::Data: ComplexField,
{
    pub fn viewpoint_center(&self) -> Option<Vector4<P::Data>> {
        let (acc, num) = if self.bounded {
            self.storage
                .iter()
                .fold((Vector4::zeros(), 0), |(acc, num), v| {
                    (acc + v.viewpoint(), num + 1)
                })
        } else {
            self.storage
                .iter()
                .fold((Vector4::zeros(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (acc + v.viewpoint(), num + 1)
                    } else {
                        (acc, num)
                    }
                })
        };

        (num > 0).then(|| {
            let mut ret = acc / <P::Data>::from_usize(num).unwrap();
            ret.w = one();
            ret
        })
    }
}
