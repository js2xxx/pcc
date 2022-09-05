use std::{borrow::Cow, ops::Index};

use nalgebra::{
    convert, one, zero, ComplexField, Matrix3, Matrix3x4, Matrix4, Matrix4x3, RealField, SVector,
    Vector3, Vector4,
};
use num::{FromPrimitive, Zero};

use super::PointCloud;
use crate::point::{Centroid, Data, Point, PointViewpoint};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointCloudRef<'a, P> {
    inner: &'a PointCloud<P>,
    indices: Option<Cow<'a, [usize]>>,
}

impl<'a, P> PointCloudRef<'a, P> {
    #[inline]
    pub fn new(inner: &'a PointCloud<P>, indices: Option<Cow<'a, [usize]>>) -> Self {
        PointCloudRef { inner, indices }
    }

    #[inline]
    pub fn point_cloud(&self) -> &'a PointCloud<P> {
        self.inner
    }

    #[inline]
    pub fn indices(&self) -> Option<&[usize]> {
        self.indices.as_ref().map(|indices| indices.as_ref())
    }

    #[inline]
    pub fn to_owned(&self, width: usize) -> PointCloud<P>
    where
        P: Clone + Data,
    {
        match self.indices() {
            Some(indices) => self.inner.create_sub(indices, width),
            None => {
                let mut new = self.inner.clone();
                new.reinterpret(width);
                new
            }
        }
    }
}

impl<'a, P> Index<usize> for PointCloudRef<'a, P> {
    type Output = P;

    fn index(&self, index: usize) -> &Self::Output {
        match self.indices.as_ref() {
            Some(indices) => &self.inner[indices[index]],
            None => &self.inner[index],
        }
    }
}

impl<'a, P> Index<(usize, usize)> for PointCloudRef<'a, P> {
    type Output = P;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.inner[y * self.inner.width + x]
    }
}

pub trait AsPointCloud<'a, P: 'a> {
    fn inner(&self) -> &'a PointCloud<P>;

    fn as_ref(&self) -> PointCloudRef<'a, P>;

    fn is_bounded(&self) -> bool;

    fn data_len(&self) -> usize;

    type DataIter<'b>: Iterator<Item = &'b P> + Clone
    where
        Self: 'b,
        P: 'b;
    fn data_iter(&self) -> Self::DataIter<'_>;

    fn centroid_coords(&self) -> (Option<Vector4<P::Data>>, usize)
    where
        P: Point,
        <P as Data>::Data: ComplexField,
    {
        let (acc, num) = if self.is_bounded() {
            self.data_iter()
                .fold((Vector4::zeros(), 0), |(acc, num), v| {
                    (acc + v.coords(), num + 1)
                })
        } else {
            self.data_iter()
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

    fn centroid(&self) -> (Option<P::Result>, usize)
    where
        P: Point + Centroid,
        <P as Data>::Data: ComplexField,
        <P as Centroid>::Accumulator: Default,
    {
        let ret = if self.is_bounded() {
            { self.data_iter() }.fold(Centroid::default_builder(), |mut acc, v| {
                acc.accumulate(v);
                acc
            })
        } else {
            { self.data_iter() }.fold(Centroid::default_builder(), |mut acc, v| {
                if v.is_finite() {
                    acc.accumulate(v);
                }
                acc
            })
        };

        let num = ret.num();
        (ret.compute(), num)
    }

    fn cov_matrix(&self, centroid: &Vector4<P::Data>) -> (Option<Matrix3<P::Data>>, usize)
    where
        P: Point,
        <P as Data>::Data: ComplexField,
    {
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

        let (acc, num) = if self.is_bounded() {
            self.data_iter()
                .fold((Matrix3::zeros(), 0), |(acc, num), v| {
                    (accum(acc, v), num + 1)
                })
        } else {
            self.data_iter()
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

    fn cov_matrix_norm(&self, centroid: &Vector4<P::Data>) -> (Option<Matrix3<P::Data>>, usize)
    where
        P: Point,
        <P as Data>::Data: ComplexField,
    {
        match self.cov_matrix(centroid) {
            (Some(ret), num) => (Some(ret / convert::<_, P::Data>(num as f64)), num),
            (None, num) => (None, num),
        }
    }

    #[allow(clippy::type_complexity)]
    fn centroid_and_cov_matrix(&self) -> (Option<(Vector4<P::Data>, Matrix3<P::Data>)>, usize)
    where
        P: Point,
        <P as Data>::Data: ComplexField,
    {
        let c = match self.data_iter().find(|v| v.is_finite()) {
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

        let (acc, num) = if self.is_bounded() {
            self.data_iter()
                .fold((SVector::zeros(), 0), |(acc, num), v| {
                    (accum(acc, v), num + 1)
                })
        } else {
            self.data_iter()
                .fold((SVector::zeros(), 0), |(acc, num), v| {
                    if v.is_finite() {
                        (accum(acc, v), num + 1)
                    } else {
                        (acc, num)
                    }
                })
        };

        if num > 0 {
            let a = acc / convert::<_, P::Data>(num as f64);
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

    #[inline]
    fn proj_matrix(&self) -> (Matrix3x4<P::Data>, P::Data)
    where
        P: Point,
        P::Data: RealField,
    {
        let mut matrix = Matrix3x4::zeros();
        let residual = self.proj_matrix_into(&mut matrix);
        (matrix, residual)
    }

    fn proj_matrix_into(&self, matrix: &mut Matrix3x4<P::Data>) -> P::Data
    where
        P: Point,
        P::Data: RealField,
    {
        matrix.set_zero();

        let mut xxt = Matrix4::zeros();
        let mut xut = Matrix4x3::zeros();
        let iter = self
            .data_iter()
            .enumerate()
            .map(|(index, point)| (self.inner().index(index), point))
            .map(|([x, y], point)| {
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

    fn box_select(&self, min: &Vector4<P::Data>, max: &Vector4<P::Data>) -> Vec<usize>
    where
        P: Point,
        <P as Data>::Data: RealField,
    {
        let mut indices = Vec::with_capacity(self.data_len());

        if self.is_bounded() {
            for (i, point) in self.data_iter().enumerate() {
                let coords = point.coords().xyz();
                if min.xyz() <= coords && coords <= max.xyz() {
                    indices.push(i);
                }
            }
        } else {
            for (i, point) in self.data_iter().enumerate() {
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

    fn finite_bound(&self) -> Option<[Vector4<P::Data>; 2]>
    where
        P: Point,
        <P as Data>::Data: RealField,
    {
        if self.is_bounded() {
            self.data_iter().fold(None, |acc, v| match acc {
                None => Some([v.coords().clone(), v.coords().clone()]),
                Some([min, max]) => Some([min.inf(v.coords()), max.sup(v.coords())]),
            })
        } else {
            self.data_iter().fold(None, |acc, v| {
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

    fn max_distance(&self, pivot: &Vector4<P::Data>) -> Option<(P::Data, Vector4<P::Data>)>
    where
        P: Point,
        <P as Data>::Data: RealField,
    {
        if self.is_bounded() {
            self.data_iter().fold(None, |acc, v| {
                let distance = (v.coords() - pivot).norm();
                match acc {
                    Some((d, c)) if distance <= d => Some((d, c)),
                    _ => Some((distance, v.coords().clone())),
                }
            })
        } else {
            self.data_iter().fold(None, |acc, v| {
                if v.is_finite() {
                    let distance = (v.coords() - pivot).norm();
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

    fn viewpoint_center(&self) -> Option<Vector4<P::Data>>
    where
        P: PointViewpoint,
        P::Data: ComplexField,
    {
        let (acc, num) = if self.is_bounded() {
            self.data_iter()
                .fold((Vector4::zeros(), 0), |(acc, num), v| {
                    (acc + v.viewpoint(), num + 1)
                })
        } else {
            self.data_iter()
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

impl<'a, P> AsPointCloud<'a, P> for PointCloudRef<'a, P>
where
    P: Clone,
{
    #[inline]
    fn inner(&self) -> &'a PointCloud<P> {
        self.inner
    }

    #[inline]
    fn as_ref(&self) -> PointCloudRef<'a, P> {
        self.clone()
    }

    #[inline]
    fn is_bounded(&self) -> bool {
        self.inner.is_bounded()
    }

    #[inline]
    fn data_len(&self) -> usize {
        self.inner.len()
    }

    type DataIter<'b> = impl Iterator<Item = &'b P> + Clone where Self: 'b, P: 'b;

    #[inline]
    fn data_iter(&self) -> Self::DataIter<'_> {
        let indices: &[usize] = match self.indices {
            Some(ref indices) => indices.as_ref(),
            None => &[],
        };

        indices.iter().map(|&index| &self.inner[index])
    }
}

impl<'a, P> AsPointCloud<'a, P> for &'a PointCloud<P> {
    #[inline]
    fn inner(&self) -> &'a PointCloud<P> {
        self
    }

    #[inline]
    fn as_ref(&self) -> PointCloudRef<'a, P> {
        PointCloudRef::new(self, None)
    }

    #[inline]
    fn is_bounded(&self) -> bool {
        self.bounded
    }

    #[inline]
    fn data_len(&self) -> usize {
        self.storage.len()
    }

    type DataIter<'b> = impl Iterator<Item = &'b P> + Clone
    where
        Self: 'b,
        P: 'b;

    #[inline]
    fn data_iter(&self) -> Self::DataIter<'_> {
        self.storage.iter()
    }
}
