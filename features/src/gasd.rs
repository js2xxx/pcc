use nalgebra::{
    convert, DVector, IsometryMatrix3, Matrix3, RealField, Rotation3, SVector, Scalar,
    Translation3, Vector2, Vector3, Vector4,
};
use num::ToPrimitive;
use pcc_common::{
    feature::Feature,
    point::{Point, PointRgba},
    point_cloud::{AsPointCloud, PointCloud},
    Interpolation,
};

use crate::HIST_MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GasdOutput<P>
where
    P: Point,
    P::Data: RealField,
{
    pub transformed: PointCloud<P>,
    pub transform: IsometryMatrix3<P::Data>,
    pub histogram: Vec<DVector<P::Data>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GasdData {
    pub half_grid_size: usize,
    pub hist_size: usize,
    pub interp: Interpolation,
}

impl GasdData {
    fn accum_hist<T: RealField>(
        &self,
        pivot: &Vector4<T>,
        max_coord: T,
        bin: T,
        inc: T,
        hist: &mut [DVector<T>],
    ) where
        T: RealField + ToPrimitive,
    {
        let grid_size = self.half_grid_size * 2;
        let half_grid_size = convert::<_, T>(self.half_grid_size as f64);
        let scaled = { pivot.xyz() }.map(|x| x / max_coord.clone() * half_grid_size.clone());
        let mut coords = scaled
            .map(|x| x + half_grid_size.clone())
            .insert_row(3, bin);
        if self.interp != Interpolation::None {
            coords.apply(|x| *x -= convert(0.5));
        }
        let bins = coords.map(|x| x.floor().to_usize().unwrap());
        let gi = ((bins.x + 1) * (grid_size + 2) + bins.y + 1) * (grid_size + 2) + bins.z + 1;
        let hi = bins.w + 1;
        if self.interp == Interpolation::None {
            hist[gi][hi] += inc
        } else {
            let [[x, y, z]] = (pivot.xyz() - bins.xyz().map(|x| convert(x as f64))).data.0;

            let d1 = inc.clone() * x;
            let d0 = inc - d1.clone();
            let d = Vector2::new(d1, d0);

            let d1 = &d * y;
            let d0 = &d - &d1;
            let d = Vector4::new(d1[0].clone(), d0[0].clone(), d1[1].clone(), d0[1].clone());

            let d1 = &d * z;
            let d0 = &d - &d1;
            let d = SVector::<_, 8>::from([
                d1[0].clone(),
                d0[0].clone(),
                d1[1].clone(),
                d0[1].clone(),
                d1[2].clone(),
                d0[2].clone(),
                d1[3].clone(),
                d0[3].clone(),
            ]);

            if self.interp == Interpolation::Trilinear {
                hist[gi][hi] += d[7].clone();
                hist[gi + 1][hi] += d[6].clone();
                hist[gi + (grid_size + 2)][hi] += d[5].clone();
                hist[gi + (grid_size + 3)][hi] += d[4].clone();
                hist[gi + (grid_size + 2) * (grid_size + 2)][hi] += d[3].clone();
                hist[gi + (grid_size + 2) * (grid_size + 2) + 1][hi] += d[2].clone();
                hist[gi + (grid_size + 3) * (grid_size + 2)][hi] += d[1].clone();
                hist[gi + (grid_size + 3) * (grid_size + 2) + 1][hi] += d[0].clone();
            } else {
                let d1 = d.scale(convert(bins[3] as f64));
                let d0 = &d - &d1;

                hist[gi][hi] += d1[7].clone();
                hist[gi][hi + 1] += d0[7].clone();
                hist[gi + 1][hi] += d1[6].clone();
                hist[gi + 1][hi + 1] += d0[6].clone();
                hist[gi + (grid_size + 2)][hi] += d1[5].clone();
                hist[gi + (grid_size + 2)][hi + 1] += d0[5].clone();
                hist[gi + (grid_size + 3)][hi] += d1[4].clone();
                hist[gi + (grid_size + 3)][hi + 1] += d0[4].clone();
                hist[gi + (grid_size + 2) * (grid_size + 2)][hi] += d1[3].clone();
                hist[gi + (grid_size + 2) * (grid_size + 2)][hi + 1] += d0[3].clone();
                hist[gi + (grid_size + 2) * (grid_size + 2) + 1][hi] += d1[2].clone();
                hist[gi + (grid_size + 2) * (grid_size + 2) + 1][hi + 1] += d0[2].clone();
                hist[gi + (grid_size + 3) * (grid_size + 2)][hi] += d1[1].clone();
                hist[gi + (grid_size + 3) * (grid_size + 2)][hi + 1] += d0[1].clone();
                hist[gi + (grid_size + 3) * (grid_size + 2) + 1][hi] += d1[0].clone();
                hist[gi + (grid_size + 3) * (grid_size + 2) + 1][hi + 1] += d0[0].clone();
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GasdEstimation<T: Scalar> {
    pub view_direction: Vector3<T>,
    pub data: GasdData,
}

impl<T: Scalar> GasdEstimation<T> {
    #[inline]
    pub fn new(view_direction: Vector3<T>, data: GasdData) -> Self {
        GasdEstimation {
            view_direction,
            data,
        }
    }

    pub fn get_transform<P>(
        view_direction: &Vector3<T>,
        input: &PointCloud<P>,
    ) -> Option<IsometryMatrix3<P::Data>>
    where
        T: RealField,
        P: Point<Data = T>,
    {
        let centroid = input.centroid_coords().0?;
        let se =
            pcc_common::cov_matrix(input.iter().map(|point| point.coords()))?.symmetric_eigen();
        let z = {
            let vec = se.eigenvectors.column(se.eigenvalues.imin());
            if vec.dot(view_direction) > T::zero() {
                -vec
            } else {
                vec.into_owned()
            }
        };
        let x = se.eigenvectors.column(se.eigenvalues.imax());
        let xyz = {
            let [y] = z.cross(&x).data.0;
            let [x] = x.into_owned().data.0;
            let [z] = z.data.0;
            Matrix3::from([x, y, z])
        };

        Some(Rotation3::from_matrix_unchecked(xyz) * Translation3::from(-centroid.xyz()))
    }
}

impl<'a, T, P> Feature<&'a PointCloud<P>, Option<GasdOutput<P>>, (), ()> for GasdEstimation<T>
where
    T: RealField + ToPrimitive,
    P: Point<Data = T>,
{
    fn compute(&self, input: &'a PointCloud<P>, _: (), _: ()) -> Option<GasdOutput<P>> {
        let transform = Self::get_transform(&self.view_direction, input)?;
        let transformed =
            input.map(|point| point.clone().with_na_point(&transform * point.na_point()));

        let grid_size = self.data.half_grid_size * 2;

        let centroid = Vector3::zeros().insert_row(3, T::one());
        let (_, far) = transformed.max_distance(&centroid)?;
        let factor = (centroid - far).norm();

        let [min, max] = transformed.finite_bound()?;
        let max_coord = min.xyz().abs().max().max(max.xyz().abs().max());
        let inc = convert::<_, T>(HIST_MAX) / convert((transformed.len() - 1) as f64);

        let mut histogram = vec![
            DVector::zeros(self.data.hist_size + 2);
            (grid_size + 2) * (grid_size + 2) * (grid_size + 2)
        ];

        let iter = transformed.iter().map(|point| {
            let distance = point.coords().xyz().norm();
            let step = factor.clone() / convert(self.data.half_grid_size as f64);
            let value = (distance / step).fract();
            (point.coords(), value * convert(self.data.hist_size as f64))
        });

        iter.for_each(|(pivot, bin)| {
            self.data
                .accum_hist(pivot, max_coord.clone(), bin, inc.clone(), &mut histogram)
        });

        Some(GasdOutput {
            transform,
            transformed,
            histogram,
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GasdColorEstimation<T: Scalar> {
    pub view_direction: Vector3<T>,
    pub data: GasdData,
    pub color: GasdData,
}

impl<T: Scalar> GasdColorEstimation<T> {
    pub fn new(view_direction: Vector3<T>, data: GasdData, color: GasdData) -> Self {
        GasdColorEstimation {
            view_direction,
            data,
            color,
        }
    }
}

impl<'a, T, P> Feature<&'a PointCloud<P>, Option<GasdOutput<P>>, (), ()> for GasdColorEstimation<T>
where
    T: RealField + ToPrimitive,
    P: PointRgba<Data = T>,
{
    fn compute(&self, input: &'a PointCloud<P>, _: (), _: ()) -> Option<GasdOutput<P>> {
        let new = GasdEstimation::new(self.view_direction.clone(), self.data);
        let mut output = new.compute(input, (), ())?;

        let [min, max] = output.transformed.finite_bound()?;
        let max_coord = min.xyz().abs().max().max(max.xyz().abs().max());
        let inc = convert::<_, T>(HIST_MAX) / convert((output.transformed.len() - 1) as f64);

        let iter = output.transformed.iter().map(|point| {
            let [b, g, r, _] = point.rgba_array();
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let diff_inv = (max - min).recip();
            let hue = if diff_inv.is_finite() {
                let hue = match max {
                    value if value == r => (g - b) * diff_inv,
                    value if value == g => (b - r) * diff_inv + 2.,
                    _ => (r - g) * diff_inv + 4.,
                };
                if hue < 0. {
                    (hue + 6.) as f64 / 6.
                } else {
                    hue as f64 / 6.
                }
            } else {
                0.
            };
            (
                point.coords(),
                convert::<_, T>(hue) * convert(self.color.hist_size as f64),
            )
        });

        iter.for_each(|(pivot, bin)| {
            self.color.accum_hist(
                pivot,
                max_coord.clone(),
                bin,
                inc.clone(),
                &mut output.histogram,
            )
        });

        Some(output)
    }
}
