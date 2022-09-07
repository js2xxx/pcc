use nalgebra::{
    convert, DVector, IsometryMatrix3, Matrix3, RealField, Rotation3, Scalar, Translation3,
    Vector3, Vector4,
};
use num::ToPrimitive;
use pcc_common::{
    feature::Feature,
    point::Point,
    point_cloud::{AsPointCloud, PointCloud},
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
pub struct GasdEstimation<T: Scalar> {
    pub view_direction: Vector3<T>,
    pub half_grid_size: usize,
    pub hist_size: usize,
}

impl<T: Scalar> GasdEstimation<T> {
    #[inline]
    pub fn new(view_direction: Vector3<T>, half_grid_size: usize, hist_size: usize) -> Self {
        GasdEstimation {
            view_direction,
            half_grid_size,
            hist_size,
        }
    }

    pub fn get_transform<P>(&self, input: &PointCloud<P>) -> Option<IsometryMatrix3<P::Data>>
    where
        T: RealField,
        P: Point<Data = T>,
    {
        let centroid = input.centroid_coords().0?;
        let se =
            pcc_common::cov_matrix(input.iter().map(|point| point.coords()))?.symmetric_eigen();
        let z = {
            let vec = se.eigenvectors.column(se.eigenvalues.imin());
            if vec.dot(&self.view_direction) > T::zero() {
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

    fn accum_hist(&self, pivot: &Vector4<T>, max_coord: T, bin: T, inc: T, hist: &mut [DVector<T>])
    where
        T: RealField + ToPrimitive,
    {
        let grid_size = self.half_grid_size * 2;
        let half_grid_size = convert::<_, T>(self.half_grid_size as f64);
        let scaled = { pivot.xyz() }.map(|x| x / max_coord.clone() * half_grid_size.clone());
        let coords = scaled
            .map(|x| x + half_grid_size.clone())
            .insert_row(3, bin);
        let bins = coords.map(|x| x.floor().to_usize().unwrap());
        let g_index = ((bins.x + 1) * (grid_size + 2) + bins.y + 1) * (grid_size + 2) + bins.z + 1;
        let h_index = bins.w + 1;
        hist[g_index][h_index] += inc;
    }
}

impl<'a, T, P> Feature<&'a PointCloud<P>, Option<GasdOutput<P>>, (), ()> for GasdEstimation<T>
where
    T: RealField + ToPrimitive,
    P: Point<Data = T>,
{
    fn compute(&self, input: &'a PointCloud<P>, _: (), _: ()) -> Option<GasdOutput<P>> {
        let transform = self.get_transform(input)?;
        let transformed =
            input.map(|point| point.clone().with_na_point(&transform * point.na_point()));

        let grid_size = self.half_grid_size * 2;

        let centroid = Vector3::zeros().insert_row(3, T::one());
        let (_, far) = (&transformed).max_distance(&centroid)?;
        let factor = (centroid - far).norm();

        let [min, max] = (&transformed).finite_bound()?;
        let max_coord = min.xyz().abs().max().max(max.xyz().abs().max());
        let inc = convert::<_, T>(HIST_MAX) / convert((transformed.len() - 1) as f64);

        let iter = transformed.iter().map(|point| {
            let distance = point.coords().xyz().norm();
            let step = factor.clone() / convert(self.half_grid_size as f64);
            let value = (distance / step).fract();
            (point.coords(), value * convert(self.hist_size as f64))
        });
        let histogram = iter.fold(
            vec![
                DVector::zeros(self.hist_size + 2);
                (grid_size + 2) * (grid_size + 2) * (grid_size + 2)
            ],
            |mut acc, (pivot, bin)| {
                self.accum_hist(pivot, max_coord.clone(), bin, inc.clone(), &mut acc);
                acc
            },
        );

        Some(GasdOutput {
            transform,
            transformed,
            histogram,
        })
    }
}
