use nalgebra::{ComplexField, Const, Matrix3, RealField, SymmetricEigen, Vector3, Vector4};
use num::{one, zero, Float, ToPrimitive};

use super::RangeImage;
use crate::point::PointRange;

#[derive(Debug, Clone)]
pub struct SurfaceInfo<T: ComplexField> {
    pub max_neighbor_distance_squared: T,
    pub mean: Vector3<T>,
    pub eigen: SymmetricEigen<T, Const<3>>,

    pub mean_all_neighbors: Option<Vector3<T>>,
    pub eigen_all_neighbors: Option<SymmetricEigen<T, Const<3>>>,
}

impl<P: PointRange> RangeImage<P>
where
    P::Data: RealField + ToPrimitive,
{
    fn normal_inner(
        &self,
        (x, y): (usize, usize),
        radius: usize,
        step: usize,
    ) -> Option<SymmetricEigen<P::Data, Const<3>>> {
        let mut num = zero::<P::Data>();
        let mut mean = Vector3::zeros();
        let mut cov = Matrix3::zeros();
        for x in ((x - radius)..=(x + radius)).step_by(step) {
            for y in ((y - radius)..=(y + radius)).step_by(step) {
                let point = &self.point_cloud[(x, y)];
                if !point.is_finite() || !point.range().is_finite() {
                    continue;
                }
                let coords = point.coords().xyz();
                cov = (cov * num.clone() + &coords * coords.transpose()) / (num.clone() + one());
                mean = (mean * num.clone() + &coords) / (num.clone() + one());
                num += one();
            }
        }
        if num.to_usize().unwrap() < 3 {
            return None;
        }
        cov -= &mean * mean.transpose();
        Some(cov.symmetric_eigen())
    }

    pub fn normal(
        &self,
        index: (usize, usize),
        radius: usize,
        step: usize,
    ) -> Option<Vector4<P::Data>> {
        let normal = self
            .normal_inner(index, radius, step)?
            .eigenvectors
            .column(0)
            .into_owned();
        Some(
            if normal.dot(&self.sensor_pose().xyz()) < zero() {
                -normal
            } else {
                normal
            }
            .insert_row(3, zero()),
        )
    }

    pub fn curvature(&self, index: (usize, usize), radius: usize, step: usize) -> Option<P::Data> {
        let eigens = self.normal_inner(index, radius, step)?.eigenvalues;
        let curvature = eigens[0].clone();
        Some(curvature / eigens.sum())
    }

    pub fn normal_data(
        &self,
        index: (usize, usize),
        radius: usize,
        step: usize,
    ) -> Option<(Vector4<P::Data>, P::Data)> {
        let eigen = self.normal_inner(index, radius, step)?;
        let normal = eigen.eigenvectors.column(0).into_owned();
        let curvature = eigen.eigenvalues[0].clone();
        Some((
            if normal.dot(&self.sensor_pose().xyz()) < zero() {
                -normal
            } else {
                normal
            }
            .insert_row(3, zero()),
            curvature / eigen.eigenvalues.sum(),
        ))
    }

    pub fn surface_info(
        &self,
        (x, y): (usize, usize),
        radius: usize,
        step: usize,
        pivot: &Vector4<P::Data>,
        num_neighbors: usize,
        all_neighbors: bool,
    ) -> Option<SurfaceInfo<P::Data>> {
        let neighbors = {
            let range = (radius * 2 + 1) / step;
            let mut vec = Vec::with_capacity(range * range);

            for x in ((x - radius)..=(x + radius)).step_by(step) {
                for y in ((y - radius)..=(y + radius)).step_by(step) {
                    let point = &self.point_cloud[(x, y)];
                    if !point.is_finite() || !point.range().is_finite() {
                        continue;
                    }
                    vec.push(((point.coords() - pivot).norm_squared(), point));
                }
            }
            vec.sort_by(|(d1, _), (d2, _)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal));
            vec
        };

        let mut num = zero::<P::Data>();
        let mut mean = Vector3::zeros();
        let mut cov = Matrix3::zeros();
        for (_, point) in neighbors.iter().take(num_neighbors) {
            let coords = point.coords().xyz();
            cov = (cov * num.clone() + &coords * coords.transpose()) / (num.clone() + one());
            mean = (mean * num.clone() + &coords) / (num.clone() + one());
            num += one();
        }

        if num.to_usize().unwrap() < 3 {
            return None;
        }

        let max_neighbor_distance_squared = neighbors.last().unwrap().0.clone();
        let eigen = (&cov - &mean * mean.transpose()).symmetric_eigen();
        let mut mean_all_neighbors = mean;
        let mean = mean_all_neighbors.clone();

        let (mean_all_neighbors, eigen_all_neighbors) = all_neighbors
            .then(move || {
                for (_, point) in neighbors.iter().skip(num_neighbors) {
                    let coords = point.coords().xyz();
                    cov =
                        (cov * num.clone() + &coords * coords.transpose()) / (num.clone() + one());
                    mean_all_neighbors =
                        (mean_all_neighbors * num.clone() + &coords) / (num.clone() + one());
                    num += one();
                }
                cov -= &mean_all_neighbors * mean_all_neighbors.transpose();
                (mean_all_neighbors, cov.symmetric_eigen())
            })
            .unzip();

        Some(SurfaceInfo {
            max_neighbor_distance_squared,
            mean,
            eigen,
            mean_all_neighbors,
            eigen_all_neighbors,
        })
    }

    pub fn normal_within(
        &self,
        index: (usize, usize),
        radius: usize,
        step: usize,
        pivot: Option<&Vector4<P::Data>>,
        num_neighbors: Option<usize>,
        pedal: Option<&mut Vector4<P::Data>>,
    ) -> Option<Vector4<P::Data>> {
        let pivot = pivot.unwrap_or_else(|| self.point_cloud[index].coords());
        let num_neighbors = num_neighbors.unwrap_or((radius + 1) * (radius + 1));
        let surface_info = self.surface_info(index, radius, step, pivot, num_neighbors, false)?;

        let normal = {
            let normal = surface_info.eigen.eigenvectors.column(0).into_owned();
            if normal.dot(&self.sensor_pose().xyz()) < zero() {
                -normal
            } else {
                normal
            }
        };

        if let Some(pedal) = pedal {
            *pedal = (&normal * (normal.dot(&surface_info.mean) - normal.dot(&pivot.xyz()))
                + pivot.xyz())
            .insert_row(3, zero());
        }

        Some(normal.insert_row(3, zero()))
    }

    pub fn impact_angle(&self, index: (usize, usize), radius: usize) -> Option<P::Data> {
        let pivot = self.point_cloud[index].coords();
        let normal = self.normal_within(index, radius, 2, Some(pivot), None, None)?;
        let sina = normal.dot(&(self.sensor_pose() - pivot).normalize());
        Some(sina.asin())
    }

    pub fn acuteness(&self, index: (usize, usize), radius: usize) -> Option<P::Data> {
        self.impact_angle(index, radius)
            .map(|ia| one::<P::Data>() - ia / P::Data::frac_pi_2())
    }
}

impl<P: PointRange> RangeImage<P>
where
    P::Data: RealField + Float,
{
    pub fn impact_angle2(&self, p1: &P, p2: &P) -> Option<P::Data> {
        let (r1, r2) = (p1.range(), p2.range());
        let (r1, r2) = (Float::min(r1, r2), Float::max(r1, r2));

        let angle = if r2 == -P::Data::infinity() {
            return None;
        } else if !r2.is_finite() && r1.is_finite() {
            zero()
        } else if r1.is_finite() {
            // r2.is_finite()
            let (r1s, r2s) = (r1 * r1, r2 * r2);
            let ds = (p2.coords() - p1.coords()).norm_squared();
            let d = Float::sqrt(ds);

            let cosa = (r2s + ds - r1s) / ((one::<P::Data>() + one()) * d * r2);
            Float::acos(cosa.clamp(zero(), one()))
        } else {
            // r2.is_finite() && !r1.is_finite()
            P::Data::frac_pi_2()
        };

        Some(if p1.range() > p2.range() {
            -angle
        } else {
            angle
        })
    }

    pub fn acuteness2(&self, p1: &P, p2: &P) -> Option<P::Data> {
        self.impact_angle2(p1, p2)
            .map(|ia| one::<P::Data>() - ia / P::Data::frac_pi_2())
    }
}
