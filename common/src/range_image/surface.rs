use std::iter;

use nalgebra::{
    convert, Affine3, ComplexField, Const, Matrix3, RealField, SymmetricEigen, Vector2, Vector3,
    Vector4,
};
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
        let x = ((x - radius)..=(x + radius)).step_by(step);
        let y = ((y - radius)..=(y + radius)).step_by(step);

        let coords = x
            .flat_map(|x| y.clone().map(move |y| &self.point_cloud[(x, y)]))
            .filter_map(|point| {
                (point.is_finite() && point.range().is_finite()).then(|| point.coords())
            });

        Some(crate::cov_matrix(coords)?.symmetric_eigen())
    }

    pub fn normal(
        &self,
        index: (usize, usize),
        radius: usize,
        step: usize,
    ) -> Option<Vector4<P::Data>> {
        let symmetric_eigen = self.normal_inner(index, radius, step)?;
        let index = symmetric_eigen.eigenvalues.imin();
        let normal = symmetric_eigen.eigenvectors.column(index).into_owned();
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
        let curvature = eigens.min();
        Some(curvature / eigens.sum())
    }

    pub fn normal_data(
        &self,
        index: (usize, usize),
        radius: usize,
        step: usize,
    ) -> Option<(Vector4<P::Data>, P::Data)> {
        let eigen = self.normal_inner(index, radius, step)?;
        let index = eigen.eigenvalues.imin();
        let normal = eigen.eigenvectors.column(index).into_owned();
        let curvature = eigen.eigenvalues[index].clone();
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
            .insert_row(3, zero())
        };

        if let Some(pedal) = pedal {
            *pedal = &normal * (normal.xyz().dot(&surface_info.mean) - normal.dot(pivot)) + pivot;
        }

        Some(normal)
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

impl<P: PointRange> RangeImage<P>
where
    P::Data: RealField + Float,
{
    pub fn interp_surface_projection(
        &self,
        pose: &Affine3<P::Data>,
        pixel_size: usize,
        world_size: P::Data,
    ) -> Vec<P::Data> {
        let max_distance = world_size / convert(2.);
        let cell_size = world_size / convert(pixel_size as f64);

        let w2c_factor = Float::recip(cell_size);
        let w2c_offset = (convert::<_, P::Data>(pixel_size as f64) - one()) / convert(2.);

        let c2w_factor = cell_size;
        let c2w_offset = cell_size / convert(2.) - max_distance;

        let inverse_pose = pose.inverse();

        let mut patches = vec![-P::Data::infinity(); pixel_size * pixel_size];
        self.get_patches(
            pose,
            pixel_size,
            max_distance,
            w2c_factor,
            w2c_offset,
            &mut patches,
        );
        self.adjust_neighbor_patches(
            inverse_pose,
            pixel_size,
            max_distance,
            (c2w_factor, c2w_offset),
            &mut patches,
        );
        patches
    }

    fn get_patches(
        &self,
        pose: &Affine3<P::Data>,
        pixel_size: usize,
        max_distance: P::Data,
        w2c_factor: P::Data,
        w2c_offset: P::Data,
        patches: &mut [P::Data],
    ) {
        let position = pose.matrix().column(3);
        let (image, _) = self.point_to_image2(&position.into_owned());

        let (mut vxmin, mut vymin) = (image.x.saturating_sub(1), image.y);
        let (mut vxmax, mut vymax) = (vxmin, vymin);

        let (wxmin, wymin) = (0, 0);
        let (wxmax, wymax) = (self.point_cloud.width(), self.point_cloud.height());
        loop {
            let mut points = {
                let top = (vxmax < wxmax && vymax < wymax)
                    .then(|| (vxmin..=(vxmax + 1)).zip(iter::repeat(vymax + 1)))
                    .into_iter()
                    .flatten();
                let bottom = (vxmin > wxmin && vymin > wymin)
                    .then(|| ((vxmin - 1)..=vxmax).zip(iter::repeat(vymin - 1)))
                    .into_iter()
                    .flatten();
                let left = (vxmin > wxmin && vymax < wymax)
                    .then(|| iter::repeat(vxmin - 1).zip(vymin..=(vymax + 1)))
                    .into_iter()
                    .flatten();
                let right = (vxmax < wxmax && vymin > wymin)
                    .then(|| iter::repeat(vxmax + 1).zip((vymin - 1)..=vymax))
                    .into_iter()
                    .flatten();
                top.chain(bottom).chain(left).chain(right).peekable()
            };
            if points.peek().is_none() {
                break;
            }

            let triangles = points.flat_map(|(x, y)| {
                [
                    [[x, y], [x + 1, y + 1], [x, y + 1]],
                    [[x, y], [x + 1, y + 1], [x + 1, y]],
                ]
                .into_iter()
                .map(|point| {
                    point.map(|point| {
                        pose * nalgebra::Point3::from_homogeneous(
                            self.image_to_point2(&point.into(), None),
                        )
                        .unwrap()
                    })
                })
            });

            for point in triangles {
                if point.iter().any(|point| {
                    Float::abs(point.x) > max_distance || Float::abs(point.y) > max_distance
                }) {
                    continue;
                }

                let cell = point.map(|point| {
                    Vector2::from([
                        point.x * w2c_factor + w2c_offset,
                        point.y * w2c_factor + w2c_offset,
                    ])
                });

                let (min, max) = cell.iter().fold(
                    (
                        Vector2::repeat(convert((pixel_size - 1) as f64)),
                        Vector2::zeros(),
                    ),
                    |(min, max), point| {
                        (
                            [Float::min(min.x, point.x), Float::min(min.y, point.y)].into(),
                            [Float::max(max.x, point.x), Float::max(max.y, point.y)].into(),
                        )
                    },
                );
                let (min, max) = (
                    min.map(|min| Float::max(Float::ceil(min), zero()).to_usize().unwrap()),
                    max.map(|max| {
                        Float::min(Float::floor(max), convert((pixel_size - 1) as f64))
                            .to_usize()
                            .unwrap()
                    }),
                );

                let v0 = cell[2] - cell[0];
                let v1 = cell[1] - cell[0];

                let dot00 = v0.dot(&v0);
                let dot01 = v0.dot(&v1);
                let dot11 = v1.dot(&v1);
                let inv_denom = Float::recip(dot00 * dot11 - dot01 * dot01);

                for x in min.x..=max.x {
                    for y in min.y..=max.y {
                        let current = Vector2::new(x, y).map(|x| convert(x as f64));
                        let v2 = current - cell[0];

                        let dot02 = v0.dot(&v2);
                        let dot12 = v1.dot(&v2);
                        let u = inv_denom * (dot11 * dot02 - dot01 * dot12);
                        let v = inv_denom * (dot00 * dot12 - dot01 * dot02);

                        if u < zero() || v < zero() || (u + v > one()) {
                            continue;
                        }

                        let value = point[0].z * (one::<P::Data>() - u - v)
                            + u * point[2].z
                            + v * point[1].z;

                        let patch = &mut patches[y * pixel_size + x];
                        *patch = if *patch == -P::Data::infinity() {
                            value
                        } else {
                            Float::min(*patch, value)
                        };
                    }
                }
            }

            vxmin = vxmin.saturating_sub(1).max(wxmin);
            vxmax = (vxmax + 1).min(wxmax);
            vymin = vymin.saturating_sub(1).max(wymin);
            vymax = (vymax + 1).min(wymax);
        }
    }

    fn adjust_neighbor_patches(
        &self,
        inverse_pose: Affine3<P::Data>,
        pixel_size: usize,
        max_distance: P::Data,
        (c2w_factor, c2w_offset): (P::Data, P::Data),
        patches: &mut [P::Data],
    ) {
        for index in 0..patches.len() {
            if !patches[index].is_finite() {
                continue;
            }
            let y = index / pixel_size;
            let x = index % pixel_size;

            let xmin = x.saturating_sub(1);
            let xmax = (x + 1).min(pixel_size - 1);
            let ymin = y.saturating_sub(1);
            let ymax = (y + 1).min(pixel_size - 1);

            let mut is_background = false;
            'outer: for nx in xmin..=xmax {
                for ny in ymin..=ymax {
                    let neighbor = patches[ny * pixel_size + nx];
                    if !neighbor.is_finite() {
                        continue;
                    }

                    let cell_x = convert::<_, P::Data>(x as f64 + 0.6 * (x as f64 - nx as f64));
                    let cell_y = convert::<_, P::Data>(y as f64 + 0.6 * (y as f64 - ny as f64));
                    let fake = inverse_pose
                        * nalgebra::Point3::new(
                            cell_x * c2w_factor + c2w_offset,
                            cell_y * c2w_factor + c2w_offset,
                            neighbor,
                        );
                    let range_diff = self
                        .range_diff(&fake.to_homogeneous())
                        .unwrap_or(-P::Data::infinity());
                    if range_diff > max_distance {
                        patches[index] = P::Data::infinity();
                        is_background = true;
                        break 'outer;
                    }
                }
            }

            if is_background {
                for nx in xmin..=xmax {
                    for ny in ymin..=ymax {
                        let neighbor = &mut patches[ny * pixel_size + nx];
                        if *neighbor == -P::Data::infinity() {
                            *neighbor = P::Data::infinity();
                        }
                    }
                }
            }
        }
    }
}
