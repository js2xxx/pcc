use std::{mem, ops::Deref};

use nalgebra::{Affine3, RealField, Vector2, Vector3, Vector4};
use num::{Float, FromPrimitive, One, ToPrimitive};

use crate::{
    point::{Point, PointRange, PointViewpoint},
    point_cloud::PointCloud,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeImage<P>
where
    P: PointRange,
    P::Data: RealField,
{
    point_cloud: PointCloud<P>,
    transform: Affine3<P::Data>,
    inverse_transform: Affine3<P::Data>,
    angular_resolution: Vector2<P::Data>,
    image_offset: Vector2<usize>,
}

fn angle_to_image<T: RealField>(
    angle: &Vector2<T>,
    ar: &Vector2<T>,
    io: &Vector2<usize>,
) -> Vector2<T> {
    let pi = T::pi();
    let frac_pi_2 = T::frac_pi_2();
    assert!(-pi.clone() <= angle.x && angle.x <= pi);
    assert!(-frac_pi_2.clone() <= angle.y && angle.y <= frac_pi_2);

    Vector2::new(
        angle.x.clone() * angle.y.clone().cos() + pi,
        angle.y.clone() + frac_pi_2,
    )
    .component_div(ar)
        - io.map(|x| T::from_usize(x).unwrap())
}

fn image_to_angle<T: RealField>(
    image: &Vector2<T>,
    ar: &Vector2<T>,
    io: &Vector2<usize>,
) -> Vector2<T> {
    let mid = image + io.map(|x| T::from_usize(x).unwrap()).component_mul(ar);
    let y = mid.y.clone() - T::frac_pi_2();
    let cosy = y.clone().cos();
    let x = if cosy == T::zero() {
        T::zero()
    } else {
        (mid.x.clone() - T::pi()) / cosy
    };
    Vector2::new(x, y)
}

fn point_to_image<T: RealField>(
    coords: &Vector4<T>,
    to_image: &Affine3<T>,
    ar: &Vector2<T>,
    io: &Vector2<usize>,
) -> (Vector2<T>, T) {
    let point = to_image * coords.xyz();
    let range = point.norm();
    if range < T::default_epsilon() {
        (Vector2::zeros(), T::zero())
    } else {
        let image = angle_to_image(
            &Vector2::new(
                point.x.clone().atan2(point.z.clone()),
                (point.y.clone() / range.clone()).asin(),
            ),
            ar,
            io,
        );
        (image, range)
    }
}

#[inline]
pub fn unobserved<P>() -> P
where
    P: PointRange,
    P::Data: Float,
{
    P::default().with_range(P::Data::infinity())
}

pub struct CreateOptions<'a, P: Point> {
    pub point_cloud: &'a PointCloud<P>,
    pub angular_resolution: Vector2<P::Data>,
    pub noise: P::Data,
    pub min_range: P::Data,
    pub border_size: usize,
}

impl<P> RangeImage<P>
where
    P: PointRange,
    P::Data: RealField,
{
    #[inline]
    pub fn empty(angular_resolution: Vector2<P::Data>) -> Self {
        RangeImage {
            point_cloud: PointCloud::new(),
            transform: Affine3::identity(),
            inverse_transform: Affine3::identity(),
            angular_resolution,
            image_offset: Vector2::zeros(),
        }
    }

    #[inline]
    pub fn contains_key(&self, x: usize, y: usize) -> bool {
        x < self.point_cloud.width() && y < self.point_cloud.height()
    }

    #[inline]
    pub fn sensor_pose(&self) -> Vector4<P::Data> {
        self.transform.matrix().column(3).into()
    }

    #[inline]
    pub fn into_inner(ri: Self) -> PointCloud<P> {
        ri.point_cloud
    }
}

impl<P> Deref for RangeImage<P>
where
    P: PointRange,
    P::Data: RealField,
{
    type Target = PointCloud<P>;

    fn deref(&self) -> &Self::Target {
        &self.point_cloud
    }
}

impl<P> From<RangeImage<P>> for PointCloud<P>
where
    P: PointRange,
    P::Data: RealField,
{
    fn from(ri: RangeImage<P>) -> Self {
        ri.point_cloud
    }
}

impl<P> RangeImage<P>
where
    P: PointRange,
    P::Data: RealField + Float,
{
    pub fn new<P2: Point<Data = P::Data>>(
        angle_size: &[P::Data; 2],
        sensor_pose: Affine3<P::Data>,
        options: &CreateOptions<P2>,
    ) -> Self {
        let angle_size = Vector2::from(*angle_size);

        let size = { angle_size.component_div(&options.angular_resolution) }
            .map(|x| Float::floor(x).to_usize().unwrap());
        let full_size = {
            Vector2::new(P::Data::two_pi(), P::Data::pi())
                .component_div(&options.angular_resolution)
        }
        .map(|x| Float::floor(x).to_usize().unwrap());

        let image_offset = (full_size - size) / 2;

        Self::new_inner(sensor_pose, image_offset, size, options)
    }

    pub fn within_sphere<P2: Point<Data = P::Data>>(
        &(center, radius): &(Vector4<P::Data>, P::Data),
        sensor_pose: Affine3<P::Data>,
        options: &CreateOptions<P2>,
    ) -> Self {
        let norm = (center - sensor_pose.matrix().column(3)).norm();
        if norm < radius {
            return Self::new(&[P::Data::pi(), P::Data::frac_pi_2()], sensor_pose, options);
        }

        let max_size = Float::asin(radius / norm) * (P::Data::one() + P::Data::one());

        let radius = options.angular_resolution.map(|r| {
            Float::ceil(max_size / r / (P::Data::one() + P::Data::one()))
                .to_usize()
                .unwrap()
        });
        let size = radius * 2;

        let inverse_transform = sensor_pose.inverse();
        let (center, _) = point_to_image(
            &center,
            &inverse_transform,
            &options.angular_resolution,
            &Vector2::zeros(),
        );

        let image_offset = center.zip_map(&radius, |center, radius| {
            center.to_usize().unwrap().saturating_sub(radius)
        });
        Self::new_inner(sensor_pose, image_offset, size, options)
    }

    pub fn with_viewpoint<P2: PointViewpoint<Data = P::Data>>(
        angle_size: &[P::Data; 2],
        options: &CreateOptions<P2>,
    ) -> Self {
        let viewpoint = options.point_cloud.viewpoint_center().unwrap();
        let mut sensor_pose = Affine3::identity();
        sensor_pose.matrix_mut_unchecked().set_column(3, &viewpoint);

        Self::new(angle_size, sensor_pose, options)
    }

    fn new_inner<P2: Point<Data = P::Data>>(
        sensor_pose: Affine3<P::Data>,
        image_offset: Vector2<usize>,
        size: Vector2<usize>,
        options: &CreateOptions<P2>,
    ) -> RangeImage<P> {
        let mut ri = RangeImage {
            point_cloud: PointCloud::new(),
            transform: sensor_pose,
            inverse_transform: sensor_pose.inverse(),
            angular_resolution: options.angular_resolution,
            image_offset,
        };

        let boundaries = ri.proc_zbuffer(
            options
                .point_cloud
                .iter()
                .map(|point| point.coords().into_owned()),
            size.x,
            size.y,
            options.noise,
            options.min_range,
        );

        ri.crop(options.border_size, &boundaries);

        for x in 0..ri.point_cloud.width() {
            for y in 0..ri.point_cloud.height() {
                ri.calculate_at(x, y);
            }
        }
        ri
    }

    fn proc_zbuffer<Iter>(
        &mut self,
        points: Iter,
        width: usize,
        height: usize,
        noise: P::Data,
        min_range: P::Data,
    ) -> [usize; 4]
    where
        Iter: Iterator<Item = Vector4<P::Data>>,
    {
        let [mut xmin, mut xmax, mut ymin, mut ymax] = [None; 4];

        unsafe {
            self.point_cloud
                .storage()
                .resize(width * height, unobserved())
        };

        let mut counter = vec![0; width * height];

        for coords in points {
            if !coords.iter().all(|x| x.is_finite()) {
                continue;
            }

            let (image, range) = point_to_image(
                &coords,
                &self.inverse_transform,
                &self.angular_resolution,
                &self.image_offset,
            );

            let (x, y) = (
                Float::round(image.x).to_usize().unwrap(),
                Float::round(image.y).to_usize().unwrap(),
            );

            if range < min_range || !self.contains_key(x, y) {
                continue;
            }

            {
                let neighbors = {
                    let (floor, ceil) = (
                        image.map(|x| Float::floor(x).to_usize().unwrap()),
                        image.map(|x| Float::ceil(x).to_usize().unwrap()),
                    );
                    [
                        (floor.x, floor.y),
                        (floor.x, ceil.y),
                        (ceil.x, floor.y),
                        (ceil.x, ceil.y),
                    ]
                    .into_iter()
                };
                for (nx, ny) in neighbors {
                    if counter[ny * width + nx] == 0 {
                        let value = self.point_cloud[ny * width + nx].range();
                        self.point_cloud[ny * width + nx].set_range(Float::min(value, range));

                        xmin = Some(xmin.map_or(nx, |xmin| nx.max(xmin)));
                        xmax = Some(xmax.map_or(nx, |xmax| nx.max(xmax)));
                        ymin = Some(ymin.map_or(ny, |ymin| ny.min(ymin)));
                        ymax = Some(ymax.map_or(ny, |ymax| ny.max(ymax)));
                    }
                }
            }

            let index = y * width + x;
            let counter = &mut counter[index];
            let min_range = self.point_cloud[index].range_mut();
            if *counter == 0 || range < *min_range - noise {
                *counter = 1;
                *min_range = range;
                xmin = Some(xmin.map_or(x, |xmin| x.max(xmin)));
                xmax = Some(xmax.map_or(x, |xmax| x.max(xmax)));
                ymin = Some(ymin.map_or(y, |ymin| y.min(ymin)));
                ymax = Some(ymax.map_or(y, |ymax| y.max(ymax)));
            } else if Float::abs(range - *min_range) < noise {
                *counter += 1;
                *min_range += (range - *min_range) / P::Data::from_usize(*counter).unwrap();
            }
        }

        self.point_cloud.reinterpret(width);

        [xmin.unwrap(), xmax.unwrap(), ymin.unwrap(), ymax.unwrap()]
    }
}

impl<P: PointRange> RangeImage<P>
where
    P::Data: RealField + Float,
{
    pub fn crop(&mut self, border_size: usize, &[xmin, xmax, ymin, ymax]: &[usize; 4]) {
        let width = xmax - xmin + 1 + border_size * 2;
        let height = ymax - ymin + 1 + border_size * 2;
        let old = mem::replace(&mut self.point_cloud, PointCloud::new());
        unsafe {
            self.point_cloud
                .storage()
                .resize(width * height, unobserved())
        };

        for x in 0..width {
            for y in 0..height {
                let old_x = x + xmin + border_size;
                let old_y = y + ymin + border_size;

                self.point_cloud[y * width + x] = if old_x <= xmax && old_y <= ymax {
                    old[old_y * old.width() + old_x].clone()
                } else {
                    unobserved()
                }
            }
        }

        self.point_cloud.reinterpret(width)
    }

    fn calculate_at(&mut self, x: usize, y: usize) {
        let width = self.point_cloud.width();
        let point = &mut self.point_cloud[y * width + x];
        let range = point.range();

        let angle = image_to_angle(
            &Vector2::new(x, y).map(|x| P::Data::from_usize(x).unwrap()),
            &self.angular_resolution,
            &self.image_offset,
        );

        let cosy = Float::cos(angle.y);
        **point.coords_mut() = *(self.transform
            * Vector3::new(
                range * Float::sin(angle.x) * cosy,
                range * Float::sin(angle.y),
                range * Float::cos(angle.x) * cosy,
            ))
        .insert_row(1, P::Data::one());
    }

    pub fn integrate_far_ranges<'a, Iter>(&mut self, far_ranges: Iter)
    where
        Iter: Iterator<Item = &'a Vector4<P::Data>>,
    {
        for coords in far_ranges {
            if !coords.iter().all(|x| x.is_finite()) {
                continue;
            }

            let (image, _) = point_to_image(
                coords,
                &self.inverse_transform,
                &self.angular_resolution,
                &self.image_offset,
            );

            let neighbors = {
                let (floor, ceil) = (
                    image.map(|x| Float::floor(x).to_usize().unwrap()),
                    image.map(|x| Float::ceil(x).to_usize().unwrap()),
                );
                [
                    (floor.x, floor.y),
                    (floor.x, ceil.y),
                    (ceil.x, floor.y),
                    (ceil.x, ceil.y),
                ]
                .into_iter()
            };
            for (nx, ny) in neighbors {
                if !self.contains_key(nx, ny) {
                    continue;
                }
                let range = &mut self.point_cloud[(nx, ny)].range();
                if !range.is_finite() {
                    *range = P::Data::infinity()
                }
            }
        }
    }

    pub fn create_sub(&self, boundaries: &[usize; 4], combine_pixels: usize) -> Self {
        let image_offset = Vector2::new(boundaries[0], boundaries[2]);

        let width = boundaries[1] - image_offset.x + 1;
        let height = boundaries[3] - image_offset.y + 1;
        let mut storage = vec![unobserved(); width * height];

        let src_base = image_offset * combine_pixels - self.image_offset;
        for x in 0..width {
            for y in 0..height {
                let dst: &mut P = &mut storage[y * width + x];
                for src_x in
                    (src_base.x + combine_pixels * x)..(src_base.x + combine_pixels * (x + 1))
                {
                    for src_y in
                        (src_base.y + combine_pixels * y)..(src_base.y + combine_pixels * (y + 1))
                    {
                        if !self.contains_key(src_x, src_y) {
                            continue;
                        }
                        let src = &self.point_cloud[(src_x, src_y)];
                        if !src.range().is_finite() || src.range() < dst.range() {
                            *dst = src.clone();
                        }
                    }
                }
            }
        }

        RangeImage {
            point_cloud: PointCloud::from_vec(storage, width),
            transform: self.transform,
            inverse_transform: self.inverse_transform,
            angular_resolution: self.angular_resolution,
            image_offset,
        }
    }
}
