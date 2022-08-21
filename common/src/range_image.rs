mod creation;
mod surface;

use std::{mem, ops::Deref};

use nalgebra::{Affine3, ComplexField, RealField, Vector2, Vector4};
use num::{Float, ToPrimitive};

pub use self::{creation::CreateOptions, surface::SurfaceInfo};
use crate::{point::PointRange, point_cloud::PointCloud};

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
    let point = to_image * nalgebra::Point3::from_homogeneous(coords.clone()).unwrap();
    let range = point.coords.norm();
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

fn image_to_point<T: RealField>(
    image: &Vector2<T>,
    range: T,
    to_world: &Affine3<T>,
    ar: &Vector2<T>,
    io: &Vector2<usize>,
) -> Vector4<T> {
    let [[ax, ay]] = image_to_angle(image, ar, io).data.0;
    let cosy = ay.clone().cos();
    (to_world
        * nalgebra::Point3::new(
            range.clone() * ax.clone().sin() * cosy.clone(),
            range.clone() * ay.sin(),
            range * ax.cos() * cosy,
        ))
    .to_homogeneous()
}

#[inline]
pub fn unobserved<P>() -> P
where
    P: PointRange,
    P::Data: Float,
{
    P::default().with_range(-P::Data::infinity())
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

impl<P: PointRange> RangeImage<P>
where
    P::Data: RealField + ToPrimitive,
{
    pub fn image_to_point(
        &self,
        image: &Vector2<P::Data>,
        range: Option<P::Data>,
    ) -> Vector4<P::Data> {
        let range = match range {
            Some(range) => range,
            None => {
                let [[x, y]] = image.map(|x| x.round().to_usize().unwrap()).data.0;
                self.point_cloud[(x, y)].range()
            }
        };
        image_to_point(
            image,
            range,
            &self.transform,
            &self.angular_resolution,
            &self.image_offset,
        )
    }

    pub fn point_to_image(&self, point: &Vector4<P::Data>) -> (Vector2<P::Data>, P::Data) {
        point_to_image(
            point,
            &self.inverse_transform,
            &self.angular_resolution,
            &self.image_offset,
        )
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
