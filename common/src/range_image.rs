use std::mem;

use nalgebra::{Affine3, RealField, Vector2, Vector3, Vector4};
use num::{Float, ToPrimitive};

use crate::{
    point_cloud::PointCloud,
    points::{Centroid, Point3Infoed, Point3Range, PointInfoRange, PointViewpoint},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeImage<T: RealField> {
    point_cloud: PointCloud<Point3Range<T>>,
    transform: Affine3<T>,
    inverse_transform: Affine3<T>,
    angular_resolution: Vector2<T>,
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
pub fn unobserved<T: RealField + Float>() -> Point3Infoed<T, PointInfoRange<T>> {
    Point3Range {
        coords: Vector4::zeros(),
        extra: PointInfoRange {
            range: T::infinity(),
        },
    }
}

impl<T: RealField> RangeImage<T> {
    pub fn contains_key(&self, x: usize, y: usize) -> bool {
        x < self.point_cloud.width() && y < self.point_cloud.height()
    }
}

impl<T: RealField + ToPrimitive + Float> RangeImage<T> {
    pub fn new<I>(
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
        angular_resolution: &[T; 2],
        angle_size: &[T; 2],
        sensor_pose: Affine3<T>,
        noise: T,
        min_range: T,
        border_size: usize,
    ) -> Self {
        let angular_resolution = Vector2::from(*angular_resolution);
        let angle_size = Vector2::from(*angle_size);

        let size = { angle_size.component_div(&angular_resolution) }
            .map(|x| Float::floor(x).to_usize().unwrap());
        let full_size = { Vector2::new(T::two_pi(), T::pi()).component_div(&angular_resolution) }
            .map(|x| Float::floor(x).to_usize().unwrap());

        let image_offset = (full_size - size) / 2;

        let mut ri = RangeImage {
            point_cloud: PointCloud::new(),
            transform: sensor_pose,
            inverse_transform: sensor_pose.inverse(),
            angular_resolution,
            image_offset,
        };

        let boundaries = ri.init(
            point_cloud.iter().map(|point| &point.coords),
            size[0],
            size[1],
            noise,
            min_range,
        );
        ri.crop(border_size, &boundaries);

        for x in 0..ri.point_cloud.width() {
            for y in 0..ri.point_cloud.height() {
                ri.calculate_at(x, y);
            }
        }

        ri
    }

    pub fn within_sphere<I>(
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
        angular_resolution: &[T; 2],
        &(center, radius): &(Vector4<T>, T),
        sensor_pose: Affine3<T>,
        noise: T,
        min_range: T,
        border_size: usize,
    ) -> Self {
        let norm = (center - sensor_pose.matrix().column(3)).norm();
        if norm < radius {
            return Self::new(
                point_cloud,
                angular_resolution,
                &[T::pi(), T::frac_pi_2()],
                sensor_pose,
                noise,
                min_range,
                border_size,
            );
        }

        let max_size = Float::asin(radius / norm) * (T::one() + T::one());
        let xradius = Float::ceil(max_size / angular_resolution[0] / (T::one() + T::one()))
            .to_usize()
            .unwrap();
        let yradius = Float::ceil(max_size / angular_resolution[1] / (T::one() + T::one()))
            .to_usize()
            .unwrap();
        let width = 2 * xradius;
        let height = 2 * yradius;

        let angular_resolution = Vector2::from(*angular_resolution);
        let inverse_transform = sensor_pose.inverse();
        let (center, _) = point_to_image(
            &center,
            &inverse_transform,
            &angular_resolution,
            &Vector2::zeros(),
        );

        let mut ri = RangeImage {
            point_cloud: PointCloud::new(),
            transform: sensor_pose,
            inverse_transform,
            angular_resolution,
            image_offset: Vector2::new(
                center.x.to_usize().unwrap().saturating_sub(xradius),
                center.y.to_usize().unwrap().saturating_sub(yradius),
            ),
        };

        let boundaries = ri.init(
            point_cloud.iter().map(|point| &point.coords),
            width,
            height,
            noise,
            min_range,
        );
        ri.crop(border_size, &boundaries);

        for x in 0..ri.point_cloud.width() {
            for y in 0..ri.point_cloud.height() {
                ri.calculate_at(x, y);
            }
        }

        ri
    }

    pub fn with_viewpoint<I: PointViewpoint<T> + Centroid>(
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
        angular_resolution: &[T; 2],
        angle_size: &[T; 2],
        noise: T,
        min_range: T,
        border_size: usize,
    ) -> Self
    where
        T: Centroid + Default,
        T::Accumulator: Default,
        I::Accumulator: Default,
    {
        let viewpoint = { point_cloud.centroid().0.unwrap() }
            .extra
            .viewpoint()
            .viewpoint;
        let mut sensor_pose = Affine3::identity();
        *sensor_pose.matrix_mut_unchecked().column_mut(3) = *viewpoint;

        Self::new(
            point_cloud,
            angular_resolution,
            angle_size,
            sensor_pose,
            noise,
            min_range,
            border_size,
        )
    }

    fn init<'a, Iter>(
        &mut self,
        points: Iter,
        width: usize,
        height: usize,
        noise: T,
        min_range: T,
    ) -> [usize; 4]
    where
        Iter: Iterator<Item = &'a Vector4<T>>,
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
                coords,
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
                        let value = self.point_cloud[ny * width + nx].extra.range;
                        self.point_cloud[ny * width + nx].extra.range = Float::min(value, range);

                        xmin = Some(xmin.map_or(nx, |xmin| nx.max(xmin)));
                        xmax = Some(xmax.map_or(nx, |xmax| nx.max(xmax)));
                        ymin = Some(ymin.map_or(ny, |ymin| ny.min(ymin)));
                        ymax = Some(ymax.map_or(ny, |ymax| ny.max(ymax)));
                    }
                }
            }

            let index = y * width + x;
            let counter = &mut counter[index];
            let min_range = &mut self.point_cloud[index].extra.range;
            if *counter == 0 || range < *min_range - noise {
                *counter = 1;
                *min_range = range;
                xmin = Some(xmin.map_or(x, |xmin| x.max(xmin)));
                xmax = Some(xmax.map_or(x, |xmax| x.max(xmax)));
                ymin = Some(ymin.map_or(y, |ymin| y.min(ymin)));
                ymax = Some(ymax.map_or(y, |ymax| y.max(ymax)));
            } else if Float::abs(range - *min_range) < noise {
                *counter += 1;
                *min_range += (range - *min_range) / T::from_usize(*counter).unwrap();
            }
        }

        self.point_cloud.reinterpret(width);

        [xmin.unwrap(), xmax.unwrap(), ymin.unwrap(), ymax.unwrap()]
    }
}

impl<T: RealField + Float> RangeImage<T> {
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
                    old[old_y * old.width() + old_x]
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
        let range = point.extra.range;

        let angle = image_to_angle(
            &Vector2::new(x, y).map(|x| T::from_usize(x).unwrap()),
            &self.angular_resolution,
            &self.image_offset,
        );

        let cosy = Float::cos(angle.y);
        point.coords = (self.transform
            * Vector3::new(
                range * Float::sin(angle.x) * cosy,
                range * Float::sin(angle.y),
                range * Float::cos(angle.x) * cosy,
            ))
        .insert_row(1, T::one());
    }
}
