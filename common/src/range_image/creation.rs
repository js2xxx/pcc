use nalgebra::{Affine3, RealField, Vector2, Vector4};
use num::{one, Float, FromPrimitive, ToPrimitive};

use super::{image_to_point, point_to_image, unobserved, RangeImage};
use crate::{
    point::{Point, PointRange, PointViewpoint},
    point_cloud::PointCloud,
};

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

        let max_size = Float::asin(radius / norm) * (one::<P::Data>() + one());

        let radius = options.angular_resolution.map(|r| {
            Float::ceil(max_size / r / (one::<P::Data>() + one()))
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
                let width = ri.point_cloud.width();
                let point: &mut P = &mut ri.point_cloud[y * width + x];
                let range = point.range();

                *point.coords_mut() = image_to_point(
                    &Vector2::new(x, y).map(|x| P::Data::from_usize(x).unwrap()),
                    range,
                    &ri.transform,
                    &ri.angular_resolution,
                    &ri.image_offset,
                );
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
