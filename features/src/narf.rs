use std::{array, mem};

use nalgebra::{convert, Affine3, RealField, Rotation3, Translation3, Vector2, Vector3, Vector4};
use num::{Float, ToPrimitive};
use pcc_common::{feature::Feature, point::PointRange, range_image::RangeImage};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SurfacePatch<T> {
    pub data: Vec<T>,
    pub pixel_size: usize,
    pub world_size: T,
    pub rotation: T,
}

impl<T: RealField + Float> SurfacePatch<T> {
    #[inline]
    pub fn new<P>(
        range_image: &RangeImage<P>,
        pose: &Affine3<T>,
        pixel_size: usize,
        world_size: T,
    ) -> Self
    where
        P: PointRange<Data = T>,
    {
        SurfacePatch {
            data: range_image.interp_surface_projection(pose, pixel_size, world_size),
            pixel_size,
            world_size,
            rotation: T::zero(),
        }
    }

    pub fn blur(&mut self, radius: usize) {
        let new_size = self.pixel_size * 2;
        let mut integral_image = vec![T::zero(); new_size];

        for x in 0..new_size {
            for y in 0..new_size {
                let (old_x, old_y) = (x / 2, y / 2);
                let index = y * new_size + x;
                let old_index = old_y * self.pixel_size + old_x;
                integral_image[index] = self.data[old_index];

                if !integral_image[index].is_finite() {
                    integral_image[index] = self.world_size / convert(2.)
                }

                let [mut left, mut top, mut left_top] = array::from_fn(|_| T::zero());
                if x > 0 {
                    left = integral_image[y * new_size + x - 1];
                    if y > 0 {
                        left_top = integral_image[(y - 1) * new_size + x - 1];
                    }
                }
                if y > 0 {
                    top = integral_image[(y - 1) * new_size + x];
                }
                integral_image[index] += left + top - left_top;
            }
        }
        let integral_image = integral_image;

        self.data.clear();
        self.data.resize(new_size * new_size, T::zero());

        let iter = (0..(new_size * new_size)).map(|index| {
            let x = index % new_size;
            let y = index / new_size;

            let xmax = (x + radius).min(new_size - 1);
            let ymax = (y + radius).min(new_size - 1);
            let xmin = x.checked_sub(radius + 1);
            let ymin = y.checked_sub(radius + 1);
            let prod = xmin.map_or(xmax + 1, |xmin| xmax - xmin)
                * ymin.map_or(ymax + 1, |ymin| ymax - ymin);
            let factor = Float::recip(convert::<_, T>(prod as f64));

            let [mut bottom_left, mut top_right, mut top_left] = array::from_fn(|_| T::zero());
            let bottom_right = integral_image[ymax * new_size + xmax];
            if let Some(xmin) = xmin {
                bottom_left = integral_image[ymax * new_size + xmin];
                if let Some(ymin) = ymin {
                    top_left = integral_image[ymin * new_size + xmin];
                }
            }
            if let Some(ymin) = ymin {
                top_right = integral_image[ymin * new_size + xmax];
            }

            factor * (bottom_right + top_left - bottom_left - top_right)
        });

        self.data.extend(iter);
        self.pixel_size = new_size;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Narf<T: RealField> {
    pub position: Vector4<T>,
    pub transform: Affine3<T>,
    pub descriptor: Vec<T>,
    pub surface_patch: SurfacePatch<T>,
}

impl<T: RealField + Float + ToPrimitive> Narf<T> {
    pub fn new<P>(
        range_image: &RangeImage<P>,
        pose: Affine3<T>,
        desc_size: usize,
        pixel_size: usize,
        world_size: T,
    ) -> Self
    where
        P: PointRange<Data = T>,
    {
        let position = pose.inverse().matrix().column(3).into_owned();
        let mut surface_patch = SurfacePatch::new(range_image, &pose, pixel_size, world_size);
        surface_patch.blur(1);
        let descriptor = Self::extract(desc_size, &surface_patch, &mut Vec::new());

        Narf {
            position,
            transform: pose,
            descriptor,
            surface_patch,
        }
    }

    pub fn rotated_into<P>(
        range_image: &RangeImage<P>,
        pose: Affine3<T>,
        desc_size: usize,
        pixel_size: usize,
        world_size: T,
    ) -> impl Iterator<Item = Narf<T>>
    where
        P: PointRange<Data = T>,
    {
        let orig = Self::new(range_image, pose, desc_size, pixel_size, world_size);
        let (rotations, _) = orig.rotations();

        rotations
            .into_iter()
            .map(move |rotation| orig.clone().rotate(rotation))
    }

    pub fn rotated_into_par<P>(
        range_image: &RangeImage<P>,
        pose: Affine3<T>,
        desc_size: usize,
        pixel_size: usize,
        world_size: T,
    ) -> impl ParallelIterator<Item = Narf<T>>
    where
        P: PointRange<Data = T>,
    {
        let orig = Self::new(range_image, pose, desc_size, pixel_size, world_size);
        let (rotations, _) = orig.rotations();

        rotations
            .into_par_iter()
            .map(move |rotation| orig.clone().rotate(rotation))
    }

    fn extract(desc_size: usize, surface_patch: &SurfacePatch<T>, storage: &mut Vec<T>) -> Vec<T> {
        let num_beams = (surface_patch.pixel_size as f64 / 2.).ceil() as usize;
        let (weight_factor, weight_offset) = {
            let weight_first = convert::<_, T>(2.);
            (
                (weight_first - T::one())
                    / ((weight_first + T::one()) * convert((num_beams - 1) as f64))
                    * convert(-2.),
                weight_first / (weight_first + T::one()) * convert(2.),
            )
        };

        let angle_step = T::two_pi() / convert(num_beams as f64);

        let cell_size = surface_patch.world_size / convert(surface_patch.pixel_size as f64);
        let cell_factor = Float::recip(cell_size);
        let cell_offset = (surface_patch.world_size - cell_size) / convert(2.);
        let max_distance = surface_patch.world_size / convert(2.);

        let beam_factor = (max_distance - cell_size / convert(2.)) / convert(num_beams as f64);

        let iter = (0..desc_size).map(|index| {
            let angle = angle_step * convert(index as f64) + surface_patch.rotation;
            let beam_factor = Vector2::new(Float::sin(angle), Float::cos(angle)) * beam_factor;

            let iter = (0..=num_beams).map(|index| {
                let beam = beam_factor.scale(convert(index as f64));
                let cell = beam.map(|beam| {
                    Float::round(cell_factor * (beam + cell_offset))
                        .to_usize()
                        .unwrap()
                });
                let value = surface_patch.data[cell.y * surface_patch.pixel_size + cell.x];
                if value.is_finite() {
                    value
                } else if value.is_sign_positive() {
                    max_distance
                } else {
                    -T::infinity()
                }
            });

            let sum = Next2Window::new(iter).enumerate().map(|(index, (b1, b2))| {
                let weight = weight_factor * convert(index as f64) + weight_offset;
                let diff = b2 - b1;
                weight * diff
            });

            Float::atan2(sum.fold(T::zero(), |acc, e| acc + e), max_distance) / T::pi()
        });
        storage.clear();
        storage.extend(iter);
        mem::take(storage)
    }

    pub fn rotations(&self) -> (Vec<T>, Vec<T>) {
        let num_angle_steps = self.descriptor.len().max(36);
        let min_angle = convert::<_, T>(70.).to_radians();

        let angle_step = T::two_pi() / convert(num_angle_steps as f64);
        let angle_step2 = T::two_pi() / convert(self.descriptor.len() as f64);
        let score_norm = Float::recip(convert::<_, T>(self.descriptor.len() as f64));

        let orientations = (0..num_angle_steps).map(|step| {
            let angle = angle_step * convert(step as f64);
            let score = self.descriptor.iter().enumerate().map(|(index, value)| {
                let angle2 = angle_step2 * convert(index as f64);
                let weight = Float::powi(
                    T::one() - Float::abs((angle - angle2) % T::two_pi()) / T::pi(),
                    2,
                );
                *value * weight
            });
            let score = score.fold(T::zero(), |acc, e| acc + e) * score_norm + convert(0.5);
            (score, angle)
        });
        let mut orientations = {
            let mut vec = orientations.collect::<Vec<_>>();
            vec.sort_by(|(s1, _), (s2, _)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal));
            vec
        };
        let min = orientations.first().unwrap().0;
        let max = orientations.last().unwrap().0;
        let bound = orientations.partition_point(|&(x, _)| x <= max - (max - min) * convert(0.2));
        orientations.truncate(bound);

        let mut rotations = Vec::new();
        let mut strengths = Vec::new();
        while let Some((score, angle)) = orientations.pop() {
            rotations.push(angle);
            strengths.push(score);
            orientations.retain(|&(_, angle)| {
                (angle - *rotations.last().unwrap()) % T::two_pi() < min_angle
            });
        }
        (rotations, strengths)
    }

    pub fn rotate(self, rotation: T) -> Self {
        let mut new = self;
        new.transform = Rotation3::new(Vector3::z() * -rotation) * new.transform;
        new.surface_patch.rotation = rotation;
        let mut storage = new.descriptor;
        new.descriptor = Self::extract(storage.len(), &new.surface_patch, &mut storage);
        new
    }

    pub fn rotate_all(self, rotations: &[T]) -> impl Iterator<Item = Self> + '_ {
        rotations
            .iter()
            .map(move |&rotation| self.clone().rotate(rotation))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Next2Window<I: Iterator> {
    windows: Option<I::Item>,
    inner: I,
}

impl<I: Iterator> Next2Window<I> {
    #[inline]
    pub fn new(inner: I) -> Next2Window<I> {
        Next2Window {
            windows: None,
            inner,
        }
    }
}

impl<I: Iterator> Iterator for Next2Window<I>
where
    I::Item: Clone,
{
    type Item = (I::Item, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let left = self.windows.take().or_else(|| self.inner.next())?;
        let right = self.windows.insert(self.inner.next()?);
        Some((left, right.clone()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a, b) = self.inner.size_hint();
        (a.saturating_sub(1), b.map(|b| b.saturating_sub(1)))
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Next2Window<I> where I::Item: Clone {}

pub struct NarfEstimation<T: RealField> {
    pub desc_size: usize,
    pub pixel_size: usize,
    pub world_size: T,
    pub rotate: bool,
}

impl<T: RealField> NarfEstimation<T> {
    pub fn new(desc_size: usize, pixel_size: usize, world_size: T, rotate: bool) -> Self {
        NarfEstimation {
            desc_size,
            pixel_size,
            world_size,
            rotate,
        }
    }
}

impl<'a, T, P> Feature<&'a RangeImage<P>, Vec<Narf<T>>, (), ()> for NarfEstimation<T>
where
    T: RealField + Float + ToPrimitive,
    P: PointRange<Data = T> + Sync,
{
    fn compute(&self, input: &'a RangeImage<P>, _: (), _: ()) -> Vec<Narf<T>> {
        let transform = (0..input.len()).into_par_iter().filter_map(|index| {
            let [x, y] = input.index(index);

            let mut pedal = Vector4::zeros();
            let normal = input.normal_within((x, y), 2, 1, None, Some(15), Some(&mut pedal))?;
            Some(
                Translation3::from(-pedal.xyz())
                    * Rotation3::look_at_lh(&normal.xyz(), &Vector3::y()),
            )
        });

        if self.rotate {
            let narfs = transform.flat_map(|transform| {
                Narf::rotated_into_par(
                    input,
                    convert(transform),
                    self.desc_size,
                    self.pixel_size,
                    self.world_size,
                )
            });
            narfs.collect()
        } else {
            let narfs = transform.map(|transform| {
                Narf::new(
                    input,
                    convert(transform),
                    self.desc_size,
                    self.pixel_size,
                    self.world_size,
                )
            });
            narfs.collect()
        }
    }
}
