use std::{array, slice};

use nalgebra::{convert, RealField};
use num::ToPrimitive;
use pcc_common::{
    feature::Feature,
    point::{Centroid, Data, DataFields, FieldInfo, PointRange},
    point_cloud::PointCloud,
    range_image::{RangeImage, SurfaceInfo},
};
use rayon::prelude::*;

bitflags::bitflags! {
    /// Has the same layout with PCL's `BorderTrait` enum.
    #[derive(Default)]
    pub struct BorderTraits: u32 {
        const OBSTACLE_BORDER = 0b0000_0000_0000_0001;
        const SHADOW_BORDER =   0b0000_0000_0000_0010;
        const VEIL_POINT =      0b0000_0000_0000_0100;

        const OBSTACLE_BORDER_TOP =    0b0000_0000_0000_1000;
        const OBSTACLE_BORDER_RIGHT =  0b0000_0000_0001_0000;
        const OBSTACLE_BORDER_BOTTOM = 0b0000_0000_0010_0000;
        const OBSTACLE_BORDER_LEFT =   0b0000_0000_0100_0000;

        const SHADOW_BORDER_TOP =    0b0000_0000_1000_0000;
        const SHADOW_BORDER_RIGHT =  0b0000_0001_0000_0000;
        const SHADOW_BORDER_BOTTOM = 0b0000_0010_0000_0000;
        const SHADOW_BORDER_LEFT =   0b0000_0100_0000_0000;

        const VEIL_POINT_TOP =    0b0000_1000_0000_0000;
        const VEIL_POINT_RIGHT =  0b0001_0000_0000_0000;
        const VEIL_POINT_BOTTOM = 0b0010_0000_0000_0000;
        const VEIL_POINT_LEFT =   0b0100_0000_0000_0000;
    }
}

impl BorderTraits {
    const OBSTACLE_BORDER_START_BIT: usize = 3;
    const SHADOW_BORDER_START_BIT: usize = 7;
    const VEIL_POINT_START_BIT: usize = 11;

    #[inline]
    fn obstacle_border(di: usize) -> Self {
        BorderTraits::OBSTACLE_BORDER
            | Self::from_bits_truncate(1 << (di + Self::OBSTACLE_BORDER_START_BIT))
    }

    #[inline]
    fn shadow_border(di: usize) -> Self {
        BorderTraits::SHADOW_BORDER
            | Self::from_bits_truncate(1 << (di + Self::SHADOW_BORDER_START_BIT))
    }

    #[inline]
    fn veil_point(di: usize) -> Self {
        BorderTraits::VEIL_POINT | Self::from_bits_truncate(1 << (di + Self::VEIL_POINT_START_BIT))
    }
}

impl Data for BorderTraits {
    type Data = u32;

    #[inline]
    fn as_slice(&self) -> &[Self::Data] {
        slice::from_ref(&self.bits)
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Data] {
        slice::from_mut(&mut self.bits)
    }

    #[inline]
    fn is_finite(&self) -> bool {
        true
    }
}

impl DataFields for BorderTraits {
    type Iter = array::IntoIter<FieldInfo, 1>;

    #[inline]
    fn fields() -> Self::Iter {
        [FieldInfo::single::<u32>("border_traits", 0)].into_iter()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct Border<T> {
    pub min_border_probability: T,
    pub radius_plane_extraction: usize,
    pub radius_borders: usize,
}

impl<T> Border<T> {
    pub fn new(
        min_border_probability: T,
        radius_plane_extraction: usize,
        radius_borders: usize,
    ) -> Self {
        Border {
            min_border_probability,
            radius_plane_extraction,
            radius_borders,
        }
    }

    pub fn surface<P>(&self, input: &RangeImage<P>) -> Option<Vec<SurfaceInfo<T>>>
    where
        T: RealField + ToPrimitive,
        P: Sync + PointRange<Data = T>,
    {
        if input.len() < 3 {
            return None;
        }

        let step = (self.radius_plane_extraction / 2).max(1);
        let num_neighbors = {
            let t = self.radius_plane_extraction / step + 1;
            t * t
        };
        let mut surface = Vec::new();
        let iter = input.par_iter().enumerate().map(|(index, point)| {
            let x = index % input.width();
            let y = index / input.width();
            let surface_info = input.surface_info(
                (x, y),
                self.radius_plane_extraction,
                step,
                point.coords(),
                num_neighbors,
                true,
            );
            surface_info.unwrap()
        });
        iter.collect_into_vec(&mut surface);

        Some(surface)
    }

    pub fn border_score<P>(
        &self,
        input: &RangeImage<P>,
        (x, y): (usize, usize),
        delta: (isize, isize),
        point: &P,
        surface: &SurfaceInfo<T>,
    ) -> Option<T>
    where
        T: RealField + Default,
        P: Sync + PointRange<Data = T> + Centroid<Result = P>,

        <P as Centroid>::Accumulator: Default,
    {
        let centroid = input.centroid_1d((x, y), delta, self.radius_borders)?;
        Some(if centroid.range().is_finite() {
            let distance_squared = (centroid.coords() - point.coords()).norm_squared();
            if distance_squared <= surface.max_neighbor_distance_squared {
                T::zero()
            } else {
                let score = distance_squared.sqrt();
                if centroid.range() < point.range() {
                    -score
                } else {
                    score
                }
            }
        } else if centroid.range() < T::zero() {
            T::zero()
        } else {
            T::one()
        })
    }

    fn adjust_scores<P>(&self, scores: &[T], input: &RangeImage<P>, mut storage: Vec<T>) -> Vec<T>
    where
        T: RealField + Default,
        P: Sync + PointRange<Data = T>,
    {
        let iter = scores.par_iter().enumerate().map(|(index, score)| {
            let bonus = convert::<_, T>(0.5);
            if score.clone() + bonus.clone() * (T::one() - score.clone())
                < self.min_border_probability
            {
                return score.clone();
            }
            let addend = {
                let [x, y] = input.index(index);
                let xmin = x.saturating_sub(1);
                let xmax = (x + 1).min(input.width() - 1);
                let ymin = y.saturating_sub(1);
                let ymax = (y + 1).min(input.height() - 1);

                let mut score = T::zero();
                for x in xmin..=xmax {
                    for y in ymin..=ymax {
                        score += scores[y * input.width() + x].clone()
                    }
                }
                score / convert::<_, T>(((xmax - xmin + 1) * (ymax - ymin + 1)) as f64)
            };
            score.clone() + bonus * addend * (T::one() - score.clone().abs())
        });

        iter.collect_into_vec(&mut storage);
        storage
    }

    const OFFSET: [(isize, isize); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

    pub fn border_scores<P>(
        &self,
        input: &RangeImage<P>,
        surface: &[SurfaceInfo<T>],
    ) -> Option<[Vec<T>; 4]>
    where
        T: RealField + Default,
        P: Sync + PointRange<Data = T> + Centroid<Result = P>,
        <P as Centroid>::Accumulator: Default,
    {
        let [mut top, mut right, mut bottom, mut left] =
            array::from_fn(|_| vec![Default::default(); input.len()]);
        let iter = input.par_iter().enumerate().zip((
            surface.par_iter(),
            top.par_iter_mut(),
            right.par_iter_mut(),
            bottom.par_iter_mut(),
            left.par_iter_mut(),
        ));
        iter.try_for_each(|((index, point), (surface, top, right, bottom, left))| {
            let [x, y] = input.index(index);

            *top = self.border_score(input, (x, y), Self::OFFSET[0], point, surface)?;
            *right = self.border_score(input, (x, y), Self::OFFSET[1], point, surface)?;
            *bottom = self.border_score(input, (x, y), Self::OFFSET[2], point, surface)?;
            *left = self.border_score(input, (x, y), Self::OFFSET[3], point, surface)?;

            Some(())
        })?;

        let t1 = self.adjust_scores(&top, input, Vec::new());
        let r1 = self.adjust_scores(&right, input, top);
        let b1 = self.adjust_scores(&bottom, input, right);
        let l1 = self.adjust_scores(&left, input, bottom);

        Some([t1, r1, b1, l1])
    }

    pub fn shadow_indices<P>(
        &self,
        input: &RangeImage<P>,
        border_scores: &mut [Vec<T>; 4],
    ) -> Vec<[Option<usize>; 4]>
    where
        T: RealField,
        P: PointRange<Data = T>,
    {
        let iter = (0..input.len()).map(|index| {
            let [x, y] = input.index(index);
            array::from_fn(|di| {
                let opposite = &border_scores[(di + 2) % 4];
                let score = border_scores[di][index].clone();
                if score < self.min_border_probability {
                    return None;
                }

                let (ox, oy) = Self::OFFSET[di];
                if score == T::one() {
                    let x = {
                        let x = (x as isize) + ox;
                        if !(0..(input.width() as isize)).contains(&x) {
                            return None;
                        }
                        x as usize
                    };
                    let y = {
                        let y = (y as isize) + oy;
                        if !(0..(input.height() as isize)).contains(&y) {
                            return None;
                        }
                        y as usize
                    };
                    let range = input[(x, y)].range();
                    if !range.is_finite() && range > T::zero() {
                        return Some(y * input.width() + x);
                    }
                }

                let (mul, ret) = {
                    let iter = (1..=self.radius_borders)
                        .map(|index| {
                            (
                                (x as isize) + (index as isize) * ox,
                                (y as isize) + (index as isize) * oy,
                            )
                        })
                        .map_while(|(x, y)| {
                            ((0..(input.width() as isize)).contains(&x)
                                && (0..(input.height() as isize)).contains(&y))
                            .then_some((x as usize, y as usize))
                        });
                    iter.fold((T::zero(), None), |(acc, old), (x, y)| {
                        let index = y * input.width() + x;
                        let opposite = opposite[index].clone();
                        if opposite < acc {
                            (opposite, Some(index))
                        } else {
                            (acc, old)
                        }
                    })
                };
                if ret.is_some() {
                    border_scores[di][index] *=
                        (T::one() - (mul + T::one()).powi(3)).max(convert(0.9));
                    if border_scores[di][index] >= self.min_border_probability {
                        return ret;
                    }
                }

                border_scores[di][index] = T::zero();
                None
            })
        });

        iter.collect()
    }

    fn check_maximum(
        &self,
        (width, height): (usize, usize),
        [x, y]: [usize; 2],
        (ox, oy): (isize, isize),
        border_scores: &[T],
        shadow_index: usize,
    ) -> bool
    where
        T: RealField,
    {
        let border_score = border_scores[y * width + x].clone();

        let iter = (1..=self.radius_borders)
            .map(|index| {
                (
                    (x as isize) + (index as isize) * ox,
                    (y as isize) + (index as isize) * oy,
                )
            })
            .map_while(|(x, y)| {
                ((0..(width as isize)).contains(&x) && (0..(height as isize)).contains(&y))
                    .then_some((x as usize, y as usize))
            });
        for (x, y) in iter {
            let index = y * width + x;
            if index == shadow_index {
                return true;
            }
            if border_scores[index] > border_score {
                return false;
            }
        }
        true
    }
}

impl<'a, T, P> Feature<&'a RangeImage<P>, Option<PointCloud<BorderTraits>>, (), ()>
    for Border<T>
where
    T: RealField + ToPrimitive + Default,
    P: Sync + PointRange<Data = T> + Centroid<Result = P>,
    <P as Centroid>::Accumulator: Default,
{
    fn compute(&self, input: &'a RangeImage<P>, _: (), _: ()) -> Option<PointCloud<BorderTraits>> {
        let surface = self.surface(input)?;
        let mut border_scores = self.border_scores(input, &surface)?;
        let shadow_indices = self.shadow_indices(input, &mut border_scores);

        let mut storage = vec![Default::default(); input.len()];
        for index in 0..input.len() {
            if !shadow_indices[index].iter().any(|x| x.is_some()) {
                continue;
            }

            let iter = shadow_indices[index]
                .into_iter()
                .enumerate()
                .filter_map(|(di, si)| {
                    si.and_then(|si| {
                        self.check_maximum(
                            (input.width(), input.height()),
                            input.index(index),
                            Self::OFFSET[di],
                            &border_scores[di],
                            si,
                        )
                        .then_some((di, si))
                    })
                });
            for (di, shadow_index) in iter {
                storage[index] |= BorderTraits::obstacle_border(di);
                let oi = (di + 2) % 4;
                storage[shadow_index] |= BorderTraits::shadow_border(oi);
                for veil_point in storage.iter_mut().take(index).skip(shadow_index + 1) {
                    *veil_point |= BorderTraits::veil_point(oi)
                }
            }
        }
        Some(unsafe { PointCloud::from_raw_parts(storage, input.width(), true) })
    }
}

#[cfg(test)]
mod tests {
    use std::{array, convert::identity};

    use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator};

    #[test]
    fn test_par_iter() {
        let orig: [_; 20] = array::from_fn(identity);

        let mut vec = Vec::new();
        orig.into_par_iter().collect_into_vec(&mut vec);
        // Check for isotonicity
        assert_eq!(&vec, &orig);
    }
}
