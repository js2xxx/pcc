use std::{array, fmt::Debug};

use nalgebra::{RealField, Scalar};
use pcc_common::{filter::ApproxFilter, point_cloud::PointCloud, points::Point3Infoed};

pub struct Median2<T: Scalar> {
    pub window: isize,
    pub max_displacement: T,
}

impl<T: Scalar> Median2<T> {
    pub fn new(window: isize, max_displacement: T) -> Self {
        Median2 {
            window,
            max_displacement,
        }
    }
}

impl<T: RealField, I: Debug + Clone> ApproxFilter<PointCloud<Point3Infoed<T, I>>> for Median2<T> {
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        let mut pc = input.clone();

        let mut values: [_; 9] = array::from_fn(|_| T::zero());
        let mut value_index;
        for x in 0..input.width() {
            for y in 0..input.height() {
                value_index = 0;

                for dx in (-self.window / 2)..=(self.window / 2) {
                    for dy in (-self.window / 2)..=(self.window / 2) {
                        let x = x as isize + dx;
                        let y = y as isize + dy;
                        if 0 <= x
                            && x as usize <= input.width()
                            && y >= 0
                            && y as usize <= input.height()
                        {
                            values[value_index] = input[(x as usize, y as usize)].coords.z.clone();
                            value_index += 1;
                        }
                    }
                }

                let (_, median, _) = values[0..value_index]
                    .select_nth_unstable_by(value_index / 2, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });

                if input[(x, y)].coords.z.clone() - median.clone() <= self.max_displacement {
                    pc[(x, y)].coords.z = median.clone()
                } else {
                    pc[(x, y)].coords.z += { self.max_displacement.clone() }
                        .copysign(median.clone() - input[(x, y)].coords.z.clone())
                }
            }
        }

        pc
    }
}
