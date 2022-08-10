use std::fmt::Debug;

use nalgebra::{RealField, Scalar, Vector3, Vector4};
use num::ToPrimitive;
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point_cloud::PointCloud,
    points::Point3Infoed,
};

pub struct UniformSampling<T: Scalar> {
    pub grid_unit: Vector4<T>,
}

impl<T: Scalar> UniformSampling<T> {
    pub fn new(grid_unit: Vector4<T>) -> Self {
        UniformSampling { grid_unit }
    }
}

impl<T: RealField + ToPrimitive> UniformSampling<T> {
    #[allow(clippy::type_complexity)]
    fn filter_data<'a, I>(
        grid_unit: &Vector4<T>,
        input: &'a PointCloud<Point3Infoed<T, I>>,
    ) -> Option<(Vector4<T>, Vec<([usize; 3], &'a Point3Infoed<T, I>)>)> {
        let (min, _) = match input.finite_bound() {
            Some(bound) => bound,
            None => return None,
        };
        let bounded = input.is_bounded();
        let mut key_point = if bounded {
            { input.iter() }
                .map(|point| {
                    let coords = &point.coords;
                    let key = (coords - &min)
                        .component_div(grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*key.xyz().as_ref(), point)
                })
                .collect::<Vec<_>>()
        } else {
            { input.iter().filter(|point| point.is_finite()) }
                .map(|point| {
                    let coords = &point.coords;
                    let key = (coords - &min)
                        .component_div(grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*key.xyz().as_ref(), point)
                })
                .collect::<Vec<_>>()
        };
        key_point.sort_by(|(i1, _), (i2, _)| i1.cmp(i2));
        Some((min, key_point))
    }

    fn filter_inner<I, F1, F2>(
        &self,
        min: &Vector4<T>,
        key_point: Vec<([usize; 3], &Point3Infoed<T, I>)>,
        mut push_point: F1,
        mut push_removed: F2,
    ) where
        F1: FnMut(usize, &Point3Infoed<T, I>),
        F2: FnMut(usize),
    {
        let get_center = |index: [usize; 3]| {
            Vector3::from(index)
                .map(|x| T::from_usize(x).unwrap())
                .insert_row(3, T::zero())
                .component_mul(&self.grid_unit)
                + min
        };

        let mut nearest = None;
        let mut last_index = [0; 3];
        let mut center = get_center(last_index);

        for (index, (key, point)) in key_point.into_iter().enumerate() {
            if key != last_index {
                last_index = key;
                center = get_center(key);

                if let Some((index, point, _)) = nearest.take() {
                    push_point(index, point);
                }
            }
            let distance = (&center - &point.coords).norm();

            nearest = match nearest {
                Some((i, _, d)) if distance < d => {
                    push_removed(i);
                    Some((index, point, distance))
                }
                _ => nearest,
            };
        }
        if let Some((index, point, _)) = nearest.take() {
            push_point(index, point);
        }
    }
}

impl<T: RealField + ToPrimitive, I> Filter<PointCloud<Point3Infoed<T, I>>> for UniformSampling<T> {
    fn filter_indices(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> Vec<usize> {
        let (min, key_point) = match Self::filter_data(&self.grid_unit, input) {
            Some(value) => value,
            None => return Vec::new(),
        };

        let mut indices = Vec::with_capacity(key_point.len() / 3);

        self.filter_inner(&min, key_point, |index, _| indices.push(index), |_| {});

        indices
    }

    fn filter_all_indices(
        &mut self,
        input: &PointCloud<Point3Infoed<T, I>>,
    ) -> (Vec<usize>, Vec<usize>) {
        let (min, key_point) = match Self::filter_data(&self.grid_unit, input) {
            Some(value) => value,
            None => return (Vec::new(), Vec::new()),
        };

        let mut indices = Vec::with_capacity(key_point.len() / 3);
        let mut removed = Vec::with_capacity(indices.capacity());

        self.filter_inner(
            &min,
            key_point,
            |index, _| indices.push(index),
            |index| removed.push(index),
        );

        (indices, removed)
    }
}

impl<T: RealField + ToPrimitive, I: Clone + Debug> ApproxFilter<PointCloud<Point3Infoed<T, I>>>
    for UniformSampling<T>
{
    fn filter(&mut self, input: &PointCloud<Point3Infoed<T, I>>) -> PointCloud<Point3Infoed<T, I>> {
        let (min, key_point) = match Self::filter_data(&self.grid_unit, input) {
            Some(value) => value,
            None => return PointCloud::new(),
        };

        let mut storage = Vec::with_capacity(key_point.len() / 3);

        self.filter_inner(
            &min,
            key_point,
            |_, point| storage.push(point.clone()),
            |_| {},
        );

        PointCloud::from_vec(storage, 1)
    }
}
