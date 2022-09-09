use std::{collections::HashMap, fmt::Debug};

use nalgebra::{RealField, Scalar, Vector2, Vector4};
use num::ToPrimitive;
use pcc_common::{
    filter::{ApproxFilter, Filter},
    point::{Centroid, Point},
    point_cloud::{AsPointCloud, PointCloud},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VoxelGrid<T: Scalar> {
    pub grid_unit: Vector4<T>,
}

impl<T: Scalar> VoxelGrid<T> {
    pub fn new(grid_unit: Vector4<T>) -> Self {
        VoxelGrid { grid_unit }
    }
}

impl<T, P> ApproxFilter<PointCloud<P>> for VoxelGrid<T>
where
    T: RealField + ToPrimitive + Centroid + Default,
    P: Point<Data = T> + Centroid<Result = P>,
    <P as Centroid>::Accumulator: Default,
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let [min, _] = match input.finite_bound() {
            Some(bound) => bound,
            None => return PointCloud::new(),
        };

        let bounded = input.is_bounded();

        let mut key_point = if bounded {
            { input.iter() }
                .map(|point| {
                    let coords = point.coords();
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), point)
                })
                .collect::<Vec<_>>()
        } else {
            { input.iter().filter(|point| point.is_finite()) }
                .map(|point| {
                    let coords = point.coords();
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), point)
                })
                .collect::<Vec<_>>()
        };

        key_point.sort_by(|(i1, _), (i2, _)| i1.cmp(i2));

        let mut centroid_builder = Centroid::default_builder();
        let mut last_key = [0; 3];
        let mut storage = Vec::with_capacity(key_point.len() / 3);

        for (key, coords) in key_point {
            if key != last_key {
                last_key = key;
                let centroid = centroid_builder.compute().unwrap();
                storage.push(centroid);

                centroid_builder = Centroid::default_builder();
            }

            centroid_builder.accumulate(coords);
        }
        let centroid = centroid_builder.compute().unwrap();
        storage.push(centroid);

        PointCloud::from_vec(storage, 1)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HashVoxelGrid<T: Scalar> {
    pub grid_unit: Vector4<T>,
}

impl<T: Scalar> HashVoxelGrid<T> {
    pub fn new(grid_unit: Vector4<T>) -> Self {
        HashVoxelGrid { grid_unit }
    }
}

impl<T, P> ApproxFilter<PointCloud<P>> for HashVoxelGrid<T>
where
    T: RealField + ToPrimitive + Centroid + Default,
    P: Point<Data = T> + Centroid<Result = P>,
    <P as Centroid>::Accumulator: Default,
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let [min, _] = match input.finite_bound() {
            Some(bound) => bound,
            None => return PointCloud::new(),
        };

        let bounded = input.is_bounded();

        let fold = |mut map: HashMap<_, _>, (index, point)| {
            match map.try_insert(index, Centroid::default_builder()) {
                Ok(builder) => builder.accumulate(point),
                Err(mut e) => e.entry.get_mut().accumulate(point),
            }
            map
        };

        let key_point = if bounded {
            { input.iter() }
                .map(|point| {
                    let coords = point.coords();
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), point)
                })
                .fold(HashMap::new(), fold)
        } else {
            { input.iter().filter(|point| point.is_finite()) }
                .map(|point| {
                    let coords = point.coords();
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), point)
                })
                .fold(HashMap::new(), fold)
        };

        let storage = key_point
            .into_iter()
            .map(|(_, builder)| builder.compute().unwrap())
            .collect::<Vec<_>>();

        PointCloud::from_vec(storage, 1)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GridMinimumZ<T: Scalar> {
    grid_unit: Vector2<T>,
}

impl<T: Scalar> GridMinimumZ<T> {
    pub fn new(grid_unit: Vector2<T>) -> Self {
        GridMinimumZ { grid_unit }
    }
}

impl<T: RealField + ToPrimitive> GridMinimumZ<T> {
    fn filter_data<P: Point<Data = T>>(
        &mut self,
        input: &PointCloud<P>,
    ) -> Option<Vec<([usize; 2], usize)>> {
        let [min, _] = match input.finite_bound() {
            Some(bound) => bound,
            None => return None,
        };
        let min = min.xy();
        let bounded = input.is_bounded();
        let mut key_index = if bounded {
            { input.iter().enumerate() }
                .map(|(index, point)| {
                    let coords = point.coords();
                    let key = (coords.xy() - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*key.as_ref(), index)
                })
                .collect::<Vec<_>>()
        } else {
            {
                input
                    .iter()
                    .enumerate()
                    .filter(|(_, point)| point.is_finite())
            }
            .map(|(index, point)| {
                let coords = point.coords();
                let key = (coords.xy() - &min)
                    .component_div(&self.grid_unit)
                    .map(|x| x.floor().to_usize().unwrap());
                (*key.as_ref(), index)
            })
            .collect::<Vec<_>>()
        };
        key_index.sort_by(|(i1, _), (i2, _)| i1.cmp(i2));
        Some(key_index)
    }

    fn filter_inner<P: Point<Data = T>, F1, F2>(
        &self,
        input: &PointCloud<P>,
        key_index: Vec<([usize; 2], usize)>,
        mut push_index: F1,
        mut push_removed: F2,
    ) where
        F1: FnMut(usize),
        F2: FnMut(usize),
    {
        let mut min = None;
        let mut last_key = [0; 2];

        for (key, index) in key_index {
            if key != last_key {
                last_key = key;
                if let Some((_, imin)) = min {
                    push_index(imin)
                }

                min = None;
            }

            min = match min {
                Some((value, i)) if value > input[index].coords().z => {
                    push_removed(i);
                    Some((input[index].coords().z.clone(), index))
                }
                _ => min,
            };
        }
        if let Some((_, imin)) = min {
            push_index(imin)
        }
    }
}

impl<T: RealField + ToPrimitive, P: Point<Data = T>> Filter<PointCloud<P>> for GridMinimumZ<T> {
    fn filter_indices(&mut self, input: &PointCloud<P>) -> Vec<usize> {
        let key_index = match self.filter_data(input) {
            Some(key_index) => key_index,
            None => return Vec::new(),
        };

        let mut indices = Vec::with_capacity(key_index.len() / 3);
        self.filter_inner(input, key_index, |index| indices.push(index), |_| {});

        indices
    }

    fn filter_all_indices(&mut self, input: &PointCloud<P>) -> (Vec<usize>, Vec<usize>) {
        let key_index = match self.filter_data(input) {
            Some(key_index) => key_index,
            None => return (Vec::new(), Vec::new()),
        };

        let mut indices = Vec::with_capacity(key_index.len() / 3);
        let mut removed = Vec::with_capacity(indices.len());
        self.filter_inner(
            input,
            key_index,
            |index| indices.push(index),
            |index| removed.push(index),
        );

        (indices, removed)
    }
}

impl<T: RealField + ToPrimitive, P: Point<Data = T>> ApproxFilter<PointCloud<P>>
    for GridMinimumZ<T>
{
    fn filter(&mut self, input: &PointCloud<P>) -> PointCloud<P> {
        let key_index = match self.filter_data(input) {
            Some(key_index) => key_index,
            None => return PointCloud::new(),
        };

        let mut storage = Vec::with_capacity(key_index.len() / 3);
        self.filter_inner(
            input,
            key_index,
            |index| storage.push(input[index].clone()),
            |_| {},
        );

        PointCloud::from_vec(storage, 1)
    }
}
