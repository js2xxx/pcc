use std::collections::HashMap;

use nalgebra::{ComplexField, Scalar, Vector4};
use num::ToPrimitive;
use pcc_common::{
    filter::ApproxFilter,
    point_cloud::PointCloud,
    points::{Centroid, Point3Infoed},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VoxelGrid<T: Scalar> {
    grid_unit: Vector4<T>,
}

impl<T: Scalar> VoxelGrid<T> {
    pub fn new(grid_unit: Vector4<T>) -> Self {
        VoxelGrid { grid_unit }
    }
}

impl<
        T: Scalar + ComplexField<RealField = T> + PartialOrd + ToPrimitive + Centroid + Default,
        I: std::fmt::Debug + Default + Centroid,
    > ApproxFilter<PointCloud<Point3Infoed<T, I>>> for VoxelGrid<T>
where
    <I as Centroid>::Accumulator: Default,
{
    fn filter(
        &mut self,
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
    ) -> PointCloud<Point3Infoed<T, I>> {
        let (min, _) = match point_cloud.finite_bound() {
            Some(bound) => bound,
            None => return PointCloud::new(),
        };

        let bounded = point_cloud.is_bounded();

        let mut index_point = if bounded {
            { point_cloud.iter() }
                .map(|point| {
                    let coords = &point.coords;
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), point)
                })
                .collect::<Vec<_>>()
        } else {
            { point_cloud.iter().filter(|point| point.is_finite()) }
                .map(|point| {
                    let coords = &point.coords;
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), point)
                })
                .collect::<Vec<_>>()
        };

        index_point.sort_by(|(i1, _), (i2, _)| i1.cmp(i2));

        let mut centroid_builder = Centroid::default_builder();
        let mut last_index = [0; 3];
        let mut storage = Vec::with_capacity(index_point.len() / 3);

        for (index, coords) in index_point {
            if index != last_index {
                last_index = index;
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
    grid_unit: Vector4<T>,
}

impl<T: Scalar> HashVoxelGrid<T> {
    pub fn new(grid_unit: Vector4<T>) -> Self {
        HashVoxelGrid { grid_unit }
    }
}

impl<
        T: Scalar + ComplexField<RealField = T> + PartialOrd + ToPrimitive + Centroid + Default,
        I: std::fmt::Debug + Default + Centroid,
    > ApproxFilter<PointCloud<Point3Infoed<T, I>>> for HashVoxelGrid<T>
where
    <I as Centroid>::Accumulator: Default,
{
    fn filter(
        &mut self,
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
    ) -> PointCloud<Point3Infoed<T, I>> {
        let (min, _) = match point_cloud.finite_bound() {
            Some(bound) => bound,
            None => return PointCloud::new(),
        };

        let bounded = point_cloud.is_bounded();

        let fold = |mut map: HashMap<_, _>, (index, point)| {
            match map.try_insert(index, Centroid::default_builder()) {
                Ok(builder) => builder.accumulate(point),
                Err(mut e) => e.entry.get_mut().accumulate(point),
            }
            map
        };

        let index_point = if bounded {
            { point_cloud.iter() }
                .map(|point| {
                    let coords = &point.coords;
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), point)
                })
                .fold(HashMap::new(), fold)
        } else {
            { point_cloud.iter().filter(|point| point.is_finite()) }
                .map(|point| {
                    let coords = &point.coords;
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), point)
                })
                .fold(HashMap::new(), fold)
        };

        let storage = index_point
            .into_iter()
            .map(|(_, builder)| builder.compute().unwrap())
            .collect::<Vec<_>>();

        PointCloud::from_vec(storage, 1)
    }
}
