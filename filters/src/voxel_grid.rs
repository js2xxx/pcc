use nalgebra::{ComplexField, Scalar, Vector4};
use num::ToPrimitive;
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed};

pub struct VoxelGrid<T: Scalar> {
    grid_unit: Vector4<T>,
}

impl<T: Scalar> VoxelGrid<T> {
    pub fn new(grid_unit: Vector4<T>) -> Self {
        VoxelGrid { grid_unit }
    }
}

impl<T: Scalar + ComplexField<RealField = T> + PartialOrd + ToPrimitive> VoxelGrid<T> {
    pub fn filter<I: std::fmt::Debug + Default>(
        &self,
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
                    (*index.xyz().as_ref(), coords)
                })
                .collect::<Vec<_>>()
        } else {
            { point_cloud.iter().filter(|point| point.is_finite()) }
                .map(|point| {
                    let coords = &point.coords;
                    let index = (coords - &min)
                        .component_div(&self.grid_unit)
                        .map(|x| x.floor().to_usize().unwrap());
                    (*index.xyz().as_ref(), coords)
                })
                .collect::<Vec<_>>()
        };

        index_point.sort_by(|(i1, _), (i2, _)| i1.cmp(i2));

        let mut temp_vec = Vector4::zeros();
        let mut temp_num = 0;
        let mut last_index = [0; 3];
        let mut storage = Vec::with_capacity(index_point.len() / 3);

        for (index, coords) in index_point {
            if index != last_index {
                last_index = index;
                let centroid = temp_vec.unscale(T::from_usize(temp_num).unwrap());
                storage.push(Point3Infoed {
                    coords: centroid,
                    extra: Default::default(),
                });

                temp_vec = Vector4::zeros();
                temp_num = 0;
            }

            temp_vec += coords;
            temp_num += 1;
        }
        let centroid = temp_vec.unscale(T::from_usize(temp_num).unwrap());
        storage.push(Point3Infoed {
            coords: centroid,
            extra: Default::default(),
        });

        PointCloud::from_vec(storage, 1)
    }
}
