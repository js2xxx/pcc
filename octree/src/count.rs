use nalgebra::{RealField, Scalar, Vector4};
use num::{Float, ToPrimitive};
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed};

use crate::{point_cloud::coords_to_key, CreateOptions, OcTreePc};

#[derive(Debug)]
pub struct OcTreePcCount<T: Scalar> {
    inner: OcTreePc<usize, T>,
}

impl<T: Scalar + num::Zero> Default for OcTreePcCount<T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T: Float + RealField> OcTreePcCount<T> {
    pub fn from_point_cloud<I>(
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
        options: CreateOptions<T>,
    ) -> Self {
        OcTreePcCount {
            inner: OcTreePc::new(point_cloud, options, |tree, mul, add| {
                for point in point_cloud.iter() {
                    let key = coords_to_key(&point.coords, mul, add);
                    *tree.get_or_insert(&key, 0) += 1;
                }
            }),
        }
    }
}

impl<T: RealField + ToPrimitive + Copy> OcTreePcCount<T> {
    pub fn count_at(&self, coords: &Vector4<T>) -> Option<usize> {
        let key = self.inner.coords_to_key(coords);
        self.inner.get(&key).copied()
    }
}
