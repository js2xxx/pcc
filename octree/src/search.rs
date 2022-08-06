use std::{array, ops::Deref};

use nalgebra::{ComplexField, Scalar, Vector4};
use num::{Float, ToPrimitive};
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed, search::SearchType};

use crate::OcTree;

pub struct OcTreePCSearcher<'a, T: Scalar> {
    inner: OcTree<Vec<&'a Vector4<T>>>,
    mul: Vector4<T>,
    add: Vector4<T>,
}

pub struct CreateOptions<T> {
    pub resolution: T,
}

impl<'a, T: Scalar + num::Zero> Default for OcTreePCSearcher<'a, T> {
    fn default() -> Self {
        OcTreePCSearcher {
            inner: OcTree::new(1),
            mul: Vector4::zeros(),
            add: Vector4::zeros(),
        }
    }
}

pub fn key_to_coords<T: Scalar + ComplexField<RealField = T>>(
    key: &[usize; 3],
    mul: &Vector4<T>,
    add: &Vector4<T>,
) -> Vector4<T> {
    let key = Vector4::from([
        T::from_usize(key[0]).unwrap(),
        T::from_usize(key[1]).unwrap(),
        T::from_usize(key[2]).unwrap(),
        T::zero(),
    ]);
    let mut result = key.component_mul(mul) + add;
    result.w = T::one();
    result
}

pub fn coords_to_key<T: Scalar + ComplexField<RealField = T> + ToPrimitive>(
    coords: &Vector4<T>,
    mul: &Vector4<T>,
    add: &Vector4<T>,
) -> [usize; 3] {
    let key = (coords - add).component_div(mul);
    let mut iter = key.into_iter().filter_map(|v| v.to_usize());
    array::from_fn(|_| iter.next().unwrap())
}

impl<'a, T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy> OcTreePCSearcher<'a, T> {
    pub fn voxel_search<'b>(&'b self, pivot: &Vector4<T>) -> &'b [&'a Vector4<T>] {
        let key = coords_to_key(pivot, &self.mul, &self.add);
        self.inner.get(&key).map_or(&[], Deref::deref)
    }
}

impl<'a, T: Scalar + Float + ComplexField<RealField = T>> pcc_common::search::Searcher<'a, T>
    for OcTreePCSearcher<'a, T>
{
    type FromExtra = CreateOptions<T>;
    fn from_point_cloud<I>(
        point_cloud: &'a PointCloud<Point3Infoed<T, I>>,
        options: CreateOptions<T>,
    ) -> Self {
        let (min, max) = match point_cloud.finite_bound() {
            Some(bound) => bound,
            None => return Default::default(),
        };

        let resolution = options.resolution;
        let len = max - min;

        let depth = Float::ceil(Float::log2((len / resolution).xyz().max()))
            .to_usize()
            .expect("Failed to get the depth of the OC tree");

        let max_value = if depth >= 1 { (1 << depth) - 1 } else { 0 };

        let add = {
            let center_value = T::from_usize(max_value / 2).unwrap();
            let center_key = Vector4::from([center_value, center_value, center_value, T::one()]);
            let center = (max + min) / (T::one() + T::one());
            center - center_key
        };

        let mul = {
            let max_value = T::from_usize(max_value).unwrap();
            let max_key = Vector4::from([max_value, max_value, max_value, T::one()]);
            len.component_div(&max_key)
        };

        let mut inner = OcTree::new(depth);
        for point in point_cloud.iter() {
            let key = coords_to_key(&point.coords, &mul, &add);
            let vec = inner.get_or_insert_with(&key, Vec::new);
            vec.push(&point.coords);
        }

        OcTreePCSearcher { inner, mul, add }
    }

    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<&'a Vector4<T>>) {
        todo!()
    }
}
