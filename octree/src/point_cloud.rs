use std::{
    array,
    ops::{Deref, DerefMut},
};

use nalgebra::{ComplexField, RealField, Scalar, Vector4};
use num::ToPrimitive;
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed};

use crate::OcTree;

#[derive(Debug)]
pub struct OcTreePc<L, T: Scalar> {
    pub(crate) inner: OcTree<L>,
    pub(crate) mul: T,
    pub(crate) add: Vector4<T>,
    bound: (Vector4<T>, Vector4<T>),
}

impl<L, T: Scalar + num::Zero> Default for OcTreePc<L, T> {
    fn default() -> Self {
        OcTreePc {
            inner: OcTree::new(1),
            mul: T::zero(),
            add: Vector4::zeros(),
            bound: (Vector4::zeros(), Vector4::zeros()),
        }
    }
}

pub struct CreateOptions<T> {
    pub resolution: T,
    pub bound: Option<(Vector4<T>, Vector4<T>)>,
}

impl<L, T: RealField + ToPrimitive> OcTreePc<L, T> {
    pub fn new<I, F>(
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
        options: CreateOptions<T>,
        build: F,
    ) -> Self
    where
        F: FnOnce(&mut OcTree<L>, T, &Vector4<T>),
    {
        let (min, max) = match options.bound.or_else(|| point_cloud.finite_bound()) {
            Some(bound) => bound,
            None => return Default::default(),
        };

        let mul = options.resolution;
        let len = &max - &min;

        let depth = ComplexField::ceil(ComplexField::log2((len / mul.clone()).xyz().max()))
            .to_usize()
            .expect("Failed to get the depth of the OC tree");

        let max_value = if depth >= 1 { (1 << depth) - 1 } else { 0 };

        let add = {
            let center_value = T::from_usize(max_value / 2).unwrap();
            let center_key = Vector4::from([
                center_value.clone(),
                center_value.clone(),
                center_value,
                T::one(),
            ]);
            let center = (&max + &min) / (T::one() + T::one());
            center - center_key
        };

        let mut inner = OcTree::new(depth);
        build(&mut inner, mul.clone(), &add);

        OcTreePc {
            inner,
            mul,
            add,
            bound: (min, max),
        }
    }
}

impl<L, T: Scalar> Deref for OcTreePc<L, T> {
    type Target = OcTree<L>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<L, T: Scalar> DerefMut for OcTreePc<L, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub(crate) fn key_to_coords<T: ComplexField>(
    key: &[usize; 3],
    mul: T,
    add: &Vector4<T>,
) -> Vector4<T> {
    let key = Vector4::from([
        T::from_usize(key[0]).unwrap(),
        T::from_usize(key[1]).unwrap(),
        T::from_usize(key[2]).unwrap(),
        T::zero(),
    ]);
    let mut result = key * mul + add;
    result.w = T::one();
    result
}

pub(crate) fn coords_to_key<T: ComplexField + ToPrimitive>(
    coords: &Vector4<T>,
    mul: T,
    add: &Vector4<T>,
) -> [usize; 3] {
    let key = (coords - add) * mul;
    let mut iter = key.into_iter().filter_map(|v| v.to_usize());
    array::from_fn(|_| iter.next().unwrap())
}

impl<L, T: ComplexField + Copy> OcTreePc<L, T> {
    pub fn key_to_coords(&self, key: &[usize; 3]) -> Vector4<T> {
        assert!(key.iter().all(|&v| v <= self.inner.max_key()));
        key_to_coords(key, self.mul, &self.add)
    }
}

impl<L, T: RealField + ToPrimitive + Copy> OcTreePc<L, T> {
    pub fn coords_to_key(&self, coords: &Vector4<T>) -> [usize; 3] {
        assert!(&self.bound.0 <= coords && coords <= &self.bound.1);
        coords_to_key(coords, self.mul, &self.add)
    }
}

impl<L, T: ComplexField + Copy> OcTreePc<L, T> {
    pub fn side(&self, depth: usize) -> T {
        self.mul * T::from_usize((self.inner.max_key() + 1) >> depth).unwrap()
    }

    pub fn diagonal(&self, depth: usize) -> T {
        self.side(depth) * T::from_usize(3).unwrap().sqrt()
    }

    pub fn center(&self, key: &[usize; 3], depth: usize) -> Vector4<T> {
        let radius = self.side(depth) / (T::one() + T::one());
        let coords = self.key_to_coords(key);
        let mut ret = coords.map(|v| v + radius);
        ret.w = T::one();
        ret
    }
}
