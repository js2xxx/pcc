use std::{array, ops::Deref};

use nalgebra::{ComplexField, Scalar, Vector4};
use num::{Float, ToPrimitive};
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed, search::SearchType};

use crate::{
    node::{key_child, Node},
    OcTree,
};

pub struct OcTreePCSearcher<'a, T: Scalar> {
    inner: OcTree<Vec<&'a Vector4<T>>>,
    mul: T,
    add: Vector4<T>,
}

pub struct CreateOptions<T> {
    pub resolution: T,
}

impl<'a, T: Scalar + num::Zero> Default for OcTreePCSearcher<'a, T> {
    fn default() -> Self {
        OcTreePCSearcher {
            inner: OcTree::new(1),
            mul: T::zero(),
            add: Vector4::zeros(),
        }
    }
}

fn key_to_coords<T: Scalar + ComplexField<RealField = T>>(
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
    let mut result = key.scale(mul) + add;
    result.w = T::one();
    result
}

fn coords_to_key<T: Scalar + ComplexField<RealField = T> + ToPrimitive>(
    coords: &Vector4<T>,
    mul: T,
    add: &Vector4<T>,
) -> [usize; 3] {
    let key = (coords - add).scale(mul);
    let mut iter = key.into_iter().filter_map(|v| v.to_usize());
    array::from_fn(|_| iter.next().unwrap())
}

impl<'a, T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy> OcTreePCSearcher<'a, T> {
    pub fn key_to_coords(&self, key: &[usize; 3]) -> Vector4<T> {
        key_to_coords(key, self.mul, &self.add)
    }

    pub fn coords_to_key(&self, coords: &Vector4<T>) -> [usize; 3] {
        coords_to_key(coords, self.mul, &self.add)
    }

    fn side(&self, depth: usize) -> T {
        self.mul * T::from_usize((self.inner.max_key() + 1) >> depth).unwrap()
    }

    fn diagonal(&self, depth: usize) -> T {
        self.side(depth) * T::from_usize(3).unwrap().sqrt()
    }

    fn half_diagonal(&self, depth: usize) -> T {
        self.diagonal(depth) / (T::one() + T::one())
    }

    fn center(&self, key: &[usize; 3], depth: usize) -> Vector4<T> {
        let radius = self.side(depth) / (T::one() + T::one());
        let coords = self.key_to_coords(key);
        let mut ret = coords.map(|v| v + radius);
        ret.w = T::one();
        ret
    }
}

impl<'a, T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy> OcTreePCSearcher<'a, T> {
    pub fn voxel_search<'b>(&'b self, pivot: &Vector4<T>) -> &'b [&'a Vector4<T>] {
        let key = self.coords_to_key(pivot);
        self.inner.get(&key).map_or(&[], Deref::deref)
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeKey<'b, 'a, T: Scalar> {
    node: &'b Node<(), Vec<&'a Vector4<T>>>,
    key: [usize; 3],
}

impl<'a, T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy + PartialOrd>
    OcTreePCSearcher<'a, T>
{
    fn knn_search(&self, pivot: &Vector4<T>, num: usize, result_set: &mut Vec<&'a Vector4<T>>) {
        let mut rs = Vec::new();
        if let Some(node) = self.inner.root() {
            self.knn_search_recursive(&NodeKey { node, key: [0; 3] }, pivot, num, 1, None, &mut rs);
        }
        result_set.clear();
        result_set.extend(rs.into_iter().map(|(coords, ..)| coords));
    }

    fn knn_search_recursive(
        &self,
        node_key: &NodeKey<'_, 'a, T>,
        pivot: &Vector4<T>,
        num: usize,
        depth: usize,
        mut min_distance: Option<T>,
        result_set: &mut Vec<(&'a Vector4<T>, T)>,
    ) -> Option<T> {
        let half_diagonal = self.half_diagonal(depth);

        let children = match node_key.node {
            Node::Leaf { .. } => panic!("Leaf node with no parent cannot be searched directly"),
            Node::Branch { children, .. } => children,
        };

        let mut search_heap = { children.iter().enumerate() }
            .filter_map(|(index, child)| {
                child.map(|child| {
                    let child_nk = NodeKey {
                        node: unsafe { child.as_ref() },
                        key: key_child(&node_key.key, index),
                    };
                    let center = self.center(&child_nk.key, depth);
                    let distance = (center - pivot).norm();
                    (child_nk, distance)
                })
            })
            .collect::<Vec<_>>();
        search_heap.sort_by(|(nk1, d1), (nk2, d2)| {
            use std::cmp::Ordering;
            match d1.partial_cmp(d2) {
                Some(Ordering::Equal) | None => {}
                Some(ord) => return ord,
            }
            nk1.key.cmp(&nk2.key)
        });

        for (child, distance) in search_heap {
            if let Some(min_distance) = min_distance {
                if distance > min_distance + half_diagonal {
                    break;
                }
            }

            match child.node {
                Node::Branch { .. } => {
                    min_distance = self.knn_search_recursive(
                        &child,
                        pivot,
                        num,
                        depth + 1,
                        min_distance,
                        result_set,
                    )
                }
                Node::Leaf { content } => {
                    for &coords in content {
                        let distance = (coords - pivot).norm();
                        if min_distance.map_or(true, |d| distance < d) {
                            result_set.push((coords, distance));
                        }
                    }

                    result_set.sort_by(|(_, d1), (_, d2)| {
                        d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if result_set.len() > num {
                        result_set.truncate(num);
                    }
                    if result_set.len() == num {
                        min_distance = Some(result_set.last().unwrap().1);
                    }
                }
            }
        }

        min_distance
    }
}

impl<'a, T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy + PartialOrd>
    OcTreePCSearcher<'a, T>
{
    fn radius_search(&self, pivot: &Vector4<T>, radius: T, result_set: &mut Vec<&'a Vector4<T>>) {
        result_set.clear();
        if let Some(node) = self.inner.root() {
            self.radius_search_recursive(
                &NodeKey { node, key: [0; 3] },
                pivot,
                radius,
                1,
                result_set,
            );
        }
    }

    fn radius_search_recursive(
        &self,
        node_key: &NodeKey<'_, 'a, T>,
        pivot: &Vector4<T>,
        radius: T,
        depth: usize,
        result_set: &mut Vec<&'a Vector4<T>>,
    ) {
        let half_diagonal = self.half_diagonal(depth);

        let children = match node_key.node {
            Node::Leaf { .. } => panic!("Leaf node with no parent cannot be searched directly"),
            Node::Branch { children, .. } => children,
        };

        for child in children.iter().enumerate().filter_map(|(index, child)| {
            child.and_then(|child| {
                let child_nk = NodeKey {
                    node: unsafe { child.as_ref() },
                    key: key_child(&node_key.key, index),
                };
                let center = self.center(&child_nk.key, depth);
                let distance = (center - pivot).norm();
                (distance <= radius + half_diagonal).then_some(child_nk)
            })
        }) {
            match child.node {
                Node::Branch { .. } => {
                    self.radius_search_recursive(&child, pivot, radius, depth + 1, result_set)
                }
                Node::Leaf { content } => {
                    for &coords in content {
                        let distance = (coords - pivot).norm();
                        if distance <= radius {
                            result_set.push(coords)
                        }
                    }
                }
            }
        }
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

        let mul = options.resolution;
        let len = max - min;

        let depth = Float::ceil(Float::log2((len / mul).xyz().max()))
            .to_usize()
            .expect("Failed to get the depth of the OC tree");

        let max_value = if depth >= 1 { (1 << depth) - 1 } else { 0 };

        let add = {
            let center_value = T::from_usize(max_value / 2).unwrap();
            let center_key = Vector4::from([center_value, center_value, center_value, T::one()]);
            let center = (max + min) / (T::one() + T::one());
            center - center_key
        };

        let mut inner = OcTree::new(depth);
        for point in point_cloud.iter() {
            let key = coords_to_key(&point.coords, mul, &add);
            let vec = inner.get_or_insert_with(&key, Vec::new);
            vec.push(&point.coords);
        }

        OcTreePCSearcher { inner, mul, add }
    }

    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<&'a Vector4<T>>) {
        match ty {
            SearchType::Knn(num) => self.knn_search(pivot, num, result),
            SearchType::Radius(radius) => self.radius_search(pivot, radius, result),
        }
    }
}
