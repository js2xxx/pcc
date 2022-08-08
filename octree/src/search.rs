use std::ops::Deref;

use nalgebra::{ComplexField, Scalar, Vector4};
use num::{Float, ToPrimitive};
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed, search::SearchType};

use crate::{
    node::{key_child, Node},
    point_cloud::{CreateOptions, OcTreePc},
};

pub struct OcTreePcSearch<'a, T: Scalar> {
    inner: OcTreePc<Vec<&'a Vector4<T>>, T>,
}

impl<'a, T: Scalar + num::Zero> Default for OcTreePcSearch<'a, T> {
    fn default() -> Self {
        OcTreePcSearch {
            inner: Default::default(),
        }
    }
}

impl<'a, T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy + PartialOrd>
    OcTreePcSearch<'a, T>
{
    fn half_diagonal(&self, depth: usize) -> T {
        self.inner.diagonal(depth) / (T::one() + T::one())
    }
}

impl<'a, T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy + PartialOrd>
    OcTreePcSearch<'a, T>
{
    pub fn voxel_search<'b>(&'b self, pivot: &Vector4<T>) -> &'b [&'a Vector4<T>] {
        let key = self.inner.coords_to_key(pivot);
        self.inner.get(&key).map_or(&[], Deref::deref)
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeKey<'b, 'a, T: Scalar> {
    node: &'b Node<(), Vec<&'a Vector4<T>>>,
    key: [usize; 3],
}

impl<'a, T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy + PartialOrd>
    OcTreePcSearch<'a, T>
{
    pub fn knn_search(&self, pivot: &Vector4<T>, num: usize, result_set: &mut Vec<&'a Vector4<T>>) {
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
                    let center = self.inner.center(&child_nk.key, depth);
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
    OcTreePcSearch<'a, T>
{
    pub fn radius_search(
        &self,
        pivot: &Vector4<T>,
        radius: T,
        result_set: &mut Vec<&'a Vector4<T>>,
    ) {
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
                let center = self.inner.center(&child_nk.key, depth);
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
    for OcTreePcSearch<'a, T>
{
    type FromExtra = CreateOptions<T>;
    fn from_point_cloud<I>(
        point_cloud: &'a PointCloud<Point3Infoed<T, I>>,
        options: CreateOptions<T>,
    ) -> Self {
        OcTreePcSearch {
            inner: OcTreePc::from_point_cloud(point_cloud, options, |tree, mul, add| {
                for point in point_cloud.iter() {
                    let key = crate::point_cloud::coords_to_key(&point.coords, mul, add);
                    let vec = tree.get_or_insert_with(&key, Vec::new);
                    vec.push(&point.coords);
                }
            }),
        }
    }

    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<&'a Vector4<T>>) {
        match ty {
            SearchType::Knn(num) => self.knn_search(pivot, num, result),
            SearchType::Radius(radius) => self.radius_search(pivot, radius, result),
        }
    }
}
