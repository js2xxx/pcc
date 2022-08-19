use std::ops::Deref;

use nalgebra::{RealField, Scalar, Vector4};
use num::{One, ToPrimitive};
use pcc_common::{point::Point, point_cloud::PointCloud, search::SearchType};

use crate::{
    node::{key_child, Node},
    point_cloud::{CreateOptions, OcTreePc},
};

type Item<'a, T> = (usize, &'a Vector4<T>);

pub struct OcTreePcSearch<'a, P: Point> {
    inner: OcTreePc<Vec<Item<'a, P::Data>>, P::Data>,
    point_cloud: &'a PointCloud<P>,
}

impl<'a, P: Point> OcTreePcSearch<'a, P>
where
    P::Data: RealField,
{
    fn half_diagonal(&self, depth: usize) -> P::Data {
        self.inner.diagonal(depth) / (P::Data::one() + P::Data::one())
    }
}

impl<'a, P: Point> OcTreePcSearch<'a, P>
where
    P::Data: RealField + ToPrimitive,
{
    pub fn voxel_search<'b>(
        &'b self,
        pivot: &Vector4<P::Data>,
    ) -> &'b [(usize, &'a Vector4<P::Data>)] {
        let key = self.inner.coords_to_key(pivot);
        self.inner.get(&key).map_or(&[], Deref::deref)
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeKey<'b, 'a, T: Scalar> {
    node: &'b Node<(), Vec<(usize, &'a Vector4<T>)>>,
    key: [usize; 3],
}

impl<'a, P: Point> OcTreePcSearch<'a, P>
where
    P::Data: RealField + ToPrimitive,
{
    pub fn knn_search(
        &self,
        pivot: &Vector4<P::Data>,
        num: usize,
        result_set: &mut Vec<(usize, P::Data)>,
    ) {
        let mut rs = Vec::new();
        if let Some(node) = self.inner.root() {
            self.knn_search_recursive(&NodeKey { node, key: [0; 3] }, pivot, num, 1, None, &mut rs);
        }
        result_set.clear();
        result_set.extend(rs.into_iter());
    }

    fn knn_search_recursive(
        &self,
        node_key: &NodeKey<'_, 'a, P::Data>,
        pivot: &Vector4<P::Data>,
        num: usize,
        depth: usize,
        mut min_distance: Option<P::Data>,
        result_set: &mut Vec<(usize, P::Data)>,
    ) -> Option<P::Data> {
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
            if let Some(min_distance) = min_distance.clone() {
                if distance > min_distance + half_diagonal.clone() {
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
                    for &(index, coords) in content {
                        let distance = (coords - pivot).norm();
                        if min_distance.clone().map_or(true, |d| distance < d) {
                            result_set.push((index, distance));
                        }
                    }

                    result_set.sort_by(|(_, d1), (_, d2)| {
                        d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if result_set.len() > num {
                        result_set.truncate(num);
                    }
                    if result_set.len() == num {
                        min_distance = Some(result_set.last().cloned().unwrap().1);
                    }
                }
            }
        }

        min_distance
    }
}

impl<'a, P: Point> OcTreePcSearch<'a, P>
where
    P::Data: RealField + ToPrimitive,
{
    pub fn radius_search(
        &self,
        pivot: &Vector4<P::Data>,
        radius: P::Data,
        result_set: &mut Vec<(usize, P::Data)>,
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
        node_key: &NodeKey<'_, 'a, P::Data>,
        pivot: &Vector4<P::Data>,
        radius: P::Data,
        depth: usize,
        result_set: &mut Vec<(usize, P::Data)>,
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
                (distance <= radius.clone() + half_diagonal.clone()).then_some(child_nk)
            })
        }) {
            match child.node {
                Node::Branch { .. } => self.radius_search_recursive(
                    &child,
                    pivot,
                    radius.clone(),
                    depth + 1,
                    result_set,
                ),
                Node::Leaf { content } => {
                    for &(index, coords) in content {
                        let distance = (coords - pivot).norm();
                        if distance <= radius {
                            result_set.push((index, distance))
                        }
                    }
                }
            }
        }
    }
}

impl<'a, P: Point> OcTreePcSearch<'a, P>
where
    P::Data: RealField + ToPrimitive,
{
    pub fn new(point_cloud: &'a PointCloud<P>, options: CreateOptions<P::Data>) -> Self {
        OcTreePcSearch {
            point_cloud,
            inner: OcTreePc::new(point_cloud, options, |tree, mul, add| {
                for (index, point) in point_cloud.iter().enumerate() {
                    let key = crate::point_cloud::coords_to_key(point.coords(), mul.clone(), add);
                    let vec = tree.get_or_insert_with(&key, Vec::new);
                    vec.push((index, point.coords()));
                }
            }),
        }
    }
}

impl<'a, P: Point> pcc_common::search::Searcher<'a, P> for OcTreePcSearch<'a, P>
where
    P::Data: RealField + ToPrimitive,
{
    fn point_cloud(&self) -> &'a PointCloud<P> {
        self.point_cloud
    }

    fn search(
        &self,
        pivot: &Vector4<P::Data>,
        ty: SearchType<P::Data>,
        result: &mut Vec<(usize, P::Data)>,
    ) {
        match ty {
            SearchType::Knn(num) => self.knn_search(pivot, num, result),
            SearchType::Radius(radius) => self.radius_search(pivot, radius, result),
        }
    }
}
