use std::{
    collections::HashMap,
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
};

use nalgebra::{ComplexField, Scalar, Vector3, Vector4};
use num::Float;
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed};
use petgraph::prelude::UnGraph;

use crate::{node::Node, point_cloud::coords_to_key, CreateOptions, OcTreePc};

#[derive(Debug, Default)]
struct Leaf<'a, L> {
    _data: L,
    num: usize,
    neighbors: Vec<NonNull<Leaf<'a, L>>>,
    _marker: PhantomData<&'a Node<(), L>>,
}

#[derive(Debug)]
pub struct OcTreePcAdjacency<'a, L, T: Scalar> {
    inner: OcTreePc<Leaf<'a, L>, T>,
}

impl<'a, L, T: Scalar + num::Zero> Default for OcTreePcAdjacency<'a, L, T> {
    fn default() -> Self {
        OcTreePcAdjacency {
            inner: Default::default(),
        }
    }
}

impl<'a, L: Default, T: Scalar + Float + ComplexField<RealField = T>> OcTreePcAdjacency<'a, L, T> {
    pub fn from_point_cloud<I>(
        point_cloud: &'a PointCloud<Point3Infoed<T, I>>,
        options: CreateOptions<T>,
    ) -> Self {
        let mut tree = OcTreePcAdjacency {
            inner: OcTreePc::new(point_cloud, options, |tree, mul, add| {
                for point in point_cloud.iter() {
                    let key = coords_to_key(&point.coords, mul, add);
                    let leaf: &mut Leaf<_> = tree.get_or_insert_with(&key, Default::default);
                    leaf.num += 1;
                }
            }),
        };

        // SAFETY: The shadow tree calls `neighbor` function, only affecting `leaf`
        // which is in the same instance.
        unsafe {
            let neigh_shadow = ptr::read(&tree);

            for (key, _, leaf) in tree.inner.depth_iter_mut() {
                neigh_shadow.neighbors(&key, leaf);
            }

            mem::forget(neigh_shadow);
        }

        tree
    }
}

impl<'a, L, T: Scalar> OcTreePcAdjacency<'a, L, T> {
    /// # Safety
    ///
    /// `leaf` must be in the same instance of this tree.
    unsafe fn neighbors(&self, key: &[usize; 3], leaf: &mut Leaf<'a, L>) {
        let key_vec = Vector3::from(*key);
        let min = key_vec.map(|x| x.saturating_sub(1));
        let max = key_vec.map(|x| (x + 1).min(self.inner.max_key()));

        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    if &[x, y, z] == key {
                        continue;
                    }
                    let neighbor = self.inner.get(&[x, y, z]);
                    if let Some(neighbor) = neighbor {
                        leaf.neighbors.push(neighbor.into())
                    }
                }
            }
        }
    }
}

impl<'a, L, T: Scalar + ComplexField<RealField = T> + Copy> OcTreePcAdjacency<'a, L, T> {
    pub fn adjacent_graph(&self) -> UnGraph<Vector4<T>, T> {
        let mut map = HashMap::new();
        let mut graph: UnGraph<Vector4<T>, T> = UnGraph::default();

        for (key, depth, leaf) in self.inner.depth_iter() {
            let center = self.inner.center(&key, depth);
            let vert = graph.add_node(center);
            map.insert(NonNull::from(leaf), vert);
        }

        for (leaf, &vert) in &map {
            let leaf = unsafe { leaf.as_ref() };
            for neighbor in &leaf.neighbors {
                if let Some(&neighbor) = map.get(neighbor) {
                    if graph.find_edge(vert, neighbor).is_none() {
                        let distance = (graph[neighbor] - graph[vert]).norm();
                        graph.add_edge(vert, neighbor, distance);
                    }
                }
            }
        }

        graph
    }
}
