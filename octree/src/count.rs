use nalgebra::{ComplexField, Scalar, Vector4};
use num::{Float, ToPrimitive};
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed};

use crate::{node::Node, point_cloud::coords_to_key, CreateOptions, OcTreePc};

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

impl<T: Scalar + Float + ComplexField<RealField = T>> OcTreePcCount<T> {
    pub fn from_point_cloud<I>(
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
        options: CreateOptions<T>,
    ) -> Self {
        OcTreePcCount {
            inner: OcTreePc::from_point_cloud(point_cloud, options, |tree, mul, add| {
                for point in point_cloud.iter() {
                    let key = coords_to_key(&point.coords, mul, add);
                    *tree.get_or_insert(&key, 0) += 1;
                }
            }),
        }
    }
}

impl<T: Scalar + ComplexField<RealField = T> + ToPrimitive + Copy + PartialOrd>
    OcTreePcCount<T>
{
    pub fn add_coords(&mut self, coords: &Vector4<T>) {
        let key = self.inner.coords_to_key(coords);
        *self.inner.get_or_insert(&key, 0) += 1;
    }

    pub fn count_at(&self, coords: &Vector4<T>) -> usize {
        let key = self.inner.coords_to_key(coords);
        { self.inner.root() }
            .map(|root| {
                let node = root.find(&key, self.inner.depth());
                self.count_recursive(unsafe { node.as_ref() })
            })
            .unwrap_or_default()
    }

    pub fn count(&self) -> usize {
        { self.inner.root() }
            .map(|root| self.count_recursive(root))
            .unwrap_or_default()
    }

    fn count_recursive(&self, node: &Node<(), usize>) -> usize {
        match node {
            Node::Leaf { content } => *content,
            Node::Branch { children, .. } => children
                .iter()
                .flatten()
                .map(|child| self.count_recursive(unsafe { child.as_ref() }))
                .sum(),
        }
    }
}
