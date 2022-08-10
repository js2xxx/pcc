use std::{iter::Sum, ops::Add};

use nalgebra::{ClosedAdd, ComplexField, RealField, Scalar, Vector4};
use num::ToPrimitive;
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed};

use crate::{node::Node, point_cloud::coords_to_key, CreateOptions, OcTreePc};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Leaf<T: Scalar> {
    sum: Vector4<T>,
    count: usize,
}

impl<T: ComplexField> Leaf<T> {
    fn consume(self) -> Vector4<T> {
        self.sum / T::from_usize(self.count).unwrap()
    }
}

impl<T: Scalar + ClosedAdd> Add for Leaf<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Leaf {
            sum: self.sum + rhs.sum,
            count: self.count + rhs.count,
        }
    }
}

impl<T: Scalar + ClosedAdd + num::Zero> Sum for Leaf<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Default::default(), |acc, elem| acc + elem)
    }
}

impl<T: Scalar + num::Zero> Default for Leaf<T> {
    fn default() -> Self {
        Leaf {
            sum: Vector4::zeros(),
            count: 0,
        }
    }
}

#[derive(Debug)]
pub struct OcTreePcCentroid<T: Scalar> {
    inner: OcTreePc<Leaf<T>, T>,
}

impl<T: Scalar + num::Zero> Default for OcTreePcCentroid<T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T: RealField + ToPrimitive> OcTreePcCentroid<T> {
    pub fn from_point_cloud<I>(
        point_cloud: &PointCloud<Point3Infoed<T, I>>,
        options: CreateOptions<T>,
    ) -> Self {
        OcTreePcCentroid {
            inner: OcTreePc::new(point_cloud, options, |tree, mul, add| {
                for point in point_cloud.iter() {
                    let key = coords_to_key(&point.coords, mul.clone(), add);
                    let leaf = tree.get_or_insert_with(&key, Leaf::default);
                    leaf.sum += &point.coords;
                    leaf.count += 1;
                }
            }),
        }
    }
}

impl<T: RealField + ToPrimitive + Copy> OcTreePcCentroid<T> {
    pub fn add_coords(&mut self, coords: &Vector4<T>) {
        let key = self.inner.coords_to_key(coords);
        let leaf = self.inner.get_or_insert_with(&key, Leaf::default);
        leaf.sum += coords;
        leaf.count += 1;
    }

    pub fn count_at(&self, coords: &Vector4<T>) -> Option<Vector4<T>> {
        let key = self.inner.coords_to_key(coords);
        self.inner.root().map(|root| {
            let node = root.find(&key, self.inner.depth());
            self.count_recursive(unsafe { node.as_ref() }).consume()
        })
    }

    pub fn count(&self) -> Option<Vector4<T>> {
        { self.inner.root() }.map(|root| self.count_recursive(root).consume())
    }

    fn count_recursive(&self, node: &Node<(), Leaf<T>>) -> Leaf<T> {
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
