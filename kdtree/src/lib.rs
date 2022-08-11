#![feature(type_alias_impl_trait)]

mod node;
mod result;

use std::ptr::NonNull;

use nalgebra::{RealField, Scalar, Vector4};
use node::Node;
use pcc_common::{point_cloud::PointCloud, points::Point3Infoed, search::SearchType};

pub use self::result::*;

pub struct KdTree<'a, T: Scalar, I> {
    point_cloud: &'a PointCloud<Point3Infoed<T, I>>,
    root: Option<NonNull<Node<'a, T>>>,
    indices: Vec<usize>,
}

unsafe impl<'a, T: Send + Scalar, I: Send> Send for KdTree<'a, T, I> {}
unsafe impl<'a, T: Sync + Scalar, I: Sync> Sync for KdTree<'a, T, I> {}

impl<'a, T: RealField, I> KdTree<'a, T, I> {
    pub fn insert(&mut self, index: usize, pivot: &'a Vector4<T>) {
        match self.root {
            Some(mut root) => unsafe { root.as_mut() }.insert(index, pivot),
            None => {
                let node = Box::leak(Box::new(Node::new_leaf(index, pivot)));
                self.root = Some(node.into());
            }
        }
        if self.indices.len() <= index {
            self.indices.resize(index + 1, 0)
        }
        self.indices[index] = index;
    }
}

impl<'a, T: RealField, I> KdTree<'a, T, I> {
    pub fn search_typed(
        &self,
        pivot: &Vector4<T>,
        result: &mut impl ResultSet<Key = T, Value = usize>,
    ) {
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.search(pivot, result)
        }
    }

    pub fn search_exact_typed(
        &self,
        pivot: &Vector4<T>,
        result: &mut impl ResultSet<Key = T, Value = usize>,
    ) {
        if let Some(root) = self.root {
            unsafe { root.as_ref() }.search_exact(pivot, result)
        }
    }
}

impl<'a, T: Scalar, I> Drop for KdTree<'a, T, I> {
    fn drop(&mut self) {
        if let Some(mut root) = self.root {
            unsafe {
                root.as_mut().destroy();
                let _ = Box::from_raw(root.as_ptr());
            }
        }
    }
}

impl<'a, T: RealField, I> KdTree<'a, T, I> {
    pub fn new(point_cloud: &'a PointCloud<Point3Infoed<T, I>>) -> Self {
        assert!(!point_cloud.is_empty());

        let mut indices = (0..point_cloud.len()).collect::<Vec<_>>();
        let root = Node::build(0, point_cloud, &mut indices, None);
        KdTree {
            point_cloud,
            root: Some(root),
            indices,
        }
    }
}

impl<'a, T: RealField, I> pcc_common::search::Searcher<'a, T, I> for KdTree<'a, T, I> {
    fn point_cloud(&self) -> &'a PointCloud<Point3Infoed<T, I>> {
        self.point_cloud
    }

    fn search(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<(usize, T)>) {
        result.clear();
        match ty {
            SearchType::Knn(num) => {
                let mut rs = KnnResultSet::new(num);
                self.search_typed(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(d, v)| (v, d)));
            }
            SearchType::Radius(radius) => {
                let mut rs = RadiusResultSet::new(radius);
                self.search_typed(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(d, v)| (v, d)));
            }
        }
    }

    fn search_exact(&self, pivot: &Vector4<T>, ty: SearchType<T>, result: &mut Vec<(usize, T)>) {
        result.clear();
        match ty {
            SearchType::Knn(num) => {
                let mut rs = KnnResultSet::new(num);
                self.search_exact_typed(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(d, v)| (v, d)));
            }
            SearchType::Radius(radius) => {
                let mut rs = RadiusResultSet::new(radius);
                self.search_exact_typed(pivot, &mut rs);
                result.extend(rs.into_iter().map(|(d, v)| (v, d)));
            }
        }
    }
}
